"""MPS-only matched generator ablations; safe negatives use fitting labels explicitly."""
from pathlib import Path
import os,sys,json,time,random,argparse,fcntl,traceback,shutil
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import tifffile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import study_v46_2_matched_five as old
import generate_v46_2_unified as gen
OUT=gen.OUT;MS=old.MS;SEEDS=old.SEEDS;NAMES=old.NAMES;js=old.js;sha=old.sha
SPECS={
 'legacy':dict(negative=.20,safe=False,geometry=False,grey=False),
 'safe20':dict(negative=.20,safe=True,geometry=False,grey=False),
 'geometry':dict(negative=.20,safe=True,geometry=True,grey=False),
 'grey':dict(negative=.20,safe=True,geometry=False,grey=True),
 'negative35':dict(negative=.35,safe=True,geometry=False,grey=False),
 'combined':dict(negative=.35,safe=True,geometry=True,grey=True),
}

def directory(m,arm,seed):return OUT/'runs'/m/arm/f'seed{seed}'

class VirtualStream(Dataset):
    def __init__(self,m,arm,seed,batch):
        self.m=m;self.spec=SPECS[arm];self.seed=seed;self.epoch=0;self.batch=batch;self.cache={};self.points={};self.counts={}
        self.base=json.loads((old.OUT/'sources.json').read_text())[m]['virtual'];self.ext=json.loads((OUT/'data'/m/'records.json').read_text());self.neg=json.loads((OUT/'data'/m/'negatives.json').read_text())
    def __len__(self):return 20*self.batch
    def arrays(self,row):
        if row['id'] not in self.cache:
            if len(self.cache)>=6:
                key=next(iter(self.cache));self.cache.pop(key)
            self.cache[row['id']]=tuple(np.load(row[k],mmap_mode='r') for k in ['image','label','weight'])
        return self.cache[row['id']]
    def __getitem__(self,i):
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,i,0]));aug=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,i,2]));sp=self.spec
        negative=rng.random()<sp['negative']
        if negative and sp['safe']:
            hard=rng.random()<.5;pool=[r for r in self.neg if r['hard']==hard] or self.neg;row=pool[int(rng.integers(len(pool)))];a=np.array(np.load(row['image'],mmap_mode='r'),np.float32);b=np.zeros((256,256),np.float32);w=np.ones_like(b)
            key='confirmed_negative'
        else:
            if negative:pool=[r for r in self.base if r['mode']=='control'];key='legacy_control'
            else:
                extended=sp['geometry'] and rng.random()<.5
                if extended:pool=[r for r in self.ext if r['kind']==('joint' if sp['grey'] else 'geometry')];key='new_geometry'
                elif sp['grey']:pool=[r for r in self.ext if r['kind']=='appearance'];key='new_grey'
                else:
                    mode='core' if rng.random()<.75 else 'challenge';pool=[r for r in self.base if r['mode']==mode];key='base_positive_source'
            row=pool[int(rng.integers(len(pool)))];a0,b0,w0=self.arrays(row);h,width=a0.shape;p=256
            if row['label'] not in self.points:self.points[row['label']]=np.argwhere(b0>0)
            points=self.points[row['label']];anchor=None
            if len(points) and rng.random()<.75:
                cy,cx=map(int,points[int(rng.integers(len(points)))]);anchor=(cy,cx)
                y=int(rng.integers(max(0,cy-p+1),min(cy,h-p)+1));x=int(rng.integers(max(0,cx-p+1),min(cx,width-p)+1))
            else:y=int(rng.integers(h-p+1));x=int(rng.integers(width-p+1))
            shift=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,i,1]));dy,dx=shift.integers(-25,26,2) if shift.random()<.5 else (0,0)
            ly,hy,lx,hx=0,h-p,0,width-p
            if anchor is not None:
                cy,cx=anchor;ly,hy=max(ly,cy-p+1),min(hy,cy);lx,hx=max(lx,cx-p+1),min(hx,cx)
            y=int(np.clip(y+dy,ly,hy));x=int(np.clip(x+dx,lx,hx));sl=np.s_[y:y+256,x:x+256]
            a=np.array(a0[sl],np.float32);b=np.array(b0[sl],np.float32)/row['scale'];w=np.array(w0[sl],np.float32)/255
        self.counts[key]=self.counts.get(key,0)+1;self.counts['empty_patches']=self.counts.get('empty_patches',0)+int(not b.any());self.counts['total']=self.counts.get('total',0)+1
        k=int(aug.integers(4));a,b,w=[np.rot90(z,k) for z in [a,b,w]]
        if aug.random()<.5:a,b,w=[z[:,::-1] for z in [a,b,w]]
        a=a.copy()
        if aug.random()<.7:a=np.clip(a,0,1)**aug.uniform(.85,1.15);a=a*aug.uniform(.9,1.1)+aug.uniform(-.03,.03)
        if aug.random()<.5:a+=aug.normal(0,aug.uniform(0,.02),a.shape)
        return dict(image=torch.from_numpy(np.ascontiguousarray(np.clip(a,0,1)[None]*2-1)),label=torch.from_numpy(np.ascontiguousarray(b)),weight=torch.from_numpy(np.ascontiguousarray(w)))

class RealStream(old.RealStream):
    def __init__(self,items,seed,batch):
        super().__init__(items,seed);self.batch=batch
        # Fourteen readonly memory maps are cheap; retaining sparse coordinates
        # avoids rescanning a complete label on nearly every random crop.
        for row in items:
            a=np.load(row['image'],mmap_mode='r');b=np.load(row['label'],mmap_mode='r')
            self.cache[row['id']]=(a,b,np.argwhere(b>0))
    def __len__(self):return 20*self.batch

def prepare():
    if (OUT/'PROTOCOL.json').exists():return
    compute=json.loads((OUT/'COMPUTE_CHOICE.json').read_text());runtime=old.require_mps()
    for m in MS:assert (OUT/'data'/m/'COMPLETE.json').exists()
    deps=[Path(__file__),Path(gen.__file__),Path(old.__file__),Path(old.baseline.__file__),Path(old.e.__file__),ROOT/'notebooks/optimized_four_material_training/GFRP_Context_Study.py']
    protocol=dict(version='V46.2 unified generator experiment v1',date='2026-11-13',backend='mps',batch=compute['batch'],parallel=compute['parallel'],patch_size=256,updates_per_epoch=20,
        specs=SPECS,seeds=SEEDS,annotation_date='2026-11-10',validation=old.VAL,split='Same14fitting/6validationaspreviousstudy; validationselectsmodels,notindependenttest.',
        fitting_negative_labels='safe20/geometry/grey/negative35/combined use archived fitting14 masks to select reliable empty patches. Explicit annotation-assisted sampling; not label-free virtual training.',
        baseline='legacy keeps original injected-geometry supervision including existing source-defect limitations; safe20 isolates replacement of original controls with fitting-label-supported negatives.',
        pilot_selection='Seed271: compare geometry/grey/negative35/combined foreground IoU per material. Highest crackIoU selected, ties by listed order. Freeze before seeds811/1459/2027/4099. All pilots retained.',
        confirmation='legacy,safe20,selected candidate,real14 five seeds each material. Adapt legacy,safe20,selected with all14realimages and own seed checkpoint; all5+middle3 statistics.',
        scratch=dict(learning_rate=.001,max_epochs=260,min_epochs=80,patience=30),finetune=dict(learning_rate=.0001,max_epochs=260,min_epochs=40,patience=20),
        convergence='Original continuous validationCE+Dice schedule, best validation-window pooledmIoUcheckpoint. Fitting-stage caps differ from old120fine protocol; all new adaptations share260.',
        final_inference='Fullsixnativeimages;Hann256/stride128;threshold0.5;noROIcleanup;foreground/backgroundIoUandmIoU.',
        compute='Common largerbatch andsame20updates/epoch acrossnewarms. More samples/update than oldbatch4; old100table is historical,notidenticalbudget.',
        limits='Same-specimen/load-correlated development images; adaptive generator selection uses6validation. No external test, no annotation-savings claim. Geometry-source supervision is inserted occupancy,notcomplete pre-existing-damage masks.',
        dependencies={str(p.relative_to(ROOT)):sha(p) for p in deps},data={m:sha(OUT/'data'/m/'COMPLETE.json') for m in MS},source_manifest_sha256=sha(old.OUT/'sources.json'),runtime=runtime)
    js(OUT/'PROTOCOL.json',protocol)
    for p in deps:
        target=OUT/'code_snapshot'/p.name;target.parent.mkdir(exist_ok=True);shutil.copy2(p,target)

def train(m,arm,seed):
    dest=directory(m,arm,seed);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'SUMMARY.json').exists():return
    protocol=json.loads((OUT/'PROTOCOL.json').read_text())
    for path,digest in protocol['dependencies'].items():assert sha(ROOT/path)==digest,path
    old.require_mps();old.e.seed_everything(seed);fine=arm.startswith('adapt_');sourcearm=arm[6:] if fine else arm;settings=protocol['finetune' if fine else 'scratch'];batch=protocol['batch']
    cfg=old.e.TrainingConfig(seed=seed,device='mps',patch_size=256,batch_size=batch,num_threads=1,export_real_count=0,max_epochs=settings['max_epochs'],min_epochs=settings['min_epochs'],patience=settings['patience'],learning_rate=settings['learning_rate'])
    sources=json.loads((old.OUT/'sources.json').read_text())[m];model=(old.ContrastUNet(cfg) if m=='gfrp' else old.e.make_model(cfg)).to('mps');random_initial=old.baseline.model_digest(model);parent=None
    if fine:
        parent=directory(m,sourcearm,seed)/'best_model.pt';parent_summary=json.loads((parent.parent/'SUMMARY.json').read_text());assert sha(parent)==parent_summary['checkpoint_sha256'];model.load_state_dict(torch.load(parent,map_location='mps',weights_only=False)['state_dict'])
    initial=old.baseline.model_digest(model);real=fine or arm=='real14';items=sources['real14'] if real else []
    stream=RealStream(items,seed,batch) if real else VirtualStream(m,arm,seed,batch)
    loader=DataLoader(stream,batch_size=batch,num_workers=0);vl=old.validation_loader(m)
    assisted=(sourcearm!='legacy' and sourcearm!='real14');assert not {r['id'] for r in items}&{r['id'] for r in sources['validation']}
    js(dest/'RUN_PROTOCOL.json',dict(material=m,arm=arm,seed=seed,backend='mps',config=vars(cfg),settings=settings,random_initial_state_sha256=random_initial,initial_state_sha256=initial,pretrained_checkpoint_sha256=sha(parent) if parent else None,
        fitting_ids=[r['id'] for r in items],negative_label_source_ids=[r['id'] for r in sources['real14']] if assisted else [],gradient_real_positive_labels=real,annotation_assisted_negative_sampling=assisted,protocol_sha256=sha(OUT/'PROTOCOL.json')))
    opt=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=.0001)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=7,min_lr=1e-6 if fine else 1e-5,threshold=.0001,threshold_mode='abs')
    last=dest/'last_model.pt';history=[];start=1;bad=0;last_reduce=-1000;reason='max_epochs';began=time.time()
    if last.exists():
        state=torch.load(last,map_location='mps',weights_only=False);assert state['initial_state_sha256']==initial;model.load_state_dict(state['state_dict']);opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler'])
        best,best_epoch,best_metrics,best_loss,bad,last_reduce,history=[state[k] for k in ['best','best_epoch','best_metrics','best_loss','bad','last_reduce','history']];start=state['epoch']+1
        torch.set_rng_state(state['torch_rng'].cpu());torch.mps.set_rng_state(state['mps_rng'].cpu());random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
        if hasattr(stream,'counts'):stream.counts=state.get('sampling_counts',{})
    else:
        best_metrics=old.validate(model,vl);best=best_metrics['miou'];best_loss=best_metrics['loss'];best_epoch=0
        old.atomic_torch(dest/'best_model.pt',dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},config=vars(cfg),material=m,arm=arm,seed=seed,epoch=0,validation=best_metrics))
    training_marker=dest/'TRAINING_COMPLETE.json'
    if not training_marker.exists():
        for epoch in range(start,settings['max_epochs']+1):
            if (OUT/'STOP_REQUESTED').exists():return False
            t=time.monotonic();stream.epoch=epoch;loss=old.train_epoch(model,loader,opt);val=old.validate(model,vl)
            if val['miou']>best+.0001:
                best=val['miou'];best_epoch=epoch;best_metrics=val;old.atomic_torch(dest/'best_model.pt',dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},config=vars(cfg),material=m,arm=arm,seed=seed,epoch=epoch,validation=val))
            if val['loss']<best_loss-.0001:best_loss=val['loss'];bad=0
            else:bad+=1
            lr=opt.param_groups[0]['lr'];sched.step(val['loss'])
            if opt.param_groups[0]['lr']<lr:last_reduce=epoch
            history.append(dict(epoch=epoch,train_loss=loss,val_loss=val['loss'],val_miou=val['miou'],val_crack_iou=val['crack_iou'],val_background_iou=val['background_iou'],lr=lr,seconds=time.monotonic()-t))
            old.e.write_rows(dest/'history.csv',history)
            old.atomic_torch(last,dict(state_dict=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),epoch=epoch,best=best,best_epoch=best_epoch,best_metrics=best_metrics,best_loss=best_loss,bad=bad,last_reduce=last_reduce,history=history,initial_state_sha256=initial,
                torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),sampling_counts=getattr(stream,'counts',{})))
            js(dest/'STATUS.json',dict(state='training',epoch=epoch,best_epoch=best_epoch,best_window_miou=best,loss_stale=bad,backend='mps',batch=batch,allocated_bytes=torch.mps.current_allocated_memory(),driver_bytes=torch.mps.driver_allocated_memory(),updated_unix=time.time()))
            if epoch%10==0:print(m,arm,seed,epoch,'best',round(best,5),flush=True)
            if epoch>=settings['min_epochs'] and bad>=settings['patience'] and epoch-last_reduce>=15:reason='early_stopping';break
        js(training_marker,dict(epochs=len(history),best_epoch=best_epoch,stop_reason=reason,backend='mps',finished_unix=time.time(),checkpoint_sha256=sha(dest/'best_model.pt'),sampling_counts=getattr(stream,'counts',{})))
    marker=json.loads(training_marker.read_text());reason=marker['stop_reason'];state=torch.load(dest/'best_model.pt',map_location='mps',weights_only=False);model.load_state_dict(state['state_dict']);recheck=old.validate(model,vl)
    assert np.isclose(recheck['miou'],state['validation']['miou'],atol=2e-5)
    js(dest/'STATUS.json',dict(state='full_image_validation',epoch=marker['epochs'],updated_unix=time.time()))
    per=[];maps=[]
    for row in sources['validation']:
        image=np.load(row['image'],mmap_mode='r');truth=np.load(row['label'],mmap_mode='r').astype(bool);folder=dest/'validation_predictions'/row['id'];folder.mkdir(parents=True,exist_ok=True);pp=folder/'probability.tiff';complete=folder/'COMPLETE.json'
        sig=dict(checkpoint_sha256=sha(dest/'best_model.pt'),input_sha256=row['image_sha256'],label_sha256=row['label_sha256'],patch_size=256,stride=128,backend='mps')
        if complete.exists():
            previous=json.loads(complete.read_text());assert previous['signature']==sig and sha(pp)==previous['probability_sha256'];prob=tifffile.imread(pp)
        else:
            with torch.inference_mode():prob=old.e.predict_native(model,image,torch.device('mps'),cfg)
            assert prob.shape==truth.shape and np.isfinite(prob).all() and prob.min()>=0 and prob.max()<=1
            tifffile.imwrite(pp,prob.astype(np.float32),compression='zlib',compressionargs={'level':1},metadata=None);js(complete,dict(signature=sig,probability_sha256=sha(pp)))
        pred=prob>=.5;hist=np.bincount((2*pred.astype(np.uint8)+truth.astype(np.uint8)).ravel(),minlength=4);c=dict(zip(['tp','fp','fn','tn'],[int(hist[i]) for i in [3,2,1,0]]))
        per.append(dict(material=m,arm=arm,seed=seed,id=row['id'],ordinal=row['ordinal'],**old.scores(c)));maps.append(dict(id=row['id'],path=str(pp),sha256=sha(pp)))
    c={k:sum(r[k] for r in per) for k in ['tp','fp','fn','tn']};old.e.write_rows(dest/'per_image_metrics.csv',per);js(dest/'probability_manifest.json',maps)
    js(dest/'SUMMARY.json',dict(material=m,arm=arm,seed=seed,epochs=marker['epochs'],best_epoch=marker['best_epoch'],stop_reason=reason,reached_cap=marker['epochs']==settings['max_epochs'],backend='mps',batch=batch,updates=20*marker['epochs'],real_gradient_images=len(items),annotation_assisted_negatives=assisted,real_validation_images=6,checkpoint_sha256=sha(dest/'best_model.pt'),initial_state_sha256=initial,random_initial_state_sha256=random_initial,pretrained_checkpoint_sha256=sha(parent) if parent else None,full_image_validation=dict(images=6,**old.scores(c)),sampling_counts=marker.get('sampling_counts',{}),finished_unix=time.time(),invocation_seconds=time.time()-began))
    js(dest/'STATUS.json',dict(state='complete',epoch=marker['epochs'],full_image_miou=old.scores(c)['miou'],updated_unix=time.time()))
    print('COMPLETE',m,arm,seed,old.scores(c)['miou'],flush=True)

def budget(seed):
    folder=OUT/'budget_diagnostic';folder.mkdir(exist_ok=True)
    source=old.OUT;dest=folder/'runs/gfrp/virtual'/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    for name in ['SUMMARY.json','best_model.pt']:
        target=dest/name
        if not target.exists():os.link(source/'runs/gfrp/virtual'/f'seed{seed}'/name,target)
    for name in ['sources.json','validation_windows.json']:
        target=folder/name
        if not target.exists():shutil.copy2(source/name,target)
    protocol=json.loads((source/'PROTOCOL.json').read_text());protocol['finetune']['max_epochs']=260
    protocol['version']='V46.2 GFRP fine-cap diagnostic, old batch4 and source, fresh fine-tuning from same pretrain'
    if not (folder/'PROTOCOL.json').exists():js(folder/'PROTOCOL.json',protocol)
    old.OUT=folder;old.train('gfrp','virtual_real14',seed)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--material',choices=MS);p.add_argument('--arm');p.add_argument('--seed',type=int);p.add_argument('--budget',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    elif a.budget:budget(a.seed)
    elif a.material:
        dest=directory(a.material,a.arm,a.seed);dest.mkdir(parents=True,exist_ok=True)
        with (dest/'worker.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:train(a.material,a.arm,a.seed)
            except Exception as ex:js(dest/'ERROR.json',dict(error=type(ex).__name__,message=str(ex),traceback=traceback.format_exc(),time=time.time()));raise
