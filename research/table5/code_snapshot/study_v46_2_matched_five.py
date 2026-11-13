"""Fresh MPS-only five-seed real14/virtual/virtual+real14,7,4 comparison."""
from pathlib import Path
import os,sys,json,csv,time,random,argparse,shutil,fcntl,traceback
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import numpy as np
import torch
import tifffile
from torch.utils.data import Dataset,DataLoader
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'notebooks/optimized_four_material_training'))
sys.path.insert(0,str(ROOT/'scripts'))
import Optimized_Crack_Engine as e
from GFRP_Context_Study import ContrastUNet
import train_translation_multiseed as baseline

OUT=ROOT/'outputs/v46_2/20261112_matched14_val6_five_seed_v1'
ANNOT=ROOT/'outputs/real_annotation_validation/20261110_v44_frozen_v1'
CACHE=ROOT/'outputs/real_fewshot_adaptation/20261110_fixed6_v1/native_cache'
MS=baseline.MS;NAMES=baseline.NAMES;SEEDS=[271,811,1459,2027,4099]
VAL=[2,5,8,12,16,19];ARMS=['virtual','real14','virtual_real14','virtual_real7','virtual_real4']
sha=baseline.sha;js=baseline.js;read=baseline.rows

def directory(m,arm,seed):return OUT/'runs'/m/arm/f'seed{seed}'
def scores(c):
    tp,fp,fn,tn=[int(c[k]) for k in ['tp','fp','fn','tn']]
    fg=tp/(tp+fp+fn) if tp+fp+fn else 0.;bg=tn/(tn+fp+fn) if tn+fp+fn else 0.
    return dict(**{k:int(v) for k,v in c.items()},crack_iou=fg,background_iou=bg,miou=(fg+bg)/2,
                dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.)

def require_mps():
    assert torch.backends.mps.is_available(),'MPS inaccessible here. Run local MPS-only launcher in native Terminal; CPU fallback is disabled.'
    torch.set_num_threads(1)
    model=torch.nn.Conv2d(1,4,3,padding=1).to('mps');x=torch.randn(4,1,256,256,device='mps')
    opt=torch.optim.AdamW(model.parameters());loss=model(x).square().mean();loss.backward();opt.step();torch.mps.synchronize()
    assert torch.isfinite(loss) and x.device.type=='mps'
    result=dict(torch=torch.__version__,backend='mps',allocation_forward_backward_optimizer=True,recommended_max_bytes=torch.mps.recommended_max_memory())
    del opt,model,x,loss;torch.mps.empty_cache();return result

def spread_order(rows):
    pool=sorted(rows,key=lambda r:int(r['ordinal']));result=[]
    while pool:
        if not result:index=(len(pool)-1)//2
        else:index=max(range(len(pool)),key=lambda i:(min(abs(int(pool[i]['ordinal'])-int(r['ordinal'])) for r in result),-int(pool[i]['ordinal'])))
        result.append(pool.pop(index))
    return result

def nested_order(rows):
    pos=spread_order([r for r in rows if r['positive']]);neg=spread_order([r for r in rows if not r['positive']]);npositive=len(pos);result=[];used=0
    for n in range(1,len(rows)+1):
        want=max(1,round(n*npositive/len(rows))) if npositive else 0
        if pos and (not neg or used<want):result.append(pos.pop(0));used+=1
        else:result.append(neg.pop(0))
    return result

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'PREPARATION_COMPLETE.json').exists():
        assert json.loads((OUT/'PROTOCOL.json').read_text())['source_code_sha256']==sha(__file__)
        return
    runtime=require_mps();manifest=read(ANNOT/'evaluation_manifest.csv');assert len(manifest)==80
    sources={};windows={};splits=[];labels=[];audit=[]
    for m in MS:
        group=sorted([r for r in manifest if r['material']==m],key=lambda r:int(r['ordinal']))
        real=[];validation=[];windows[m]=[]
        for r in group:
            assert sha(ROOT/r['image'])==r['image_sha256'] and sha(ROOT/r['label_01'])==r['label_01_sha256']
            imagepath=CACHE/r['id']/'image.npy';a=np.load(imagepath,mmap_mode='r');b=tifffile.imread(ROOT/r['label_01']).astype(np.uint8)
            assert a.shape==b.shape and set(np.unique(b))<={0,1}
            lp=OUT/'real_cache'/r['id']/'label.npy';lp.parent.mkdir(parents=True,exist_ok=True)
            if lp.exists():assert np.array_equal(np.load(lp,mmap_mode='r'),b)
            else:np.save(lp,b)
            item=dict(id=r['id'],ordinal=int(r['ordinal']),image=str(imagepath),label=str(lp),scale=1,positive=bool(b.any()),image_sha256=sha(imagepath),label_sha256=sha(lp))
            if item['ordinal'] not in VAL:real.append(item)
            else:
                validation.append(item);h,w=b.shape;yy,xx=np.where(b);origins=[]
                if len(yy):
                    for q in [.125,.375,.625,.875]:
                        i=min(len(yy)-1,int(q*len(yy)));origins.append([int(np.clip(yy[i]-128,0,h-256)),int(np.clip(xx[i]-128,0,w-256))])
                    positions=[(.25,.25),(.25,.75),(.75,.25),(.75,.75)]
                else:positions=[(.2,.2),(.2,.5),(.2,.8),(.5,.2),(.5,.8),(.8,.2),(.8,.5),(.8,.8)]
                origins += [[int(np.clip(round(y*h)-128,0,h-256)),int(np.clip(round(x*w)-128,0,w-256))] for y,x in positions]
                assert len(origins)==8;windows[m]+=[dict(id=r['id'],image=str(imagepath),label=str(lp),y=y,x=x) for y,x in origins]
            labels.append(dict(**r,archived_label=str(ROOT/r['label_01']),native_input=str(imagepath),native_input_sha256=item['image_sha256'],label_array=str(lp),label_array_sha256=item['label_sha256']))
        assert len(real)==14 and len(validation)==6 and len(windows[m])==48
        order=nested_order(real);subsets={str(n):order[:n] for n in [4,7,14]}
        synth=[]
        for row in read(baseline.DATA/m/'dataset/manifest.csv'):
            if row['split']!='train':continue
            folder=baseline.OUT/'native_cache'/m/row['id'];done=json.loads((folder/'COMPLETE.json').read_text())
            item=dict(id=row['id'],mode=row['mode'],image=str(folder/'image.npy'),label=str(folder/'coverage.npy'),weight=str(folder/'confidence.npy'),scale=16)
            for key,filename in [('image','image'),('label','coverage'),('weight','confidence')]:
                assert sha(item[key])==done[filename]['array_sha256']
                assert sha(baseline.DATA/m/'dataset'/row[filename])==done[filename]['source_sha256']
                item[key+'_sha256']=done[filename]['array_sha256']
            synth.append(item)
        assert len(synth)==60
        sources[m]=dict(real14=order,subsets=subsets,validation=validation,virtual=synth)
        for item in order+validation:
            ordinal=item['ordinal'];splits.append(dict(material=m,id=item['id'],ordinal=ordinal,split='validation6' if ordinal in VAL else 'fitting14',positive=item['positive'],in_fit7=item['id'] in {r['id'] for r in subsets['7']},in_fit4=item['id'] in {r['id'] for r in subsets['4']}))
        audit.append(dict(material=m,fit14=14,fit14_positive=sum(r['positive'] for r in order),fit7_positive=sum(r['positive'] for r in subsets['7']),fit4_positive=sum(r['positive'] for r in subsets['4']),validation6=6,validation_windows=48,virtual_pairs=len(synth)))
    protocol=dict(version='V46.2 matched14-val6 five-seed study v1',annotation_date='2026-11-10',seeds=SEEDS,materials=MS,arms=ARMS,training_runs=100,
        fixed_validation_ordinals=VAL,fitting='All remaining14realimages used byreal14andvirtual_real14; nested fixed7and4subsets, identical acrossseeds; no additional two-image holdout.',
        independence='Five independently initialised virtual pretrainings/material. Finetuning seedS starts only from its own virtualseedS; paired real14 has identical random initialstate to virtualseedS.',
        virtual_source=str(baseline.DATA),virtual_meaning='Exact data/network/sampler family of the previous V46.2pre-adaptation base256 arm. OriginalV45renderer retained; no displacement or new rendering. Synthetic-gradient-only, but common real validation labels DO select checkpoints/schedule/stopping.',
        common=dict(patch_size=256,batch_size=4,updates_per_epoch=20,weight_decay=.0001,loss='Original CE+foreground softDice; realbinarytargets weight1, virtualfractionaltargets original validitymask.',augmentation='Paired nativeintegertranslation upto25px, rotations/flips, gamma/gain/offset andnoise; originalV46.2samplerpolicy.',normalisation='Exact existing specimen-envelope0.5–99.5percentile nativecache; GFRPlocalchannels9and31; othergroupssinglechannel.',threshold=.5),
        from_scratch=dict(learning_rate=.001,max_epochs=260,min_epochs=80,patience=30),finetune=dict(learning_rate=.0001,max_epochs=120,min_epochs=40,patience=20),
        convergence='Allarms use identical common6realvalidationwindows. Scheduler/stopping monitor continuous global48-window CE+Dice loss, min_delta1e-4,LRplateaupatience7,factor0.5,15epochpost-reductiongrace. Checkpoint highest two-classpooledwindowmIoU,delta1e-4,epoch0eligible.',
        final_validation='Frozen bestcheckpoint, fullnative6images,Hannoverlap256/stride128,threshold0.5,noROI/referenceclipping orpostprocessing. Two-classmIoU pooledacrossall6fullimages, foreground/backgroundIoUalsoreported.',
        trimmed_statistics='USERFIXED:sort5full-imagepooledmIoUsby(score,seed),droponehighestandonelowest,reportmiddle3meanandsampleSD(ddof1). Thisistrimming,notoutlierdetection. Alsoall5mean/sampleSDandallseeds. No failedrunomission.',
        limits='Siximagesnowexplicitlyusedforvalidation/checkpointselection; notindependenttest. Previouslyinspected correlatedsame-acquisitionslices. Equalupdatecapswithineachstage,not equal totalcompute acrosspretrain+finetune versusrealonly.',
        source_code_sha256=sha(__file__),annotation_manifest_sha256=sha(ANNOT/'evaluation_manifest.csv'),sources_sha256='',dependencies={str(Path(x).relative_to(ROOT)):sha(x) for x in [e.__file__,baseline.__file__,ROOT/'notebooks/optimized_four_material_training/GFRP_Context_Study.py']})
    js(OUT/'sources.json',sources);js(OUT/'validation_windows.json',windows);e.write_rows(OUT/'fixed_split.csv',splits);e.write_rows(OUT/'label_manifest.csv',labels);protocol['sources_sha256']=sha(OUT/'sources.json')
    js(OUT/'PROTOCOL.json',protocol);js(OUT/'MPS_RUNTIME.json',runtime)
    protected=[ROOT/'output/manuscript/CURRENT_RELEASE.json',*sorted((ROOT/'output/manuscript').glob('*v45.docx')),ROOT/'outputs/v46_2/20261112_controlled_search_v1/FINAL_COMPLETE.json']
    js(OUT/'PROTECTED.json',{str(p.relative_to(ROOT)):sha(p) for p in protected})
    snapshot=OUT/'code_snapshot';snapshot.mkdir(exist_ok=True)
    for p in [Path(__file__),Path(e.__file__),Path(baseline.__file__),ROOT/'notebooks/optimized_four_material_training/GFRP_Context_Study.py']:shutil.copy2(p,snapshot/p.name)
    js(OUT/'PREPARATION_COMPLETE.json',dict(result='passed',materials=audit,all14used_no_extra_holdout=True,subsets_nested=True,validation_disjoint=True,backend='mps',originals_unchanged=True))
    print('PREPARED100freshMPSruns;Nov10labels;14/7/4fit;6commonvalidation;usertrimhighest/lowest.',flush=True)

class RealStream(Dataset):
    def __init__(self,items,seed):self.items=items;self.seed=seed;self.epoch=0;self.cache={}
    def __len__(self):return 80
    def __getitem__(self,i):
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,i,0]));shift=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,i,1]));aug=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,i,2]))
        row=self.items[int(rng.integers(len(self.items)))];ident=row['id']
        if ident not in self.cache:
            if len(self.cache)>=4:self.cache.pop(next(iter(self.cache)))
            a=np.load(row['image'],mmap_mode='r');b=np.load(row['label'],mmap_mode='r');self.cache[ident]=(a,b,np.argwhere(b>0))
        a,b,pos=self.cache[ident];h,w=a.shape;p=256;anchor=None
        if len(pos) and rng.random()<.75:
            cy,cx=map(int,pos[int(rng.integers(len(pos)))]);anchor=(cy,cx);y=int(rng.integers(max(0,cy-p+1),min(cy,h-p)+1));x=int(rng.integers(max(0,cx-p+1),min(cx,w-p)+1))
        else:y=int(rng.integers(h-p+1));x=int(rng.integers(w-p+1))
        dy,dx=shift.integers(-25,26,2) if shift.random()<.5 else (0,0);ly,hy,lx,hx=0,h-p,0,w-p
        if anchor is not None:
            cy,cx=anchor;ly,hy=max(ly,cy-p+1),min(hy,cy);lx,hx=max(lx,cx-p+1),min(hx,cx)
        y=int(np.clip(y+dy,ly,hy));x=int(np.clip(x+dx,lx,hx));a=a[y:y+p,x:x+p].astype(np.float32).copy();b=b[y:y+p,x:x+p].astype(np.float32)
        k=int(aug.integers(4));a,b=np.rot90(a,k),np.rot90(b,k)
        if aug.random()<.5:a,b=a[:,::-1],b[:,::-1]
        a=a.copy()
        if aug.random()<.7:a=np.clip(a,0,1)**aug.uniform(.85,1.15);a=a*aug.uniform(.9,1.1)+aug.uniform(-.03,.03)
        if aug.random()<.5:a+=aug.normal(0,aug.uniform(0,.02),a.shape)
        return dict(image=torch.from_numpy(np.ascontiguousarray(np.clip(a,0,1)[None]*2-1)),label=torch.from_numpy(np.ascontiguousarray(b)),weight=torch.ones((p,p)))

def validation_loader(m):
    data=[]
    for r in json.loads((OUT/'validation_windows.json').read_text())[m]:
        y,x=r['y'],r['x'];a=np.load(r['image'],mmap_mode='r')[y:y+256,x:x+256].astype(np.float32);b=np.load(r['label'],mmap_mode='r')[y:y+256,x:x+256].astype(np.float32)
        data.append(dict(image=torch.from_numpy(a[None]*2-1),label=torch.from_numpy(b),weight=torch.ones_like(torch.from_numpy(b))))
    return DataLoader(data,batch_size=4,num_workers=0)

def validate(model,loader):
    model.eval();outputs=[];targets=[]
    with torch.inference_mode():
        for batch in loader:outputs.append(model(batch['image'].to('mps')));targets.append(batch['label'].to('mps'))
        logits=torch.cat(outputs);truth=torch.cat(targets);loss=e.segmentation_loss(logits,truth,torch.ones_like(truth));p=logits.softmax(1)[:,1]>=.5;t=truth>=.5
        c=dict(tp=int((p&t).sum()),fp=int((p&~t).sum()),fn=int((~p&t).sum()),tn=int((~p&~t).sum()))
        assert torch.isfinite(loss);return dict(loss=float(loss),**scores(c))

def train_epoch(model,loader,optimizer):
    model.train();total=torch.zeros((),device='mps');n=0
    for b in loader:
        x=b['image'].to('mps');y=b['label'].to('mps');w=b['weight'].to('mps');optimizer.zero_grad(set_to_none=True);z=model(x);loss=e.segmentation_loss(z,y,w)
        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();total+=loss.detach()*len(x);n+=len(x)
    return float(total/n)

def atomic_torch(p,value):
    tmp=p.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(p)

def train(m,arm,seed):
    dest=directory(m,arm,seed);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'SUMMARY.json').exists():return
    protocol=json.loads((OUT/'PROTOCOL.json').read_text());assert protocol['source_code_sha256']==sha(__file__)
    require_mps();e.seed_everything(seed);fine=arm.startswith('virtual_real');settings=protocol['finetune' if fine else 'from_scratch']
    cfg=e.TrainingConfig(seed=seed,device='mps',patch_size=256,num_threads=1,export_real_count=0,max_epochs=settings['max_epochs'],min_epochs=settings['min_epochs'],patience=settings['patience'],learning_rate=settings['learning_rate'])
    sources=json.loads((OUT/'sources.json').read_text())[m]
    model=(ContrastUNet(cfg) if m=='gfrp' else e.make_model(cfg)).to('mps');random_initial=baseline.model_digest(model);pretrained=None
    if fine:
        parent=directory(m,'virtual',seed);assert (parent/'SUMMARY.json').exists();pretrained=parent/'best_model.pt'
        assert sha(pretrained)==json.loads((parent/'SUMMARY.json').read_text())['checkpoint_sha256']
        ck=torch.load(pretrained,map_location='mps',weights_only=False);model.load_state_dict(ck['state_dict'])
    initial=baseline.model_digest(model)
    if arm=='virtual':
        train_data=baseline.PairedTranslationDataset(m,cfg,True)
        for row in sources['virtual']:
            for k in ['image','label','weight']:assert sha(row[k])==row[k+'_sha256']
        fitting=[]
    else:
        budget=14 if arm=='real14' else int(arm.replace('virtual_real',''));fitting=sources['subsets'][str(budget)];train_data=RealStream(fitting,seed)
        for row in fitting:assert sha(row['image'])==row['image_sha256'] and sha(row['label'])==row['label_sha256']
    tl=DataLoader(train_data,batch_size=4,num_workers=0);vl=validation_loader(m)
    fitids=[r['id'] for r in fitting];valids=[r['id'] for r in sources['validation']];assert not set(fitids)&set(valids)
    runprotocol=dict(material=m,arm=arm,seed=seed,backend='mps',config=vars(cfg),random_initial_state_sha256=random_initial,initial_state_sha256=initial,pretrained_checkpoint_sha256=sha(pretrained) if pretrained else None,fitting_ids=fitids,validation_ids=valids,gradient_real_labels=arm!='virtual',real_labels_used_for_selection=True,real_gradient_images=len(fitting),settings=settings,protocol_sha256=sha(OUT/'PROTOCOL.json'))
    js(dest/'RUN_PROTOCOL.json',runprotocol)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=7,min_lr=1e-6 if fine else 1e-5,threshold=.0001,threshold_mode='abs')
    last=dest/'last_model.pt';history=[];start=1;bad=0;last_reduce=-1000;reason='max_epochs'
    if last.exists():
        state=torch.load(last,map_location='mps',weights_only=False);assert state['initial_state_sha256']==initial
        model.load_state_dict(state['state_dict']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        best,best_epoch,best_metrics,best_loss,bad,last_reduce,history=[state[k] for k in ['best','best_epoch','best_metrics','best_loss','bad','last_reduce','history']];start=state['epoch']+1
        torch.set_rng_state(state['torch_rng'].cpu());torch.mps.set_rng_state(state['mps_rng'].cpu());random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
    else:
        best_metrics=validate(model,vl);best=best_metrics['miou'];best_loss=best_metrics['loss'];best_epoch=0
        atomic_torch(dest/'best_model.pt',dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},config=vars(cfg),material=m,arm=arm,seed=seed,epoch=0,validation=best_metrics))
    for epoch in range(start,settings['max_epochs']+1):
        if (OUT/'STOP_REQUESTED').exists():return False
        t=time.monotonic();train_data.epoch=epoch;loss=train_epoch(model,tl,optimizer);val=validate(model,vl)
        if val['miou']>best+.0001:
            best=val['miou'];best_epoch=epoch;best_metrics=val
            atomic_torch(dest/'best_model.pt',dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},config=vars(cfg),material=m,arm=arm,seed=seed,epoch=epoch,validation=val))
        if val['loss']<best_loss-.0001:best_loss=val['loss'];bad=0
        else:bad+=1
        lr=optimizer.param_groups[0]['lr'];scheduler.step(val['loss'])
        if optimizer.param_groups[0]['lr']<lr:last_reduce=epoch
        history.append(dict(epoch=epoch,train_loss=loss,val_loss=val['loss'],val_miou=val['miou'],val_crack_iou=val['crack_iou'],val_background_iou=val['background_iou'],lr=lr,seconds=time.monotonic()-t))
        e.write_rows(dest/'history.csv',history)
        atomic_torch(last,dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,best=best,best_epoch=best_epoch,best_metrics=best_metrics,best_loss=best_loss,bad=bad,last_reduce=last_reduce,history=history,initial_state_sha256=initial,torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state()))
        js(dest/'STATUS.json',dict(state='training',material=m,arm=arm,seed=seed,epoch=epoch,best_epoch=best_epoch,best_window_miou=best,loss_stale=bad,backend='mps',allocated_bytes=torch.mps.current_allocated_memory(),driver_bytes=torch.mps.driver_allocated_memory(),updated_unix=time.time()))
        print(m,arm,seed,epoch,'windowmIoU',round(best,6),'loss',round(val['loss'],5),flush=True)
        if epoch>=settings['min_epochs'] and bad>=settings['patience'] and epoch-last_reduce>=15:reason='early_stopping';break
    state=torch.load(dest/'best_model.pt',map_location='mps',weights_only=False);model.load_state_dict(state['state_dict']);recheck=validate(model,vl)
    for k in ['loss','miou','dice']:assert np.isclose(recheck[k],best_metrics[k],rtol=2e-5,atol=2e-5),(k,recheck,best_metrics)
    js(dest/'TRAINING_COMPLETE.json',dict(material=m,arm=arm,seed=seed,epochs=len(history),best_epoch=best_epoch,best_window_miou=best,checkpoint_sha256=sha(dest/'best_model.pt'),backend='mps',stop_reason=reason))
    js(dest/'STATUS.json',dict(state='full_image_validation',material=m,arm=arm,seed=seed,epoch=len(history),backend='mps',updated_unix=time.time()))
    per=[];probabilities=[]
    for row in sources['validation']:
        image=np.load(row['image'],mmap_mode='r');truth=np.load(row['label'],mmap_mode='r').astype(bool);folder=dest/'validation_predictions'/row['id'];folder.mkdir(parents=True,exist_ok=True);pp=folder/'probability.tiff';marker=folder/'COMPLETE.json'
        signature=dict(checkpoint_sha256=sha(dest/'best_model.pt'),input_sha256=row['image_sha256'],label_sha256=row['label_sha256'],patch_size=256,stride=128,backend='mps')
        if marker.exists():
            old=json.loads(marker.read_text());assert old['signature']==signature and sha(pp)==old['probability_sha256'];prob=tifffile.imread(pp)
        else:
            with torch.inference_mode():prob=e.predict_native(model,image,torch.device('mps'),cfg)
            assert prob.shape==truth.shape and np.isfinite(prob).all() and prob.min()>=0 and prob.max()<=1
            tifffile.imwrite(pp,prob.astype(np.float32),compression='zlib',compressionargs={'level':1},metadata=None);js(marker,dict(signature=signature,probability_sha256=sha(pp)))
        pred=prob>=.5;hist=np.bincount((2*pred.astype(np.uint8)+truth.astype(np.uint8)).ravel(),minlength=4);c=dict(zip(['tp','fp','fn','tn'],[int(hist[i]) for i in [3,2,1,0]]))
        per.append(dict(material=m,arm=arm,seed=seed,id=row['id'],ordinal=row['ordinal'],**scores(c)))
        probabilities.append(dict(id=row['id'],probability=str(pp.relative_to(ROOT)),probability_sha256=sha(pp),input_sha256=row['image_sha256'],label_sha256=row['label_sha256']))
        print('FULLVALID',m,arm,seed,row['id'],flush=True)
    c={k:sum(r[k] for r in per) for k in ['tp','fp','fn','tn']};e.write_rows(dest/'per_image_metrics.csv',per);e.write_rows(dest/'probability_manifest.csv',probabilities)
    js(dest/'SUMMARY.json',dict(material=m,material_label=NAMES[m],arm=arm,seed=seed,epochs=len(history),best_epoch=best_epoch,stop_reason=reason,reached_cap=len(history)==settings['max_epochs'],backend='mps',real_gradient_images=len(fitting),real_validation_images=6,pretrained_checkpoint_sha256=sha(pretrained) if pretrained else None,checkpoint_sha256=sha(dest/'best_model.pt'),initial_state_sha256=initial,random_initial_state_sha256=random_initial,full_image_validation=dict(images=6,**scores(c))))
    js(dest/'STATUS.json',dict(state='complete',material=m,arm=arm,seed=seed,epoch=len(history),full_image_miou=scores(c)['miou'],updated_unix=time.time()))
    del model,optimizer,scheduler,tl,vl,train_data;torch.mps.empty_cache();return True

def benchmark(steps):
    require_mps();cfg=e.TrainingConfig(device='mps');model=ContrastUNet(cfg).to('mps');opt=torch.optim.AdamW(model.parameters());x=torch.rand(4,1,256,256,device='mps');y=torch.zeros(4,256,256,device='mps');y[:,80:90,:]=1
    for i in range(3):opt.zero_grad();loss=e.segmentation_loss(model(x),y,torch.ones_like(y));loss.backward();opt.step()
    torch.mps.synchronize();t=time.monotonic()
    for i in range(steps):opt.zero_grad();loss=e.segmentation_loss(model(x),y,torch.ones_like(y));loss.backward();opt.step()
    torch.mps.synchronize();print(json.dumps(dict(steps=steps,seconds=time.monotonic()-t,driver_bytes=torch.mps.driver_allocated_memory(),allocated_bytes=torch.mps.current_allocated_memory())),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--material',choices=MS);p.add_argument('--arm',choices=ARMS);p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--benchmark',type=int);a=p.parse_args()
    if a.prepare:prepare()
    if a.benchmark:benchmark(a.benchmark)
    if a.material:
        folder=directory(a.material,a.arm,a.seed);folder.mkdir(parents=True,exist_ok=True)
        with (folder/'worker.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:train(a.material,a.arm,a.seed)
            except Exception as ex:
                js(folder/'ERROR.json',dict(error=type(ex).__name__,message=str(ex),traceback=traceback.format_exc(),updated_unix=time.time()));raise

if __name__=='__main__':main()
