"""Reproducible synthetic-only repetitions and a paired translation control.

All choices are made on unchanged synthetic validation windows. Real evaluation
labels and the reserved synthetic groups are never read by this script.
"""
from pathlib import Path
from dataclasses import asdict
from collections import OrderedDict
import argparse, csv, hashlib, html, json, os, subprocess, sys, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'notebooks/optimized_four_material_training'))
import Optimized_Crack_Engine as e
from Refined_Training import EfficientDataset
from GFRP_Context_Study import ContrastUNet

OUT=ROOT/'outputs/translation_revision/20261111_multiseed_translate10_v3'
DATA=ROOT/'outputs/optimized_crack_training/20261029_nativeW_softcoverage'
PREVIOUS=ROOT/'outputs/translation_revision/20261111_fromscratch_translate10_v2'
MS=['mortar','recycled_mortar','gfrp','sic']
SEEDS=[271,811,1459,2027,4099]
NAMES=dict(zip(MS,['Mortar','Recycled-aggregate mortar','GFRP','SiC composite']))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def js(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)

def rows(path):return list(csv.DictReader(Path(path).open(encoding='utf-8-sig')))

def model_digest(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()

class PairedTranslationDataset(Dataset):
    """Native crop offsets with separate geometry and appearance RNG streams.

    Clamping the unshifted origin to an interior halo is unnecessary: crop the
    final position directly from native arrays. A positive sampling anchor is
    retained after the offset, preserving the intended positive-crop policy.
    """
    def __init__(self,material,cfg,translate=True):
        self.material=material;self.cfg=cfg;self.translate=translate;self.epoch=0
        self.data=DATA/material/'dataset'
        self.rows=[r for r in e.read_rows(self.data/'manifest.csv') if r['split']=='train']
        self.groups={m:[r for r in self.rows if r['mode']==m] for m in ['core','challenge','control']}
        self.scale=json.loads((self.data/'configuration.json').read_text())['generation']['raster_supersampling']**2
        self.cache=OrderedDict()
    def __len__(self):return self.cfg.steps_per_epoch*self.cfg.batch_size
    def arrays(self,row):
        ident=row['id']
        if ident not in self.cache:
            if len(self.cache)>=8:self.cache.popitem(last=False)
            folder=OUT/'native_cache'/self.material/ident
            arrays=tuple(np.load(folder/(k+'.npy'),mmap_mode='r') for k in ['image','coverage','confidence'])
            self.cache[ident]=(arrays,np.argwhere(arrays[1]>0))
        self.cache.move_to_end(ident)
        return self.cache[ident]
    def plan(self,index):
        key=[self.cfg.seed,self.epoch,index]
        rng=np.random.default_rng(np.random.SeedSequence(key+[0]))
        shift=np.random.default_rng(np.random.SeedSequence(key+[1]))
        aug=np.random.default_rng(np.random.SeedSequence(key+[2]))
        mode=str(rng.choice(['core','challenge','control'],p=[.6,.2,.2]))
        group=self.groups[mode];row=group[int(rng.integers(len(group)))];arrays,points=self.arrays(row)
        h,w=arrays[0].shape;p=self.cfg.patch_size;m=int(p*.1);anchor=None
        if len(points) and rng.random()<.75:
            cy,cx=map(int,points[int(rng.integers(len(points)))]);anchor=(cy,cx)
            y=int(rng.integers(max(0,cy-p+1),min(cy,h-p)+1));x=int(rng.integers(max(0,cx-p+1),min(cx,w-p)+1))
        else:y=int(rng.integers(h-p+1));x=int(rng.integers(w-p+1))
        original=(y,x);requested=(0,0)
        if shift.random()<.5:requested=tuple(map(int,shift.integers(-m,m+1,size=2)))
        dy,dx=requested if self.translate else (0,0)
        low_y,high_y,low_x,high_x=0,h-p,0,w-p
        if anchor is not None:
            cy,cx=anchor;low_y,high_y=max(low_y,cy-p+1),min(high_y,cy)
            low_x,high_x=max(low_x,cx-p+1),min(high_x,cx)
        y=int(np.clip(y+dy,low_y,high_y));x=int(np.clip(x+dx,low_x,high_x))
        return row,arrays,aug,original,(y,x),anchor,requested
    def __getitem__(self,index):
        row,arrays,rng,original,(y,x),anchor,requested=self.plan(index);p=self.cfg.patch_size
        sl=np.s_[y:y+p,x:x+p]
        a=arrays[0][sl].astype(np.float32).copy()
        b=arrays[1][sl].astype(np.float32)/self.scale;c=arrays[2][sl].astype(np.float32)/255
        k=int(rng.integers(4));a,b,c=[np.rot90(z,k) for z in [a,b,c]]
        if rng.random()<.5:a,b,c=[z[:,::-1] for z in [a,b,c]]
        a=a.copy()
        if rng.random()<.7:a=np.clip(a,0,1)**rng.uniform(.85,1.15);a=a*rng.uniform(.9,1.1)+rng.uniform(-.03,.03)
        if rng.random()<.5:a+=rng.normal(0,rng.uniform(0,.02),a.shape)
        return dict(image=torch.from_numpy(np.ascontiguousarray(np.clip(a,0,1)[None]*2-1)),
            label=torch.from_numpy(np.ascontiguousarray(b)),weight=torch.from_numpy(np.ascontiguousarray(c)))

def protocol():
    return dict(seeds=SEEDS,materials=MS,runs_per_material=5,initialisation='Independent random initialisation; no previous weights loaded.',
        synthetic_only=True,real_labels_used=False,reserved_test_used=False,source_code_sha256=sha(Path(__file__)),
        training=dict(patch_size=256,batch_size=4,steps_per_epoch=20,learning_rate=.001,max_epochs=260,min_epochs=80,
            patience=30,min_delta=.0001,lr_patience=7,lr_factor=.5,min_lr=1e-5,post_lr_grace_epochs=15,num_threads=2),
        sampling='60/20/20 core/challenge/control; positive anchor with probability 0.75 when available; anchor retained after translation.',
        augmentation='Native integer crop offset up to +/-25 pixels with probability 0.5, paired right-angle rotations and horizontal flip; original moderate gamma/gain/offset/noise.',
        padding=False,interpolation=False,translation_stream_separate=True,appearance_stream_separate=True,
        stopping_and_selection_metric='Maximise fixed synthetic-validation soft_area_dice at threshold 0.5; min_delta 0.0001.',
        selection_ties='Higher validation Area Dice, then lower validation loss, then smaller seed.',
        selection_scope='All five runs are retained and summarised; selected deployment checkpoint is separate from the five-run mean and SD.',
        control='GFRP seed271 has a matched no-translation control; initial state, base crop stream, appearance stream, validation windows and optimisation rules match.',
        changes_to_diagnose='Sampler removes forced interior halo origins and preserves the selected positive pixel. Learning-rate reductions receive a grace period before stopping. These are development changes, not isolated translation effects versus v2.',
        manuscript_updates_automatic=False)

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    p=protocol();dest=OUT/'PROTOCOL.json'
    if dest.exists():assert json.loads(dest.read_text())==p,'Use a new study directory if code or protocol changes.'
    else:js(dest,p)
    (OUT/'code_snapshot').mkdir(exist_ok=True)
    import shutil
    shutil.copy2(Path(__file__),OUT/'code_snapshot'/Path(__file__).name)
    for m in MS:
        data=DATA/m/'dataset';rr=e.read_rows(data/'manifest.csv');train=[r for r in rr if r['split']=='train'];val=[r for r in rr if r['split']=='validation']
        prior=json.loads((PREVIOUS/m/'protocol.json').read_text())
        assert [r['id'] for r in train]==prior['train_ids'] and [r['id'] for r in val]==prior['validation_ids']
        assert not({r['background_group'] for r in train}&{r['background_group'] for r in val})
        for i,r in enumerate(train):
            folder=OUT/'native_cache'/m/r['id'];folder.mkdir(parents=True,exist_ok=True)
            marker=folder/'COMPLETE.json'
            if marker.exists():continue
            info={}
            for k in ['image','coverage','confidence']:
                source=data/r[k];a=e.tifffile.imread(source);target=folder/(k+'.npy');np.save(target,a)
                assert np.array_equal(a,np.load(target,mmap_mode='r'))
                info[k]=dict(source_sha256=sha(source),array_sha256=sha(target),shape=list(a.shape),dtype=str(a.dtype))
            js(marker,info)
        print('PREPARED',m,len(train),'exact native training arrays',flush=True)
    checks=[]
    for m in MS:
        cfg=e.TrainingConfig(seed=SEEDS[0]);on=PairedTranslationDataset(m,cfg,True);off=PairedTranslationDataset(m,cfg,False)
        on.epoch=off.epoch=1;active=0
        for i in range(32):
            a=on.plan(i);b=off.plan(i)
            assert a[0]['id']==b[0]['id'] and a[3]==b[3] and a[5:]==b[5:]
            assert np.array_equal(a[2].random(8),b[2].random(8))
            oy,ox=a[3];y,x=a[4];assert abs(oy-y)<=25 and abs(ox-x)<=25
            if a[5] is not None:
                cy,cx=a[5];assert y<=cy<y+256 and x<=cx<x+256
            z=on[i];assert all(z[k].shape[-2:]==(256,256) for k in ['image','label','weight'])
            assert z['image'].dtype==z['label'].dtype==z['weight'].dtype==torch.float32
            assert set(torch.unique(z['weight']).tolist())<={0.,1.}
            assert torch.allclose(z['label']*16,(z['label']*16).round())
            active+=a[3]!=a[4]
        checks.append(dict(material=m,samples=32,nonzero_offsets=active,paired_streams=True,positive_anchors_retained=True))
        del on,off
    js(OUT/'TECHNICAL_CHECKS.json',dict(result='pass',checks=checks))
    page()

def train(material,seed,translate=True):
    assert (OUT/'TECHNICAL_CHECKS.json').exists()
    pr=json.loads((OUT/'PROTOCOL.json').read_text());assert pr==protocol()
    cfg=e.TrainingConfig(seed=seed,learning_rate=.001,max_epochs=260,min_epochs=80,patience=30,num_threads=2,export_real_count=0)
    torch.set_num_threads(2);e.seed_everything(seed)
    dest=OUT/('runs' if translate else 'controls')/material/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'summary.json').exists():return
    model=ContrastUNet(cfg) if material=='gfrp' else e.make_model(cfg);initial=model_digest(model)
    data=DATA/material/'dataset';rr=e.read_rows(data/'manifest.csv');val=[r for r in rr if r['split']=='validation']
    tr=PairedTranslationDataset(material,cfg,translate);va=EfficientDataset.create(e,data,val,cfg,False)
    assert va.fixed==json.loads((PREVIOUS/material/'protocol.json').read_text())['validation_windows']
    tl=DataLoader(tr,batch_size=cfg.batch_size,num_workers=0);vl=DataLoader(va,batch_size=cfg.batch_size,num_workers=0)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=7,min_lr=1e-5,threshold=.0001,threshold_mode='abs')
    js(dest/'protocol.json',dict(material=material,seed=seed,translate=translate,config=asdict(cfg),initial_state_sha256=initial,
        study_protocol_sha256=sha(OUT/'PROTOCOL.json'),validation_windows=va.fixed,source_dataset_sha256=sha(data/'COMPLETE.json')))
    best=-1.;best_epoch=0;best_metrics=None;bad=0;history=[];start_epoch=1;last_reduce=-1000
    last=dest/'last_model.pt'
    if last.exists():
        ck=torch.load(last,map_location='cpu',weights_only=False);assert ck['initial_state_sha256']==initial
        model.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler'])
        best,best_epoch,best_metrics,bad,history,last_reduce=(ck[k] for k in ['best','best_epoch','best_metrics','bad','history','last_reduce'])
        start_epoch=ck['epoch']+1;torch.set_rng_state(ck['torch_rng'])
    started=time.time();reason='max_epochs'
    for epoch in range(start_epoch,cfg.max_epochs+1):
        tr.epoch=epoch;a=e.run_epoch(model,tl,torch.device('cpu'),.5,opt);b=e.run_epoch(model,vl,torch.device('cpu'),.5)
        score=b['soft_area_dice']
        if score>best+.0001:
            best=score;best_epoch=epoch;best_metrics=b;bad=0
            tmp=dest/'best_model.tmp';torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},config=asdict(cfg),epoch=epoch,
                validation=b,material=material,architecture='contrast256' if material=='gfrp' else 'raw',seed=seed,initial_state_sha256=initial),tmp);tmp.replace(dest/'best_model.pt')
        else:bad+=1
        lr=opt.param_groups[0]['lr'];sched.step(score)
        if opt.param_groups[0]['lr']<lr:last_reduce=epoch
        history.append(dict(epoch=epoch,train_loss=a['loss'],**{'val_'+k:v for k,v in b.items()},lr=lr,lr_next=opt.param_groups[0]['lr'],last_lr_reduction_epoch=last_reduce))
        e.write_rows(dest/'history.csv',history)
        status=dict(material=material,seed=seed,translate=translate,epoch=epoch,best_epoch=best_epoch,area_dice=score,best_area_dice=best,stale=bad,
            lr=opt.param_groups[0]['lr'],last_lr_reduction_epoch=last_reduce,elapsed_seconds=time.time()-started)
        js(dest/'STATUS.json',status)
        tmp=last.with_suffix('.tmp');torch.save(dict(state_dict=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),epoch=epoch,
            best=best,best_epoch=best_epoch,best_metrics=best_metrics,bad=bad,history=history,last_reduce=last_reduce,initial_state_sha256=initial,torch_rng=torch.get_rng_state()),tmp);tmp.replace(last)
        print(material,seed,'translation' if translate else 'control',epoch,'area',round(score,4),'best',best_epoch,'stale',bad,flush=True)
        if epoch>=cfg.min_epochs and bad>=cfg.patience and epoch-last_reduce>=15:reason='early_stopping';break
    # Independently reload the selected weight file and recompute the stored scores.
    ck=torch.load(dest/'best_model.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['state_dict'])
    recomputed=e.run_epoch(model,vl,torch.device('cpu'),.5)
    for k,v in best_metrics.items():assert np.isclose(v,recomputed[k],atol=1e-7,rtol=1e-7),(k,v,recomputed[k])
    js(dest/'summary.json',dict(material=material,seed=seed,translate=translate,initial_state_sha256=initial,validation=recomputed,best_epoch=best_epoch,
        epochs=len(history),stop_reason=reason,duration_seconds=time.time()-started,checkpoint_sha256=sha(dest/'best_model.pt'),recomputed=True,
        real_labels_used=False,reserved_test_used=False))

def page():
    display=[];done=0
    for m in MS:
        for seed in SEEDS:
            d=OUT/'runs'/m/f'seed{seed}'
            try:s=json.loads((d/'STATUS.json').read_text())
            except (FileNotFoundError,json.JSONDecodeError):s={}
            complete=(d/'summary.json').exists();done+=complete
            display.append([NAMES[m],seed,'Complete' if complete else 'Training' if s else 'Queued',s.get('epoch','—'),s.get('best_epoch','—'),f"{s['best_area_dice']:.4f}" if s else '—'])
    body=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in display)
    (OUT/'training_status.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="20"><title>Model training progress</title><style>body{font:16px/1.7 system-ui;max-width:1100px;margin:30px auto}table{border-collapse:collapse;width:100%;border-top:2px solid;border-bottom:2px solid}td,th{padding:7px;text-align:left}th{border-bottom:1px solid}</style><h1>Model training progress</h1><p>Complete '+str(done)+'/20. Synthetic data only, trained from random initialisation using the original fixed validation windows. All seed results are retained; see PROTOCOL.json for the actual training and selection rules. This process has not replaced model or manuscript files.</p><table><tr><th>Material</th><th>Seed</th><th>Status</th><th>Epoch</th><th>Selected epoch</th><th>Validation Area Dice</th></tr>'+body+'</table>')

def finish():
    import shutil
    table=[];selected={};stats=[]
    for m in MS:
        results=[json.loads((OUT/'runs'/m/f'seed{s}'/'summary.json').read_text()) for s in SEEDS]
        for r in results:table.append(dict(material=m,seed=r['seed'],epochs=r['epochs'],selected_epoch=r['best_epoch'],stop_reason=r['stop_reason'],**r['validation']))
        winner=min(results,key=lambda r:(-r['validation']['soft_area_dice'],r['validation']['loss'],r['seed']))
        source=OUT/'runs'/m/f"seed{winner['seed']}";dest=OUT/'selected'/m;dest.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/'best_model.pt',dest/'best_model.pt');shutil.copy2(source/'history.csv',dest/'history.csv');shutil.copy2(source/'summary.json',dest/'summary.json')
        selected[m]=dict(seed=winner['seed'],checkpoint=str((dest/'best_model.pt').relative_to(ROOT)),checkpoint_sha256=sha(dest/'best_model.pt'),validation=winner['validation'])
        area=np.array([r['validation']['soft_area_dice'] for r in results])
        stats.append(dict(material=m,n=5,area_dice_mean=float(area.mean()),area_dice_sd=float(area.std(ddof=1)),area_dice_min=float(area.min()),area_dice_max=float(area.max()),selected_seed=winner['seed']))
    control=json.loads((OUT/'controls/gfrp/seed271/summary.json').read_text());treated=json.loads((OUT/'runs/gfrp/seed271/summary.json').read_text())
    assert control['initial_state_sha256']==treated['initial_state_sha256']
    js(OUT/'PAIRED_TRANSLATION_CONTROL.json',dict(material='gfrp',seed=271,same_initial_state=True,translation=treated['validation'],no_translation=control['validation'],
        area_dice_difference=treated['validation']['soft_area_dice']-control['validation']['soft_area_dice'],interpretation='One paired development run; not a multi-seed causal effect estimate.'))
    e.write_rows(OUT/'all_run_results.csv',table);e.write_rows(OUT/'five_seed_summary.csv',stats);js(OUT/'SELECTED_MODELS.json',selected)
    js(OUT/'TRAINING_COMPLETE.json',dict(models=20,control_models=1,recomputed=True,real_selection=False,manuscript_updated=False))
    page();print(json.dumps(stats,indent=2),flush=True)

def supervise():
    assert json.loads((OUT/'TECHNICAL_CHECKS.json').read_text())['result']=='pass'
    # Stagger material/seed order while allowing at most four two-thread workers.
    jobs=[(m,seed,True) for seed in SEEDS for m in MS]
    jobs.insert(4,('gfrp',SEEDS[0],False))
    active=[];(OUT/'logs').mkdir(exist_ok=True)
    while jobs or active:
        while jobs and len(active)<4:
            m,seed,translate=jobs.pop(0);d=OUT/('runs' if translate else 'controls')/m/f'seed{seed}'
            if (d/'summary.json').exists():continue
            command=[sys.executable,str(Path(__file__)),'--worker','--material',m,'--seed',str(seed)]
            if not translate:command+=['--no-translation']
            log=(OUT/'logs'/f'{m}_{seed}_{"translation" if translate else "control"}.log').open('a')
            proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            active.append((proc,log,m,seed,translate));print('STARTED',m,seed,translate,proc.pid,flush=True)
        remaining=[]
        for proc,log,m,seed,translate in active:
            code=proc.poll()
            if code is None:remaining.append((proc,log,m,seed,translate))
            else:
                log.close()
                if code:
                    js(OUT/'ERROR.json',dict(material=m,seed=seed,translate=translate,exit_code=code))
                    for other,*_ in remaining:other.terminate()
                    raise RuntimeError(f'Worker {m} {seed} failed; see logs.')
        active=remaining;page();js(OUT/'SUPERVISOR_STATUS.json',dict(pid=os.getpid(),active=[dict(pid=p.pid,material=m,seed=s,translate=t) for p,l,m,s,t in active],queued=len(jobs)))
        if jobs or active:time.sleep(5)
    finish()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true');parser.add_argument('--worker',action='store_true')
    parser.add_argument('--material',choices=MS);parser.add_argument('--seed',type=int,choices=SEEDS);parser.add_argument('--no-translation',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    if args.worker:train(args.material,args.seed,not args.no_translation)
    if args.run:supervise()
