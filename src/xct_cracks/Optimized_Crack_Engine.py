# %% 1. Imports and explicit experiment configuration
"""Optimized native-W XCT synthesis with partial-volume supervision and four independent U-Nets.

Image-domain approximation, not a reconstruction or calibrated fracture simulator.
Synthetic truth describes injected geometry, not every existing defect.
"""
from pathlib import Path
from dataclasses import dataclass, asdict, replace
from functools import lru_cache
import csv, json, hashlib, random, time, os, sys, copy
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
os.environ.setdefault('MPLCONFIGDIR','/tmp/xct-rewrite-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/xct-rewrite-cache')
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage import morphology
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import BasicUNet
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


def find_project():
    """Resolve a user-configured project without requiring research data on import."""
    configured = os.environ.get('XCT_PROJECT_ROOT')
    return Path(configured).expanduser().resolve() if configured else Path(__file__).resolve().parents[2]

ENGINE_FILE=Path(__file__).resolve()
PROJECT=find_project()
from . import xct_preview_io as xio
from .robust_structure_cracks import normalize_image, structural_prior, Settings as RouteSettings
from .image_guided_cracks import generate_geometry
MATERIALS=('mortar','recycled_mortar','gfrp','sic')
_source_data = Path(os.environ.get('XCT_SOURCE_DATA', 'datasets/prepared/four_materials_100train_10real_v1')).expanduser()
SOURCE_DATA = (_source_data if _source_data.is_absolute() else PROJECT / _source_data).resolve()

@dataclass(frozen=True)
class GenerationConfig:
    seed: int=20261028
    backgrounds: int=20
    variants: int=5  # 3 resolved/core + 1 difficult + 1 no-injection control
    core_width_px: tuple=(0.8,8.0)
    core_width_bands: tuple=((0.8,2.0),(2.0,4.0),(4.0,8.0))
    challenge_width_px: tuple=(0.6,2.5)
    core_mixing: tuple=(0.65,0.98)
    challenge_mixing: tuple=(0.40,0.80)
    psf_sigma_px: tuple=(0.55,1.10)
    dark_percentile: float=1.0
    planning_stride: int=1  # only route planning; image pixels never resampled
    raster_supersampling: int=4
    raster_tile: int=256
    n_paths: tuple=(1,2)
    branch_probability: float=0.35
    positive_weight_floor: float=0.0  # superseded: binary validity, not confidence reweighting

@dataclass(frozen=True)
class TrainingConfig:
    seed: int=20261028
    patch_size: int=256
    batch_size: int=4
    steps_per_epoch: int=20
    max_epochs: int=120
    min_epochs: int=40
    patience: int=20
    learning_rate: float=1e-3
    threshold: float=0.5
    features: tuple=(8,8,16,32,64,8)
    validation_windows_per_image: int=4
    device: str='mps'  # portable neural entry points require Apple MPS
    num_threads: int=4
    export_real_count: int=1  # unlabelled real inference, no accuracy claimed
    evaluate_test: bool=False  # enable only after freezing the full workflow

# Same pixel interval is a detectability convention, NOT the same physical crack.
# Image-dependent dark reference and noise/texture differ for each acquisition.
MATERIAL_NOTES={
    'mortar':'Pores and existing reference defects are not exhaustively labelled.',
    'recycled_mortar':'Aggregate/paste interfaces can mimic injected cracks.',
    'gfrp':'Dark inter-fibre texture can be indistinguishable from damage in one slice.',
    'sic':'Dark phases and phase-contrast fringes confound cracks; voxel pitch disputed.'}


def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False))

def read_rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def write_rows(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_tiff(path,array):tifffile.imwrite(path,array,compression='zlib',compressionargs={'level':1})

# %% 2. Full-slice routing and bounded geometry

def route_paths(original,specimen,seed,mode,cfg):
    """Native structural attraction, uniform endpoints, no gray-boundary repulsion.

    W is image structure evidence, not measured mechanical weakness. The retained
    8-neighbor geodesic discretization can still introduce directional bias.
    """
    field=cached_native_field(original,specimen)
    domain=ndi.binary_erosion(specimen,iterations=6)
    settings=RouteSettings(scales_px=(1.,2.,4.),guidance_strength=3.,
        n_paths=cfg.n_paths,branch_probability=cfg.branch_probability,length_fraction=(.18,.45))
    rng=np.random.default_rng(np.random.SeedSequence([seed,11]))
    routing=dict(field,specimen=domain,prior=domain.astype(np.float32),
        cost=np.where(domain,1/(1+3*field['prior']),np.inf))
    paths=generate_geometry(routing,settings,rng)
    if not paths:raise RuntimeError('No native route; no silent straight-line fallback.')
    variant=(seed-cfg.seed)%100
    low,high=cfg.core_width_bands[variant%3] if mode=='core' else cfg.challenge_width_px
    wrng=np.random.default_rng(np.random.SeedSequence([seed,12]));result=[]
    for path in paths:
        points=path['points'].astype(float)
        distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
        at=np.linspace(0,distance[-1],max(8,int(np.ceil(distance[-1]*cfg.raster_supersampling))+1))
        points=np.column_stack([np.interp(at,distance,points[:,k]) for k in (0,1)])
        nominal=np.exp(wrng.uniform(np.log(low),np.log(high)))
        knots=wrng.uniform(.75,1.25,9);knots[[0,-1]]=.6
        factor=ndi.gaussian_filter1d(np.interp(np.linspace(0,1,len(points)),np.linspace(0,1,9),knots),max(1,len(points)/100))
        widths=np.clip(nominal*factor,low,high).astype(np.float32)
        if not specimen[tuple(np.rint(points).astype(int).T)].all():raise ValueError('Route left specimen')
        result.append(dict(points=points.astype(np.float32),widths=widths,kind=path['kind']))
    return result


def rasterize_tiled(paths,shape,specimen,cfg):
    """Supersampled variable-radius support, bounded memory at native tile borders."""
    s,t=cfg.raster_supersampling,cfg.raster_tile
    max_width=max(float(p['widths'].max()) for p in paths)
    halo=int(np.ceil(max_width/2))+3
    coverage=np.zeros(shape,np.float32);width=np.zeros(shape,np.float32)
    all_points=np.concatenate([p['points'] for p in paths]);all_widths=np.concatenate([p['widths'] for p in paths])
    for y in range(0,shape[0],t):
        for x in range(0,shape[1],t):
            h,w=min(t,shape[0]-y),min(t,shape[1]-x)
            selected=(all_points[:,0]>=y-halo)&(all_points[:,0]<y+h+halo)&(all_points[:,1]>=x-halo)&(all_points[:,1]<x+w+halo)
            if not selected.any():continue
            center=np.zeros(((h+2*halo)*s,(w+2*halo)*s),bool)
            apertures=np.zeros(center.shape,np.float32)
            ij=np.rint((all_points[selected]-[y-halo,x-halo])*s+(s-1)/2).astype(int)
            good=(ij[:,0]>=0)&(ij[:,0]<center.shape[0])&(ij[:,1]>=0)&(ij[:,1]<center.shape[1])
            ij=ij[good];ww=all_widths[selected][good]
            if not len(ij):continue
            center[ij[:,0],ij[:,1]]=True
            np.maximum.at(apertures,(ij[:,0],ij[:,1]),ww)
            distance,nearest=ndi.distance_transform_edt(~center,return_indices=True)
            local_width=apertures[nearest[0],nearest[1]]
            support=distance<=local_width*s/2
            sl=(slice(halo*s,(halo+h)*s),slice(halo*s,(halo+w)*s))
            cov=support[sl].reshape(h,s,w,s).mean((1,3))
            coverage[y:y+h,x:x+w]=cov
            width[y:y+h,x:x+w]=local_width[sl][s//2::s,s//2::s]*(cov>0)
    coverage*=specimen;width*=coverage>0
    return coverage,coverage>0,width

# %% 3. Image-conditioned grayscale, PSF and explicit confidence diagnostics

def render_crack(original,specimen,coverage,seed,mode,cfg,dark_reference):
    """Blur the injected perturbation; do not blur an already reconstructed image again.

    I_out = clip(I_original - PSF*(coverage*mixing*max(I_local-I_void,0)),0,1).
    This retains measured background noise. No label-correlated extra noise is added.
    Dark reference is a low-tail heuristic unless replaced by measured calibration.
    """
    rng=np.random.default_rng(np.random.SeedSequence([seed,13]))
    mixing=float(rng.uniform(*(cfg.core_mixing if mode=='core' else cfg.challenge_mixing)))
    sigma=float(rng.uniform(*cfg.psf_sigma_px))
    local=ndi.gaussian_filter(original,.5)
    decrement=ndi.gaussian_filter(coverage*mixing*np.maximum(local-dark_reference,0),sigma)*specimen
    virtual=np.clip(original-decrement,0,1).astype(np.float32)
    drop=original-virtual
    residual=original-ndi.gaussian_filter(original,1.)
    texture=np.sqrt(np.maximum(ndi.gaussian_filter(residual**2,2.),0))
    ratio=drop/(texture+.005)  # texture proxy, not calibrated scanner-noise SNR
    quality=np.clip(ratio/2.,0,1)*np.clip(drop/.04,0,1)
    confidence=np.ones(original.shape,np.float32)
    positive=coverage>0
    # This return is a placeholder; prepare_material supplies explicit binary validity.
    hh=ndi.convolve(original,np.array([[1.,-1.],[-1.,1.]])/2.)
    hh=hh[specimen];noise=float(np.median(np.abs(hh-np.median(hh)))/.67448975)
    return virtual,drop,confidence,dict(dark_reference=float(dark_reference),mixing=mixing,
        psf_sigma_px=sigma,hh_mad_noise_proxy=noise,weak_gray_fraction=float(((drop<.025)|(ratio<1))[positive].mean()) if positive.any() else None,
        confidence_policy='binary validity from counterfactual ambiguity; no positive confidence scaling')

# %% 4. Reproducible preparation: generate complete native slices before patching

def prepare_material(material,output,cfg=GenerationConfig()):
    output=Path(output)
    if material not in MATERIALS:raise ValueError(material)
    requested=dict(material=material,generation=asdict(cfg),source_manifest_sha256=sha(SOURCE_DATA/material/'train/manifest.csv'),
        engine_sha256=sha(ENGINE_FILE))
    requested=json.loads(json.dumps(requested))
    if (output/'COMPLETE.json').exists():
        done=json.loads((output/'COMPLETE.json').read_text())
        if done['configuration']!=requested:raise ValueError('Configuration/source changed; choose a new output directory')
        verify_dataset(output);return output
    if output.exists() and any(output.iterdir()):raise FileExistsError('Use a fresh directory; incomplete output is never overwritten')
    if cfg.backgrounds<3 or cfg.backgrounds>20 or cfg.variants<3:raise ValueError('Require3-20 background groups and at least3 variants')
    if cfg.planning_stride<1 or cfg.raster_supersampling<1:raise ValueError('Invalid spatial sampling')
    output.mkdir(parents=True);(output/'backgrounds').mkdir();(output/'samples').mkdir()
    source=read_rows(SOURCE_DATA/material/'train/manifest.csv')
    groups=sorted({r['background_group'] for r in source})
    groups=[groups[i] for i in np.rint(np.linspace(0,len(groups)-1,cfg.backgrounds)).astype(int)]
    # Random group split; all variants of each background stay together.
    order=list(np.random.default_rng(cfg.seed).permutation(groups))
    nv=max(1,int(round(len(groups)*.2)));nt=nv
    assignments={g:('validation' if g in order[:nv] else 'test' if g in order[nv:nv+nt] else 'train') for g in groups}
    write_json(output/'configuration.json',requested);rows=[]
    for bi,g in enumerate(groups):
        record=next(r for r in source if r['background_group']==g)
        raw,header=xio.read_plane(record);support,support_info=xio.specimen_envelope(raw,header['kind'])
        original,normalization=normalize_image(raw,support)
        specimen_core=ndi.binary_erosion(support,iterations=6)
        if specimen_core.sum()<64:specimen_core=support
        dark=float(np.percentile(original[specimen_core],cfg.dark_percentile))
        save_tiff(output/'backgrounds'/f'{g}_original.tiff',original)
        save_tiff(output/'backgrounds'/f'{g}_specimen.tiff',support.astype(np.uint8))
        write_json(output/'backgrounds'/f'{g}_metadata.json',dict(source=record['source'],source_slice=int(record['slice_index_zero_based']),
            original_shape=list(raw.shape),raw_pixel_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),normalization=normalization,specimen=support_info,dark_reference=dark,
            voxel_um_recorded=record['voxel_um'],calibration_note=record.get('calibration_note',''),
            dark_reference_status='unlabelled low-tail heuristic; not measured crack air'))
        native_field=cached_native_field(original,support)
        save_tiff(output/'backgrounds'/f'{g}_native_W.tiff',native_field['prior'].astype(np.float16))
        for variant in range(cfg.variants):
            mode='control' if variant==cfg.variants-1 else 'challenge' if variant==cfg.variants-2 else 'core'
            seed=cfg.seed+10000*MATERIALS.index(material)+100*bi+variant
            if mode=='control':
                paths=[];coverage=np.zeros_like(original);mask=np.zeros(original.shape,bool);width=np.zeros_like(original)
                virtual=original.copy();drop=np.zeros_like(original);confidence=np.ones_like(original);info={'weak_gray_fraction':None}
            else:
                paths=route_paths(original,support,seed,mode,cfg)
                coverage,mask,width=rasterize_tiled(paths,original.shape,support,cfg)
                virtual,drop,confidence,info=render_crack(original,support,coverage,seed,mode,cfg,dark)
            if mode=='control': ignored=np.zeros(original.shape,bool)
            else: ignored,ambiguity_info=supervision_ambiguity(original,support,virtual,drop,coverage,cached_native_field(original,support))
            confidence=(~ignored).astype(np.float32)
            if mode!='control': info['ambiguity_diagnostics']=ambiguity_info
            info.update(supervision='fractional coverage; equal-class soft CE + masked soft Dice',ignored_positive_pixels=int(ignored.sum()),ignore_rule='strong existing edge AND gray similarity AND weak new signal')
            ident=f'{g}_v{variant+1:02d}_{mode}';folder=output/'samples'/ident;folder.mkdir()
            for key,arr in dict(image=virtual,mask=mask.astype(np.uint8),ignore=ignored.astype(np.uint8),training_mask=np.where(ignored,255,(coverage>=.5).astype(np.uint8)).astype(np.uint8),coverage=np.rint(coverage*cfg.raster_supersampling**2).astype(np.uint8),
                width=width.astype(np.float16),confidence=np.rint(confidence*255).astype(np.uint8),decrement=drop.astype(np.float16)).items():
                save_tiff(folder/(key+'.tiff'),arr)
            # Geometry is retained; coverage is the soft target and validity is separate.
            geometry=[dict(kind=p['kind'],points_yx=p['points'][::cfg.raster_supersampling].tolist(),
                width_px=p['widths'][::cfg.raster_supersampling].tolist()) for p in paths]
            write_json(folder/'geometry_preview.json',geometry)
            write_json(folder/'rendering.json',dict(**info,geometry_label='coverage>0; full touched-pixel geometry',
                width_bounds_px=list(cfg.core_width_px if mode=='core' else cfg.challenge_width_px) if mode!='control' else None,
                geometry_preview_decimation=cfg.raster_supersampling))
            row=dict(id=ident,material=material,background_group=g,split=assignments[g],mode=mode,seed=seed,
                shape_yx=json.dumps(list(original.shape)),source=record['source'],source_slice=record['slice_index_zero_based'],
                voxel_um=record['voxel_um'],calibration_note=record.get('calibration_note',''),
                image=str((folder/'image.tiff').relative_to(output)),mask=str((folder/'mask.tiff').relative_to(output)),
                confidence=str((folder/'confidence.tiff').relative_to(output)),original=f'backgrounds/{g}_original.tiff',
                specimen=f'backgrounds/{g}_specimen.tiff',foreground_pixels=int(mask.sum()),coverage=str((folder/'coverage.tiff').relative_to(output)),ignore=str((folder/'ignore.tiff').relative_to(output)),width_band=str(variant%3) if mode=='core' else mode)
            rows.append(row)
        print(material,g,list(original.shape),'full native slice generated; split',assignments[g],flush=True)
    write_rows(output/'manifest.csv',rows)
    verification=verify_dataset(output)
    write_json(output/'COMPLETE.json',dict(configuration=requested,verification=verification))
    return output


def verify_dataset(root):
    root=Path(root);rows=read_rows(root/'manifest.csv');group_splits={};count=0
    cfg=json.loads((root/'configuration.json').read_text())['generation'];s=cfg['raster_supersampling']
    for row in rows:
        group_splits.setdefault(row['background_group'],set()).add(row['split'])
        folder=(root/row['image']).parent
        image=tifffile.imread(folder/'image.tiff');mask=tifffile.imread(folder/'mask.tiff');coverage=tifffile.imread(folder/'coverage.tiff')
        width=tifffile.imread(folder/'width.tiff');original=tifffile.imread(root/row['original'])
        support=tifffile.imread(root/row['specimen']).astype(bool)
        assert image.shape==mask.shape==original.shape==tuple(json.loads(row['shape_yx']))
        assert np.array_equal(mask,coverage>0) and not mask[~support].any()
        ignored=tifffile.imread(folder/'ignore.tiff').astype(bool)
        training_mask=tifffile.imread(folder/'training_mask.tiff')
        confidence=tifffile.imread(folder/'confidence.tiff')
        assert np.array_equal(training_mask==255,ignored) and not ignored[mask==0].any()
        assert np.array_equal(confidence==0,ignored) and set(np.unique(confidence))<= {0,255}
        assert np.isfinite(image).all() and image.min()>=0 and image.max()<=1
        assert coverage.max()<=s*s
        if row['mode']=='control':assert not mask.any() and np.array_equal(image,original)
        else:
            low,high=cfg['core_width_bands'][int(row['width_band'])] if row['mode']=='core' else cfg['challenge_width_px']
            assert width[mask>0].min()>=low-.003 and width.max()<=high+.003
        count+=1
    assert all(len(v)==1 for v in group_splits.values())
    return dict(samples_checked=count,background_groups=len(group_splits),group_leakage=False,
        full_native_shapes=True,geometry_equals_coverage_positive=True,finite_and_width_bounds=True,
        independence='background slice groups only; not independent specimens or acquisitions')

# %% 5. Sampling, no-resize augmentation, weighted training loss, full-label validation

def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def validation_selections(root,rows,p,windows):
    """One fixed foreground-rich window plus uniform interior windows per image.

    Label-selected windows are explicitly stratified diagnostics, not random prevalence.
    Final full-slice inference is evaluated separately.
    """
    selected=[]
    for row in rows:
        mask=tifffile.imread(Path(root)/row['mask']);h,w=mask.shape
        if min(h,w)<p:raise ValueError('Patch larger than source; resizing is forbidden')
        candidates=[]
        if mask.any():
            density=ndi.uniform_filter(mask.astype(float),size=p,mode='constant')
            density[:p//2]=0;density[-p//2:]=0;density[:,:p//2]=0;density[:,-p//2:]=0
            cy,cx=np.unravel_index(density.argmax(),density.shape)
            candidates.append((int(np.clip(cy-p//2,0,h-p)),int(np.clip(cx-p//2,0,w-p))))
        for fy,fx in ((.25,.25),(.25,.75),(.75,.25),(.75,.75),(.5,.5)):
            candidates.append((int((h-p)*fy),int((w-p)*fx)))
        unique=list(dict.fromkeys(candidates))[:windows]
        for y,x in unique:selected.append(dict(id=row['id'],origin_yx=[y,x],kind='fixed stratified native window'))
    return selected


class NativeDataset(Dataset):
    def __init__(self,root,rows,cfg,training=False):
        self.root,self.rows,self.cfg,self.training=Path(root),rows,cfg,training
        self.index={r['id']:r for r in rows};self.groups={}
        for r in rows:self.groups.setdefault(r['mode'],[]).append(r)
        self.fixed=[] if training else validation_selections(root,rows,cfg.patch_size,cfg.validation_windows_per_image)
        self.cache={}
        generation=json.loads((self.root/'configuration.json').read_text())['generation']
        self.coverage_scale=float(generation['raster_supersampling']**2)
    def __len__(self):return self.cfg.steps_per_epoch*self.cfg.batch_size if self.training else len(self.fixed)
    def arrays(self,row):
        ident=row['id']
        if ident not in self.cache:
            if len(self.cache)>=16:self.cache.pop(next(iter(self.cache)))
            self.cache[ident]=(tifffile.imread(self.root/row['image']),tifffile.imread(self.root/row['coverage']),
                tifffile.imread(self.root/row['confidence']))
        return self.cache[ident]
    def __getitem__(self,index):
        p=self.cfg.patch_size
        if self.training:
            # Deliberate mix:60% resolved/core,20% challenge,20% no-injection.
            mode=random.choices(('core','challenge','control'),weights=(.6,.2,.2))[0]
            row=random.choice(self.groups[mode]);image,mask,confidence=self.arrays(row);h,w=image.shape
            if mask.any() and random.random()<.75:
                points=np.argwhere(mask>0);cy,cx=points[random.randrange(len(points))]
                y=random.randint(max(0,int(cy)-p+1),min(int(cy),h-p));x=random.randint(max(0,int(cx)-p+1),min(int(cx),w-p))
            else:y,x=random.randint(0,h-p),random.randint(0,w-p)
        else:
            sample=self.fixed[index];row=self.index[sample['id']];image,mask,confidence=self.arrays(row);y,x=sample['origin_yx']
        a=image[y:y+p,x:x+p];b=mask[y:y+p,x:x+p].astype(np.float32)/self.coverage_scale;c=confidence[y:y+p,x:x+p].astype(np.float32)/255
        if self.training:
            k=random.randrange(4);a,b,c=(np.rot90(q,k) for q in (a,b,c))
            if random.random()<.5:a,b,c=(q[:,::-1] for q in (a,b,c))
        return dict(image=torch.from_numpy(np.ascontiguousarray(a[None]*2-1)),
            label=torch.from_numpy(np.ascontiguousarray(b.astype(np.float32))),
            weight=torch.from_numpy(np.ascontiguousarray(c)),id=row['id'],origin=torch.tensor([y,x]))


def segmentation_loss(logits,target,weight):
    """Equal-class soft CE + soft Dice; every ignored pixel has zero gradient.

    Targets are native subpixel area fractions, not calibrated CT crack probabilities.
    Weight is binary validity. The PSF gray tail never expands target geometry.
    """
    logp=logits.log_softmax(1)
    ce=-((1-target)*logp[:,0]+target*logp[:,1])
    ce=(ce*weight).sum()/weight.sum().clamp_min(1)
    probability=logits.softmax(1)[:,1]
    dice=1-(2*(weight*probability*target).sum()+1)/((weight*(probability+target)).sum()+1)
    return ce+dice


def metrics(tp,fp,fn):
    return dict(tp=int(tp),fp=int(fp),fn=int(fn),dice=float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 1.,
        iou=float(tp/(tp+fp+fn)) if tp+fp+fn else 1.,precision=float(tp/(tp+fp)) if tp+fp else None,
        recall=float(tp/(tp+fn)) if tp+fn else None)


def run_epoch(model,loader,device,threshold,optimizer=None):
    if torch.device(device).type != 'mps':
        raise RuntimeError('Neural execution requires Apple MPS; CPU fallback is disabled.')
    training=optimizer is not None;model.train(training);total=count=0
    tp=fp=fn=0;overlap=predicted=target_area=0.
    with torch.set_grad_enabled(training):
        for batch in loader:
            x,y,w=batch['image'].to(device),batch['label'].to(device),batch['weight'].to(device)
            if training:optimizer.zero_grad(set_to_none=True)
            logits=model(x);loss=segmentation_loss(logits,y,w)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            if training:loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            total+=float(loss.detach())*len(x);count+=len(x)
            pred=logits.detach().softmax(1)[:,1]>=threshold;true=y>=.5
            # Metrics include ignored regions; omission cannot inflate reported metrics.
            tp+=int((pred&true).sum());fp+=int((pred&~true).sum());fn+=int((~pred&true).sum())
            overlap+=float((pred*y).sum());predicted+=float(pred.sum());target_area+=float(y.sum())
    result=dict(loss=total/max(count,1),**metrics(tp,fp,fn))
    result['soft_area_dice']=2*overlap/(predicted+target_area) if predicted+target_area else 1.
    return result

# %% 6. Training one independent model per material, checkpoints and loss curves

def make_model(cfg):return BasicUNet(spatial_dims=2,in_channels=1,out_channels=2,features=cfg.features,dropout=0.)

def train_material(material,data_root,output,cfg=TrainingConfig(),model_factory=None):
    if torch.device(cfg.device).type != 'mps':
        raise RuntimeError('Training requires Apple MPS; CPU fallback is disabled.')
    from .mps import require_mps
    require_mps()
    seed_everything(cfg.seed);torch.set_num_threads(cfg.num_threads)
    if cfg.patch_size not in (128,256,512):raise ValueError('Use128(smoke),256 or512 native pixels')
    device=torch.device(cfg.device);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows=read_rows(Path(data_root)/'manifest.csv');train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='validation']
    assert train and val and not ({r['background_group'] for r in train}&{r['background_group'] for r in val})
    write_json(output/'configuration.json',dict(material=material,training=asdict(cfg),dataset=str(Path(data_root).resolve()),
        dataset_complete_sha256=sha(Path(data_root)/'COMPLETE.json'),engine_sha256=sha(ENGINE_FILE),
        meaning='native W attraction + soft coverage supervision; geometry retained; no real accuracy claim'))
    write_json(output/'split.json',{k:[r['id'] for r in rows if r['split']==k] for k in ('train','validation','test')})
    tr_ds=NativeDataset(data_root,train,cfg,True);va_ds=NativeDataset(data_root,val,cfg,False)
    write_json(output/'validation_windows.json',va_ds.fixed)
    tr=DataLoader(tr_ds,batch_size=cfg.batch_size,shuffle=False,num_workers=0)
    va=DataLoader(va_ds,batch_size=cfg.batch_size,shuffle=False,num_workers=0)
    model=(make_model(cfg) if model_factory is None else model_factory(cfg)).to(device);opt=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg.max_epochs,eta_min=1e-5)
    best=float('inf');stale=0;history=[];start=time.time()
    for epoch in range(1,cfg.max_epochs+1):
        a=run_epoch(model,tr,device,cfg.threshold,opt);b=run_epoch(model,va,device,cfg.threshold)
        history.append(dict(epoch=epoch,train_loss=a['loss'],val_loss=b['loss'],val_dice=b['dice'],val_recall=b['recall'],val_soft_area_dice=b['soft_area_dice'],lr=opt.param_groups[0]['lr']))
        pd.DataFrame(history).to_csv(output/'history.csv',index=False)
        if b['loss']<best-1e-4:
            best=b['loss'];stale=0
            torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},material=material,
                epoch=epoch,config=asdict(cfg),validation=b),output/'best_model.pt')
        else:stale+=1
        scheduler.step()
        print(f'{material} {epoch:03d}/{cfg.max_epochs}: train {a["loss"]:.4f} val {b["loss"]:.4f} synthetic Dice {b["dice"]:.3f}',flush=True)
        if epoch>=cfg.min_epochs and stale>=cfg.patience:break
    checkpoint=torch.load(output/'best_model.pt',map_location=device,weights_only=False);model.load_state_dict(checkpoint['state_dict'])
    result=dict(material=material,epochs=len(history),best_epoch=checkpoint['epoch'],duration_seconds=time.time()-start,
        validation=checkpoint['validation'],evaluation='model-selection windows: majority-coverage binary Dice and all-coverage soft-area Dice; ignored regions included in metrics, not loss; no real accuracy')
    write_json(output/'summary.json',result)
    fig=Figure(figsize=(7,4));FigureCanvasAgg(fig);ax=fig.subplots();table=pd.DataFrame(history)
    ax.plot(table.epoch,table.train_loss,label='Training: soft coverage, ignored overlap');ax.plot(table.epoch,table.val_loss,label='Validation: same masked soft objective')
    ax.set(xlabel='Epoch',ylabel='CE + Dice',title=material);ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/'loss_vs_epoch.png',dpi=150)
    visualize_validation(model,va_ds,output,material,device,cfg)
    # Test is opt-in; default notebooks never use it to select settings/checkpoints.
    if cfg.evaluate_test:
        test=[r for r in rows if r['split']=='test']
        write_json(output/'test_full_slice_metrics.json',evaluate_full_slices(model,data_root,test,device,cfg))
    if cfg.export_real_count:
        real_inference(model,material,output,device,cfg)
    return result

# %% 7. Native-pixel tiled inference and honest synthetic/real visualization

@torch.no_grad()
def predict_native(model,image,device,cfg):
    """Overlap-add probabilities with a nonzero Hann window; no image resizing."""
    if torch.device(device).type != 'mps' or not torch.backends.mps.is_available():
        raise RuntimeError('Inference requires accessible Apple MPS; CPU fallback is disabled.')
    model.eval();p=cfg.patch_size;h,w=image.shape
    if min(h,w)<p:raise ValueError('Input smaller than model patch')
    stride=p//2
    ys=sorted(set(list(range(0,h-p+1,stride))+[h-p]));xs=sorted(set(list(range(0,w-p+1,stride))+[w-p]))
    window=np.maximum(np.outer(np.hanning(p),np.hanning(p)),.01).astype(np.float32)
    total=np.zeros((h,w),np.float32);weight=np.zeros((h,w),np.float32)
    for y in ys:
        for start in range(0,len(xs),cfg.batch_size):
            columns=xs[start:start+cfg.batch_size]
            a=np.stack([image[y:y+p,x:x+p] for x in columns])[:,None]*2-1
            probabilities=model(torch.from_numpy(a.astype(np.float32)).to(device)).softmax(1)[:,1].cpu().numpy()
            for x,prob in zip(columns,probabilities):
                total[y:y+p,x:x+p]+=prob*window;weight[y:y+p,x:x+p]+=window
    assert (weight>0).all()
    return total/weight


def evaluate_full_slices(model,root,rows,device,cfg):
    result=[]
    for row in rows:
        image=tifffile.imread(Path(root)/row['image']);geometry=tifffile.imread(Path(root)/row['mask']).astype(bool)
        scale=json.loads((Path(root)/'configuration.json').read_text())['generation']['raster_supersampling']**2
        coverage=tifffile.imread(Path(root)/row['coverage']).astype(np.float32)/scale;majority=coverage>=.5
        pred=predict_native(model,image,device,cfg)>=cfg.threshold
        area=float(coverage.sum());intersection=float((coverage*pred).sum());pred_area=int(pred.sum())
        majority_metrics=metrics(int((pred&majority).sum()),int((pred&~majority).sum()),int((~pred&majority).sum()))
        result.append(dict(id=row['id'],mode=row['mode'],majority_coverage_metrics=majority_metrics,
            soft_area_dice=2*intersection/(area+pred_area) if area+pred_area else 1.,
            soft_area_recall=intersection/area if area else None,coverage_area_px=area,prediction_area_px=pred_area,
            prediction_area_over_coverage=pred_area/area if area else None,
            touched_geometry_recall=float((pred&geometry).sum()/geometry.sum()) if geometry.any() else None))
    return dict(per_image=result,scope='prespecified whole synthetic validation slices; all coverage incl. ignored pixels; no real accuracy')



def selected_validation_rows(root):
    rows=read_rows(Path(root)/'manifest.csv');groups=sorted({r['background_group'] for r in rows if r['split']=='validation'})
    chosen=[]
    for i in range(3):
        group=groups[min(i,len(groups)-1)]
        chosen.append([r for r in rows if r['background_group']==group and r['mode']=='core'][i])
    for mode in ('challenge','control'):
        chosen.append(next(r for r in rows if r['background_group']==groups[0] and r['mode']==mode))
    return chosen


def visualize_validation(model,dataset,output,material,device,cfg):
    selected=selected_validation_rows(dataset.root);details=[]
    fig=Figure(figsize=(14,11));FigureCanvasAgg(fig);axes=fig.subplots(5,6)
    for i,row in enumerate(selected):
        root=dataset.root;folder=(root/row['image']).parent
        original=tifffile.imread(root/row['original']);virtual=tifffile.imread(root/row['image'])
        coverage=tifffile.imread(root/row['coverage']).astype(np.float32)/dataset.coverage_scale
        ignored=tifffile.imread(root/row['ignore']).astype(bool)
        selection=validation_selections(root,[row],cfg.patch_size,1)[0];y,x=selection['origin_yx'];p=cfg.patch_size
        sl=np.s_[y:y+p,x:x+p];prob=predict_native(model,virtual[sl],device,cfg);pred=prob>=cfg.threshold
        for j,(array,title) in enumerate(zip((original[sl],virtual[sl],coverage[sl],ignored[sl],prob,pred),
            ('Original XCT','Synthetic XCT','Soft coverage target','Ignored overlap','Probability','Prediction'))):
            axes[i,j].imshow(array,cmap='gray',vmin=0,vmax=1,interpolation='nearest');axes[i,j].axis('off')
            axes[i,j].set_title((row['mode']+' / band '+row['width_band']+'\n' if j==0 else '')+title,fontsize=8)
        save_tiff(Path(output)/(row['id']+'_probability.tiff'),prob.astype(np.float32))
        save_tiff(Path(output)/(row['id']+'_prediction.tiff'),pred.astype(np.uint8))
        area=float(coverage[sl].sum());predarea=int(pred.sum());intersection=float((pred*coverage[sl]).sum())
        details.append(dict(id=row['id'],mode=row['mode'],width_band=row['width_band'],origin_yx=[y,x],threshold=cfg.threshold,
            soft_area_dice=2*intersection/(area+predarea) if area+predarea else 1.,prediction_area_over_coverage=predarea/area if area else None))
    fig.suptitle(material+': fixed validation windows | fine, medium, wide, challenge, control | no real accuracy',fontsize=11)
    fig.tight_layout();fig.savefig(Path(output)/'validation_examples.png',dpi=140)
    write_json(Path(output)/'visualization_selection.json',details)



def generation_preview(data_root,destination):
    root=Path(data_root);rows=read_rows(root/'manifest.csv')
    # Training-only first background previews; no peeking at held-out test images.
    selected=[next(r for r in rows if r['split']=='train' and r['mode']==mode) for mode in ('core','challenge','control')]
    fig=Figure(figsize=(13,10));FigureCanvasAgg(fig);axes=fig.subplots(3,4);selections=[]
    for i,row in enumerate(selected):
        image=tifffile.imread(root/row['image']);mask=tifffile.imread(root/row['mask']);original=tifffile.imread(root/row['original'])
        confidence=tifffile.imread(root/row['confidence']).astype(float)/255
        p=256;h,w=image.shape
        if mask.any():
            density=ndi.uniform_filter(mask.astype(float),p);density[:p//2]=0;density[-p//2:]=0;density[:,:p//2]=0;density[:,-p//2:]=0
            cy,cx=np.unravel_index(density.argmax(),density.shape);y,x=int(cy-p//2),int(cx-p//2)
        else:y,x=(h-p)//2,(w-p)//2
        for j,a in enumerate((original,image,mask,confidence)):
            axes[i,j].imshow(a[y:y+p,x:x+p],cmap='viridis' if j==3 else 'gray',vmin=0,vmax=1,interpolation='nearest');axes[i,j].axis('off')
            axes[i,j].set_title((row['mode']+'\n' if j==0 else '')+('Original','Synthetic input','Full geometry','Training confidence')[j])
        selections.append(dict(id=row['id'],origin_yx=[y,x],native_shape=[h,w],display_shape=[p,p]))
    fig.suptitle(rows[0]['material']+': native full-slice generation, training-group diagnostic windows')
    fig.tight_layout();fig.savefig(destination,dpi=140);write_json(Path(destination).with_suffix('.json'),selections)


def real_inference(model,material,output,device,cfg):
    """Unlabelled damaged slices: retain predictions, never report real Dice/accuracy."""
    rows=read_rows(SOURCE_DATA/material/'real_annotation/manifest.csv')[:cfg.export_real_count]
    report=[]
    for row in rows:
        raw=tifffile.imread(SOURCE_DATA/row['image']);support,_=xio.specimen_envelope(raw,'TXM' if Path(row['source']).suffix.lower()=='.txm' else 'TIFF')
        image,norm=normalize_image(raw,support);prob=predict_native(model,image,device,cfg);pred=prob>=cfg.threshold
        folder=Path(output)/'real_unlabelled';folder.mkdir(exist_ok=True)
        save_tiff(folder/(row['id']+'_probability.tiff'),prob);save_tiff(folder/(row['id']+'_prediction.tiff'),pred.astype(np.uint8))
        fig=Figure(figsize=(12,4));FigureCanvasAgg(fig);axes=fig.subplots(1,3)
        for ax,a,title in zip(axes,(image,prob,pred),('Real damaged slice','Probability (unvalidated)','Prediction (no ground truth)')):
            ax.imshow(a,cmap='gray',vmin=0,vmax=1);ax.set_title(title);ax.axis('off')
        fig.tight_layout();fig.savefig(folder/(row['id']+'.png'),dpi=130)
        report.append(dict(id=row['id'],normalization=norm,shape=list(raw.shape),annotation_status='unlabelled',accuracy=None))
    write_json(Path(output)/'real_inference.json',report)



_NATIVE_FIELD_CACHE={}
def cached_native_field(original,specimen):
    entry=_NATIVE_FIELD_CACHE
    if entry.get('image') is not original or entry.get('specimen') is not specimen:
        entry.clear();entry.update(image=original,specimen=specimen,field=structural_prior(original,specimen,RouteSettings(scales_px=(1.,2.,4.),guidance_strength=3.)))
    return entry['field']


def supervision_ambiguity(original,specimen,virtual,drop,coverage,field):
    smooth=ndi.gaussian_filter(original,.8);gy,gx=np.gradient(smooth);mag=np.hypot(gy,gx)
    yy,xx=np.indices(original.shape,dtype=np.float32);ny=gy/(mag+1e-8);nx=gx/(mag+1e-8)
    plus=ndi.map_coordinates(smooth,[yy+2*ny,xx+2*nx],order=1,mode='nearest')
    minus=ndi.map_coordinates(smooth,[yy-2*ny,xx-2*nx],order=1,mode='nearest')
    noise=field['structure_info']['noise_std'];w=field['prior']
    edge=(w>=max(.02,float(np.percentile(w[specimen],80))))&(np.abs(plus-minus)>max(.025,2*noise))&specimen
    near=ndi.binary_dilation(edge,iterations=1)
    dark_side=ndi.minimum_filter(np.minimum(plus,minus),size=3)
    similar=np.abs(virtual-dark_side)<=max(.02,noise)
    residual=original-ndi.gaussian_filter(original,1.)
    texture=np.sqrt(np.maximum(ndi.gaussian_filter(residual**2,2.),0))
    weak=(drop<max(.02,noise))|(drop/(texture+.005)<.75)
    ignored=(coverage>0)&near&similar&weak
    return ignored,dict(noise_proxy=float(noise),ignored_fraction_of_geometry=float(ignored.sum()/max((coverage>0).sum(),1)),
        status='fixed image-domain heuristic, not calibrated perception or phase labels')
