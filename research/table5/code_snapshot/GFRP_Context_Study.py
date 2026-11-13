"""Synthetic-only GFRP input-representation and native-context ablation.

The canonical engine, rendering, targets and split remain frozen. Channels are
computed identically in training and inference, without median filtering or
geometric resizing. No real image enters the training loader or loss.
"""
from pathlib import Path
from dataclasses import replace,asdict
import json,time
import numpy as np
import torch
import torch.nn.functional as F
from monai.networks.nets import BasicUNet

def local_channels(x):
    channels=[x]
    for size in (9,31):
        radius=size//2
        def mean(a):
            a=F.avg_pool2d(F.pad(a,(radius,radius,0,0),mode='reflect'),(1,size),stride=1)
            return F.avg_pool2d(F.pad(a,(0,0,radius,radius),mode='reflect'),(size,1),stride=1)
        mu=mean(x);variance=(mean(x*x)-mu*mu).clamp_min(0)
        z=(x-mu)/torch.sqrt(variance+.03**2)
        channels.append(z.clamp(-3,3)/3)
    return torch.cat(channels,dim=1)

class ContrastUNet(torch.nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.net=BasicUNet(spatial_dims=2,in_channels=3,out_channels=2,features=cfg.features,dropout=0.)
    def forward(self,x):return self.net(local_channels(x))

def prepare_cache(e,root):
    root=Path(root);data=e.PROJECT/'outputs/optimized_crack_training/20261029_nativeW_softcoverage/gfrp/dataset'
    cache=root/'native_mmap';cache.mkdir(parents=True,exist_ok=True)
    signature=dict(dataset_complete_sha256=e.sha(data/'COMPLETE.json'),format='exact uncompressed numpy arrays, no quantization',splits=['train','validation'])
    done=cache/'COMPLETE.json'
    if done.exists():
        assert json.loads(done.read_text())==signature
        return cache
    rows=[r for r in e.read_rows(data/'manifest.csv') if r['split']!='test']
    for index,row in enumerate(rows):
        folder=cache/row['id'];folder.mkdir(exist_ok=True)
        for key in ('image','coverage','confidence'):
            dest=folder/(key+'.npy')
            if not dest.exists():np.save(dest,e.tifffile.imread(data/row[key]))
            assert np.array_equal(np.load(dest,mmap_mode='r'),e.tifffile.imread(data/row[key]))
        if index%20==0:print('Exact native cache:',index+1,'/',len(rows),flush=True)
    e.write_json(done,signature)
    return cache

def input_fixtures(e):
    x=torch.full((1,1,64,64),.4)
    z=local_channels(x)
    assert z.shape==(1,3,64,64) and torch.equal(z[:,0:1],x)
    assert z[:,1:].abs().max()<.001
    x[:,:,30:34,8:56]=-.7
    gain=local_channels(x*1.5+.1)
    base=local_channels(x)
    assert (gain[:,1:]-base[:,1:]).abs().mean()<.03
    cfg=e.TrainingConfig(patch_size=128,batch_size=1,num_threads=2)
    torch.set_num_threads(2);model=ContrastUNet(cfg)
    logits=model(torch.rand(1,1,128,128)*2-1);target=torch.zeros(1,128,128);valid=torch.ones_like(target)
    target[:,30:34,8:110]=1;valid[:,40:50,40:50]=0
    logits.retain_grad();loss=e.segmentation_loss(logits,target,valid);loss.backward()
    assert torch.isfinite(loss) and logits.grad[:,:,40:50,40:50].abs().max()==0
    return dict(channels=3,raw_channel_exact=True,flat_local_channels_zero=True,
        affine_intensity_fixture_mean_error=float((gain[:,1:]-base[:,1:]).abs().mean()),
        finite_gradient=True,ignored_pixels_zero_gradient=True)

def train_context(e,root,patch_size):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    data=e.PROJECT/'outputs/optimized_crack_training/20261029_nativeW_softcoverage/gfrp/dataset'
    output=root/('contrast'+str(patch_size))/'training'
    cfg=replace(e.TrainingConfig(),patch_size=patch_size,batch_size=4 if patch_size==256 else 1,
        validation_windows_per_image=4 if patch_size==256 else 1,
        max_epochs=120,min_epochs=40,patience=20,num_threads=2)
    cache=prepare_cache(e,root)
    protocol=dict(training=asdict(cfg),dataset_complete_sha256=e.sha(data/'COMPLETE.json'),
        implementation_sha256=e.sha(Path(__file__)),input_channels=['raw','local_z9','local_z31'],
        local_std_floor=.03,local_clip=3,real_training_images=0,source_split='unchanged12/4/4 background groups',
        targets='unchanged fractional coverage and binary validity',loss='unchanged equal-class soft CE + soft Dice',
        scope='synthetic development-validation; real inference unlabelled; no claim of proven generalization',
        cache='exact mmap input/coverage/confidence arrays; byte-equivalent array values checked before training',
        comparison='256 isolates representation relative to frozen baseline;512 also changes context/batch size and validation windows while preserving pixels per optimizer step and validation pixel budget; compare whole-slice metrics on common fixed cases')
    protocol=json.loads(json.dumps(protocol));path=root/('contrast'+str(patch_size)+'_protocol.json')
    if path.exists():assert json.loads(path.read_text())==protocol
    else:e.write_json(path,protocol)
    if (output/'summary.json').exists():return json.loads((output/'summary.json').read_text())
    original,original_dataset=e.make_model,e.NativeDataset
    class CachedDataset(original_dataset):
        def arrays(self,row):
            ident=row['id']
            if ident not in self.cache:
                if len(self.cache)>=16:self.cache.pop(next(iter(self.cache)))
                self.cache[ident]=tuple(np.load(cache/ident/(key+'.npy'),mmap_mode='r') for key in ('image','coverage','confidence'))
            return self.cache[ident]
    e.make_model=lambda config:ContrastUNet(config);e.NativeDataset=CachedDataset
    try:return e.train_material('gfrp',data,output,cfg)
    finally:e.make_model,e.NativeDataset=original,original_dataset
