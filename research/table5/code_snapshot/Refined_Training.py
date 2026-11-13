"""Controlled warm-start training on frozen synthetic partitions.

The development validation set is reused for selection. No real labels, test
images, altered masks, threshold tuning or morphological postprocessing.
"""
from pathlib import Path
from dataclasses import asdict, replace
import random, json, time, copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from GFRP_Context_Study import ContrastUNet


class EfficientDataset:
    """Build an engine-compatible dataset with bounded caches and exact sampling."""
    @staticmethod
    def create(e, root, rows, cfg, training):
        class Dataset(e.NativeDataset):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.positive_positions = {}

            def arrays(self, row):
                ident = row['id']
                if ident not in self.cache:
                    if len(self.cache) >= 64:
                        expired = next(iter(self.cache))
                        self.cache.pop(expired)
                        self.positive_positions.pop(expired, None)
                    self.cache[ident] = tuple(e.tifffile.imread(self.root / row[k])
                                              for k in ('image', 'coverage', 'confidence'))
                return self.cache[ident]

            def __getitem__(self, index):
                p = self.cfg.patch_size
                if self.training:
                    mode = random.choices(('core', 'challenge', 'control'), (.6, .2, .2))[0]
                    row = random.choice(self.groups[mode])
                    image, mask, confidence = self.arrays(row)
                    h, w = image.shape
                    ident = row['id']
                    if ident not in self.positive_positions:
                        self.positive_positions[ident] = np.argwhere(mask > 0)
                    points = self.positive_positions[ident]
                    if len(points) and random.random() < .75:
                        cy, cx = points[random.randrange(len(points))]
                        y = random.randint(max(0, int(cy)-p+1), min(int(cy), h-p))
                        x = random.randint(max(0, int(cx)-p+1), min(int(cx), w-p))
                    else:
                        y, x = random.randint(0, h-p), random.randint(0, w-p)
                else:
                    spec = self.fixed[index]
                    row = self.index[spec['id']]
                    image, mask, confidence = self.arrays(row)
                    y, x = spec['origin_yx']
                a = image[y:y+p, x:x+p].astype(np.float32, copy=True)
                b = mask[y:y+p, x:x+p].astype(np.float32) / self.coverage_scale
                c = confidence[y:y+p, x:x+p].astype(np.float32) / 255
                if self.training:
                    k = random.randrange(4)
                    a, b, c = (np.rot90(z, k) for z in (a, b, c))
                    if random.random() < .5:
                        a, b, c = (z[:, ::-1] for z in (a, b, c))
                    # Appearance only: targets/validity and native geometry unchanged.
                    if random.random() < .7:
                        a = np.clip(a, 0, 1) ** random.uniform(.85, 1.15)
                        a = a * random.uniform(.90, 1.10) + random.uniform(-.03, .03)
                    if random.random() < .5:
                        a = a + np.random.normal(0, random.uniform(0, .02), a.shape)
                    a = np.clip(a, 0, 1).astype(np.float32)
                return dict(image=torch.from_numpy(np.ascontiguousarray(a[None]*2-1)),
                            label=torch.from_numpy(np.ascontiguousarray(b)),
                            weight=torch.from_numpy(np.ascontiguousarray(c)),
                            id=row['id'], origin=torch.tensor([y, x]))
        return Dataset(root, rows, cfg, training)


def load_model(e, material, checkpoint):
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = e.TrainingConfig(**ck['config'])
    model = ContrastUNet(cfg) if material == 'gfrp' else e.make_model(cfg)
    model.load_state_dict(ck['state_dict'])
    return model, cfg


def train_refined(e, material, destination):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=False)
    old = e.PROJECT / 'outputs/optimized_crack_training/20261029_nativeW_softcoverage'
    data = old / material / 'dataset'
    source = (e.PROJECT / 'outputs/gfrp_context_study/20261029/contrast256/training/best_model.pt'
              if material == 'gfrp' else old / material / 'training/best_model.pt')
    model, base_cfg = load_model(e, material, source)
    cfg = replace(base_cfg, seed=20261029, max_epochs=40 if material == 'gfrp' else 30,
                  learning_rate=2e-4, num_threads=2, export_real_count=0)
    e.seed_everything(cfg.seed); torch.set_num_threads(cfg.num_threads)
    rows = e.read_rows(data / 'manifest.csv')
    train = [r for r in rows if r['split'] == 'train']
    val = [r for r in rows if r['split'] == 'validation']
    assert not ({r['background_group'] for r in train} & {r['background_group'] for r in val})
    tr = EfficientDataset.create(e, data, train, cfg, True)
    va = EfficientDataset.create(e, data, val, cfg, False)
    reference = e.NativeDataset(data, val, cfg, False)
    assert reference.fixed == va.fixed
    for i in range(len(va)):
        sample, exact = va[i], reference[i]
        for key in ('image', 'label', 'weight'):
            assert torch.equal(sample[key], exact[key]), key
    del reference
    tl = DataLoader(tr, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    vl = DataLoader(va, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    device = torch.device('cpu')
    baseline = e.run_epoch(model, vl, device, .5)
    best = baseline['loss']; best_epoch = 0; best_metrics = baseline
    def save(epoch, metrics):
        torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                        config=asdict(cfg), epoch=epoch, validation=metrics,
                        architecture='contrast256' if material=='gfrp' else 'raw',
                        material=material), destination/'best_model.pt')
    save(0, baseline)
    protocol = dict(training=asdict(cfg), source_checkpoint=str(source),
        source_checkpoint_sha256=e.sha(source), dataset=str(data),
        dataset_sha256=e.sha(data/'COMPLETE.json'), implementation_sha256=e.sha(Path(__file__)),
        validation_windows=va.fixed, train_ids=[r['id'] for r in train],
        validation_ids=[r['id'] for r in val], threshold=.5, test_used=False,
        target='unchanged fractional coverage', loss='unchanged masked soft CE + Dice',
        augmentation=dict(gamma=[.85,1.15], gain=[.9,1.1], offset=[-.03,.03], noise_std=[0,.02]),
        selection='minimum unchanged validation loss, including the source model at epoch zero',
        scope='continued training plus appearance augmentation; not a causal ablation or real-image accuracy')
    e.write_json(destination/'protocol.json', protocol)
    e.write_json(destination/'baseline.json', baseline)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.max_epochs, eta_min=1e-5)
    history=[]; start=time.time()
    for epoch in range(1,cfg.max_epochs+1):
        a=e.run_epoch(model,tl,device,.5,optimizer)
        b=e.run_epoch(model,vl,device,.5)
        if b['loss'] < best-1e-5:
            best=b['loss']; best_metrics=b; best_epoch=epoch; save(epoch,b)
        history.append(dict(epoch=epoch,train_loss=a['loss'],**{'val_'+k:v for k,v in b.items()}))
        e.write_rows(destination/'history.csv',history)
        scheduler.step()
        print(material,epoch,'Dice',round(b['dice'],4),'area',round(b['soft_area_dice'],4),'best',best_epoch,flush=True)
    summary=dict(material=material,baseline=baseline,validation=best_metrics,best_epoch=best_epoch,
                 epochs=len(history),duration_seconds=time.time()-start,test_used=False,real_accuracy=None)
    e.write_json(destination/'summary.json',summary)
    return summary
