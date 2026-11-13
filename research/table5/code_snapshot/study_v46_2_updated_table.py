"""Fixed new generator; reproduce the original five-row table with80 new fits.

Twenty original matched real-only fits are reused unchanged. Virtual-gradient
training uses injected-geometry labels and original no-injection controls only;
manual fitting masks never select its background patches.
"""
from pathlib import Path
import sys,os,json,time,shutil,argparse,fcntl,traceback
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import study_v46_2_matched_five as engine
import study_v46_2_unified as prior
ORIGINAL=ROOT/'outputs/v46_2/20261112_matched14_val6_five_seed_v1'
DATA=ROOT/'outputs/v46_2/20261113_unified_generator_v1/data'
OUT=ROOT/'outputs/v46_2/20261113_updated14_val6_table_v1'
MS=engine.MS;SEEDS=engine.SEEDS;ARMS=engine.ARMS;NAMES=engine.NAMES
BaseVirtual=prior.VirtualStream

class SourceComponent(BaseVirtual):
    def __init__(self,m,kind,seed,batch):
        self.m=m;self.spec=dict(prior.SPECS[kind],safe=False);self.seed=seed;self.epoch=0;self.batch=batch
        self.cache={};self.points={};self.counts={};self.neg=[]
        self.base=json.loads((ORIGINAL/'sources.json').read_text())[m]['virtual']
        self.ext=json.loads((DATA/m/'records.json').read_text())

class SourceOnlyMixture:
    def __init__(self,m,cfg,translate=True):
        assert cfg.patch_size==256 and cfg.batch_size==4 and translate
        self.original=SourceComponent(m,'negative35',cfg.seed,4)
        self.expanded=SourceComponent(m,'combined',cfg.seed,4)
        self.seed=cfg.seed;self.epoch=0;self.cfg=cfg
    def __len__(self):return self.cfg.steps_per_epoch*self.cfg.batch_size
    def __getitem__(self,index):
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,index,928]));stream=self.expanded if rng.random()<.20 else self.original
        stream.epoch=self.epoch;return stream[index]

def configure():
    engine.OUT=OUT;engine.baseline.PairedTranslationDataset=SourceOnlyMixture
    engine.RealStream=lambda items,seed:prior.RealStream(items,seed,4)

def directory(m,arm,seed):return OUT/'runs'/m/arm/f'seed{seed}'

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'PREPARATION_COMPLETE.json').exists():
        p=json.loads((OUT/'PROTOCOL.json').read_text());assert p['dependencies'][str(Path(__file__).relative_to(ROOT))]==engine.sha(__file__);return
    sources=json.loads((ORIGINAL/'sources.json').read_text());imports=[];checks=[]
    for name in ['sources.json','validation_windows.json','fixed_split.csv','label_manifest.csv']:
        shutil.copy2(ORIGINAL/name,OUT/name)
    for m in MS:
        for seed in SEEDS:
            src=ORIGINAL/'runs'/m/'real14'/f'seed{seed}';summary=json.loads((src/'SUMMARY.json').read_text());rp=json.loads((src/'RUN_PROTOCOL.json').read_text())
            assert summary['backend']=='mps' and rp['config']['batch_size']==4 and rp['config']['patch_size']==256
            assert set(rp['fitting_ids'])=={r['id'] for r in sources[m]['real14']}
            assert set(rp['validation_ids'])=={r['id'] for r in sources[m]['validation']}
            assert rp['settings']==dict(learning_rate=.001,max_epochs=260,min_epochs=80,patience=30)
            assert engine.sha(src/'best_model.pt')==summary['checkpoint_sha256']
            dest=directory(m,'real14',seed);dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():dest.symlink_to(src,target_is_directory=True)
            imports.append(dict(material=m,arm='real14',seed=seed,source=str(src),summary_sha256=engine.sha(src/'SUMMARY.json'),checkpoint_sha256=summary['checkpoint_sha256']))
        cfg=engine.e.TrainingConfig(seed=271,device='mps',patch_size=256,batch_size=4)
        dataset=SourceOnlyMixture(m,cfg)
        assert dataset.original.neg==dataset.expanded.neg==[]
        assert not dataset.original.spec['safe'] and not dataset.expanded.spec['safe']
        for i in range(12):
            r=dataset[i];assert r['image'].shape==(1,256,256) and r['label'].shape==(256,256)
            assert r['label'].min()>=0 and r['label'].max()<=1 and r['weight'].min()>=0 and r['weight'].max()<=1
        checks.append(dict(material=m,samples=12,no_fitting_label_backgrounds=True))
    protocol=json.loads((ORIGINAL/'PROTOCOL.json').read_text());protocol['version']='V46.2 updated generator matched14-val6 five-seed table v1'
    protocol['finetune']['max_epochs']=260
    protocol.update(training_runs=100,new_training_runs=80,reused_real_only_runs=20,
        generator='Fixed:35%originalsource no-injection controls;positive-source80%originalV45,10%originalgeometry/newgrey,10%extendedgeometry/newgrey. No manuallylabelled fitting-image backgrounds.',
        virtual_meaning='Synthetic-gradient-only. Targetsdescribeinjectedgeometry;sourcecontrolsmean no NEW injection,notcertifiedabsenceofallnative damage. Realvalidation6selectcheckpoints.',
        original_table=str(ORIGINAL),data_root=str(DATA),batch_size=4,max_parallel=6,
        compute_reason='Keeporiginalbatch4to reuse20matchedreal-onlyfits;largerbatchdidnotshowmaterialthroughputgain. AllnewneuralworkMPSonly.',
        stage_budget_change='Fine cap260insteadofold120forall14/7/4budgets;generatorandfinebudgetbothdifferfromoldtable,notacausalclaimaboutgeneratoralone.',
        runtime_bindings='Originalverifiedtrainingengine;newvirtualdatasetandreadonlycachedrealsampler;neuralarchitecture/loss/validationunchanged.',
        dependencies={**protocol['dependencies'],str(Path(__file__).relative_to(ROOT)):engine.sha(__file__),str(Path(prior.__file__).relative_to(ROOT)):engine.sha(prior.__file__)})
    engine.js(OUT/'PROTOCOL.json',protocol);engine.js(OUT/'REUSED_REAL14.json',imports)
    engine.js(OUT/'PREPARATION_COMPLETE.json',dict(result='passed',reused_real14=20,new_fits=80,checks=checks,all14used=True,no_extra_holdout=True,nested4_7_14=True,source_only_virtual=True,checked_unix=time.time()))
    engine.js(OUT/'PROTECTED.json',json.loads((ORIGINAL/'PROTECTED.json').read_text()))
    snap=OUT/'code_snapshot';snap.mkdir(exist_ok=True)
    for name in [*protocol['dependencies'],str(Path(engine.__file__).relative_to(ROOT))]:shutil.copy2(ROOT/name,snap/Path(name).name)

def worker(m,arm,seed):
    configure();p=json.loads((OUT/'PROTOCOL.json').read_text())
    for path,digest in p['dependencies'].items():assert engine.sha(ROOT/path)==digest,path
    dest=directory(m,arm,seed);dest.mkdir(parents=True,exist_ok=True)
    with (dest/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:engine.train(m,arm,seed)
        except Exception as ex:engine.js(dest/'ERROR.json',dict(error=type(ex).__name__,message=str(ex),traceback=traceback.format_exc()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--material',choices=MS);p.add_argument('--arm',choices=ARMS);p.add_argument('--seed',type=int,choices=SEEDS);a=p.parse_args()
    if a.prepare:prepare()
    elif a.material:worker(a.material,a.arm,a.seed)
