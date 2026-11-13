"""Finite-opening diversity and explicitly annotation-assisted negative patches.

V45 decrement imaging is retained. Reuses existing V46 native paths rather than
claiming a new mechanical crack solver. No neural-network work in this module.
"""
from pathlib import Path
import os,sys,json,csv,time,hashlib,argparse
import numpy as np
from scipy import ndimage as ndi
import tifffile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'notebooks/optimized_four_material_training'))
import study_v46_2_matched_five as old
import Optimized_Crack_Engine as e
OUT=ROOT/'outputs/v46_2/20261113_unified_generator_v1'
BASE=old.baseline.DATA
DIV=ROOT/'outputs/v46_candidate/20261112_diversity_v1'
js=old.js;sha=old.sha

def save(folder,name,array):
    folder.mkdir(parents=True,exist_ok=True);p=folder/(name+'.npy');np.save(p,array);return str(p)

def raster(paths,shape,specimen,holes=(),scale=4):
    if not paths or max(float(p['widths'].max()) for p in paths)==0:return np.zeros(shape,np.float32)
    tile=256;halo=int(np.ceil(max(float(p['widths'].max()) for p in paths)/2))+4
    points=np.concatenate([p['points'] for p in paths]);widths=np.concatenate([p['widths'] for p in paths]);q=np.zeros(shape,np.float32)
    for y in range(0,shape[0],tile):
        for x in range(0,shape[1],tile):
            h,w=min(tile,shape[0]-y),min(tile,shape[1]-x)
            chosen=(points[:,0]>=y-halo)&(points[:,0]<y+h+halo)&(points[:,1]>=x-halo)&(points[:,1]<x+w+halo)
            if not chosen.any():continue
            center=np.zeros(((h+2*halo)*scale,(w+2*halo)*scale),bool);ww=np.zeros(center.shape,np.float32)
            ij=np.rint((points[chosen]-[y-halo,x-halo])*scale+(scale-1)/2).astype(int)
            valid=(ij[:,0]>=0)&(ij[:,0]<center.shape[0])&(ij[:,1]>=0)&(ij[:,1]<center.shape[1]);ij=ij[valid]
            center[ij[:,0],ij[:,1]]=True;np.maximum.at(ww,(ij[:,0],ij[:,1]),widths[chosen][valid])
            d,nearest=ndi.distance_transform_edt(~center,return_indices=True);support=(d<=ww[nearest[0],nearest[1]]*scale/2)&(ww[nearest[0],nearest[1]]>0)
            for hole in holes:
                cy,cx=hole['center'];ry,rx=hole['radii']
                if cy+ry<y-halo or cy-ry>y+h+halo or cx+rx<x-halo or cx-rx>x+w+halo:continue
                iy0=max(0,int((cy-ry-y+halo)*scale));iy1=min(center.shape[0],int((cy+ry-y+halo)*scale)+2)
                ix0=max(0,int((cx-rx-x+halo)*scale));ix1=min(center.shape[1],int((cx+rx-x+halo)*scale)+2)
                yy=y-halo+(np.arange(iy0,iy1)-(scale-1)/2)/scale;xx=x-halo+(np.arange(ix0,ix1)-(scale-1)/2)/scale
                support[iy0:iy1,ix0:ix1]&=(((yy[:,None]-cy)/ry)**2+((xx[None,:]-cx)/rx)**2>1)
            crop=support[halo*scale:(halo+h)*scale,halo*scale:(halo+w)*scale]
            q[y:y+h,x:x+w]=crop.reshape(h,scale,w,scale).mean((1,3))
    return q*specimen

def extend_geometry(source,seed,zero_guidance=False,specimen=None,short=False):
    rng=np.random.default_rng(seed);result=[];holes=[]
    for j,path in enumerate(source):
        p=np.asarray(path['points_yx'],float)
        if short:
            distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
            length=float(rng.uniform(min(12.,distance[-1]*.5),min(80.,distance[-1]*.95)))
            start=float(rng.uniform(0,max(distance[-1]-length,0)))
            at=np.linspace(start,start+length,max(16,int(np.ceil(length*4))+1))
            p=np.column_stack([np.interp(at,distance,p[:,k]) for k in [0,1]])
        n=len(p)
        # In a constant-cost continuum the shortest route is straight, not an
        # arbitrary axis/diagonal combination of the 8-neighbour lattice.
        straight=np.linspace(p[0],p[-1],n);ij=np.rint(straight).astype(int)
        straight_inside=specimen is None or specimen[tuple(ij.T)].all()
        if zero_guidance and straight_inside:p=straight
        else:
            smooth=ndi.gaussian_filter1d(p,12,axis=0,mode='nearest');delta_s=smooth-p
            delta_s*=np.minimum(1,3/(np.linalg.norm(delta_s,axis=1)+1e-9))[:,None]
            smooth=p+delta_s;smooth[0]=p[0];smooth[-1]=p[-1];p=smooth
        if path.get('parent') is not None:
            start=result[int(path['parent'])]['points'][int(path['parent_sample'])]
            p+=(start-p[0])[None,:]*(1-np.linspace(0,1,n))[:,None]
        arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];length=arc[-1]
        tangent=np.gradient(p,axis=0);tangent/=np.linalg.norm(tangent,axis=1)[:,None]+1e-9;normal=np.stack([-tangent[:,1],tangent[:,0]],1)
        z=ndi.gaussian_filter1d(rng.normal(size=n),max(2,n/30));z/=max(float(z.std()),1e-8)
        nominal=float(np.exp(rng.uniform(np.log(.65),np.log(2. if short else 6.))))
        width=np.clip(nominal*np.exp(.20*z),.55,7.5)
        taper_length=min(length*.22,max(3.,nominal*2))
        taper=np.minimum(1,np.sqrt(np.maximum(length-arc,0)/max(taper_length,1e-6)))
        if path['kind']=='main':taper*=np.minimum(1,np.sqrt(arc/max(taper_length,1e-6)))
        delta=.20*np.sin(np.linspace(0,2*np.pi,n)+rng.uniform(0,2*np.pi));delta*=np.minimum(1,np.minimum(arc,length-arc)/max(3*nominal,1))
        # Preserve exact branch attachment and its neighbouring parent geometry.
        childanchors=[c['parent_sample'] for c in source if c.get('parent')==j]
        for k in childanchors:
            distance=np.abs(arc-arc[int(k)]);delta*=np.minimum(1,distance/max(3*nominal,1));taper=np.maximum(taper,(distance<nominal).astype(float))
        shifted=p+(.5*width*delta)[:,None]*normal
        if path.get('parent') is not None:
            parent=result[int(path['parent'])];shifted[0]=parent['points'][int(path['parent_sample'])]
            root=float(parent['widths'][int(path['parent_sample'])]);fade=np.minimum(1,arc/max(3*nominal,1));width=root*(1-fade)+width*fade
        width=(width*taper).astype(np.float32)
        result.append(dict(points=shifted.astype(np.float32),widths=width,kind=path['kind'],parent=path.get('parent'),parent_sample=path.get('parent_sample')))
        if width.max()>3 and rng.random()<.35:
            k=int(np.argmax(width*np.sin(np.linspace(0,np.pi,n))));cy,cx=shifted[k]+normal[k]*(.28*width[k])
            holes.append(dict(center=[float(cy),float(cx)],radii=[float(.10*width[k]),float(.10*width[k])]))
    return result,holes

def render(image,specimen,q,dark,alpha,sigma):
    local=ndi.gaussian_filter(image,.5)
    return np.clip(image-ndi.gaussian_filter(q*alpha*np.maximum(local-dark,0),sigma)*specimen,0,1).astype(np.float32)

def prepare_material(m):
    dest=OUT/'data'/m;dest.mkdir(parents=True,exist_ok=True)
    done=dest/'COMPLETE.json'
    if done.exists():return
    sources=json.loads((old.OUT/'sources.json').read_text())[m];rows=old.read(BASE/m/'dataset/manifest.csv');train=[r for r in rows if r['split']=='train'];groups=sorted({r['background_group'] for r in train})
    records=[];geometry_audit=[];divrows=old.read(DIV/m/'dataset/manifest.csv')
    for gi,g in enumerate(groups):
        groupdir=dest/g;marker=groupdir/'COMPLETE.json'
        if marker.exists():
            d=json.loads(marker.read_text());records+=d['records'];geometry_audit+=d['geometry_audit'];continue
        gr=[r for r in train if r['background_group']==g];ref=gr[0];root=BASE/m/'dataset'
        image=tifffile.imread(root/ref['original']).astype(np.float32);specimen=tifffile.imread(root/ref['specimen']).astype(bool)
        metadata=json.loads((root/'backgrounds'/f'{g}_metadata.json').read_text());dark=float(metadata['dark_reference']);field=e.cached_native_field(image,specimen)
        for r in gr:
            if r['mode']=='control':continue
            q=tifffile.imread(root/r['coverage']).astype(np.float32)/16
            rng=np.random.default_rng(np.random.SeedSequence([20261113,old.MS.index(m),gi,int(r['seed']),13]));sigma=float(rng.uniform(.55,1.10))
            for vis,bounds in [('low',(.35,.60)),('high',(.70,.98))]:
                folder=groupdir/(r['id']+'_'+vis);alpha=float(rng.uniform(*bounds));syn=render(image,specimen,q,dark,alpha,sigma)
                ignore,_=e.supervision_ambiguity(image,specimen,syn,image-syn,q,field)
                # Labels remain exactly the pre-existing fractional geometry.
                source=next(v for v in sources['virtual'] if v['id']==r['id'])
                records.append(dict(id=r['id']+'_'+vis,kind='appearance',visibility=vis,group=g,
                    image=save(folder,'image',syn),label=source['label'],weight=save(folder,'weight',((~ignore)*255).astype(np.uint8)),scale=16,alpha=alpha,sigma=sigma))
        profiles=['short','fork'] if gi%2==0 else ['single','tree']
        for pi,profile in enumerate(profiles):
            src=DIV/m/'dataset/geometry'/f'{g}_{profile}'/'paths.json'
            guide=float(next(r['guidance'] for r in divrows if r['background_group']==g and r['profile']==profile))
            paths,holes=extend_geometry(json.loads(src.read_text()),20261113+10000*old.MS.index(m)+100*gi+pi,zero_guidance=guide==0,specimen=specimen,short=profile=='short')
            for path in paths:
                ij=np.rint(path['points']).astype(int);assert specimen[tuple(ij.T)].all(),'Extended path left specimen; inspect before training.'
            q=raster(paths,image.shape,specimen,holes);folder=groupdir/profile
            label=save(folder,'coverage',np.rint(q*16).astype(np.uint8));rng=np.random.default_rng(271+gi*100+pi)
            sigma=float(rng.uniform(.55,1.10));variants=[('geometry','original',(.65,.98)),('joint','low',(.35,.60)),('joint','high',(.70,.98))]
            js(folder/'geometry.json',dict(source=str(src),source_sha256=sha(src),guidance=guide,zero_guidance_straight_if_within_specimen=guide==0,holes=holes,paths=[dict(kind=p['kind'],points_yx=p['points'].tolist(),width_px=p['widths'].tolist(),parent=p['parent'],parent_sample=p['parent_sample']) for p in paths]))
            for kind,vis,bounds in variants:
                alpha=float(rng.uniform(*bounds));syn=render(image,specimen,q,dark,alpha,sigma);ignore,_=e.supervision_ambiguity(image,specimen,syn,image-syn,q,field);f=folder/vis
                records.append(dict(id=f'{g}_{profile}_{vis}',kind=kind,visibility=vis,group=g,profile=profile,
                    image=save(f,'image',syn),label=label,weight=save(f,'weight',((~ignore)*255).astype(np.uint8)),scale=16,alpha=alpha,sigma=sigma))
                assert np.isfinite(syn).all() and np.array_equal(syn[~specimen],image[~specimen])
            geometry_audit.append(dict(material=m,group=g,profile=profile,guidance=guide,source_sha256=sha(src),holes=len(holes),foreground_pixels=int((q>0).sum()),area=float(q.sum()),free_tip_widths=[float(p['widths'][-1]) for p in paths]))
        rr=[r for r in records if r['group']==g];aa=[r for r in geometry_audit if r['group']==g]
        js(marker,dict(records=rr,geometry_audit=aa));print('GENERATED',m,g,len(rr),flush=True)
    # Manual masks are used only on fitting14; negatives are label-assisted.
    negatives=[]
    for row in sources['real14']:
        a=np.load(row['image'],mmap_mode='r');b=np.load(row['label'],mmap_mode='r').astype(bool);h,w=b.shape
        specimen=a>.04;outside=ndi.uniform_filter(specimen.astype(float),size=256,mode='constant')
        blocked=ndi.maximum_filter(b.astype(np.uint8),size=273,mode='constant')
        candidates=[]
        for cy in range(128,h-127,64):
            for cx in range(128,w-127,64):
                if blocked[cy,cx] or outside[cy,cx]<.70:continue
                patch=np.asarray(a[cy-128:cy+128,cx-128:cx+128]);gy,gx=np.gradient(ndi.gaussian_filter(patch,.8));difficulty=float(np.quantile(np.hypot(gy,gx),.95))
                candidates.append((difficulty,cy-128,cx-128))
        # Keep difficult and ordinary negatives without any validation-image mining.
        ordered=sorted(candidates,reverse=True);chosen=ordered[:8]+ordered[-4:]
        used=set()
        for difficulty,y,x in chosen:
            if (y,x) in used:continue
            used.add((y,x));f=dest/'negatives'/row['id']/f'y{y}_x{x}';patch=np.array(a[y:y+256,x:x+256],np.float32)
            assert not b[y:y+256,x:x+256].any()
            negatives.append(dict(id=f'{row["id"]}_y{y}_x{x}',source_id=row['id'],ordinal=row['ordinal'],image=save(f,'image',patch),y=y,x=x,difficulty=difficulty,hard=(difficulty>=ordered[min(7,len(ordered)-1)][0]),source_label=row['label'],source_label_sha256=sha(row['label']),annotation_assisted=True))
    assert len(negatives)>=20,(m,len(negatives))
    for r in records:
        r['hashes']={k:sha(r[k]) for k in ['image','label','weight']}
    js(dest/'records.json',records);js(dest/'negatives.json',negatives);e.write_rows(dest/'geometry_audit.csv',geometry_audit)
    js(done,dict(material=m,records=len(records),negative_patches=len(negatives),negative_source_ids=sorted({r['source_id'] for r in negatives}),records_sha256=sha(dest/'records.json'),negative_manifest_sha256=sha(dest/'negatives.json'),source_code_sha256=sha(__file__),no_validation_negatives=True))
    print('DATA_COMPLETE',m,len(records),len(negatives),flush=True)

def checks():
    specimen=np.ones((128,128),bool);points=np.stack([np.full(321,64.),np.linspace(24,104,321)],1)
    path=dict(points=points,widths=np.full(321,4.,np.float32),kind='main')
    q4=raster([path],specimen.shape,specimen,scale=4);q8=raster([path],specimen.shape,specimen,scale=8)
    analytic=80*4+np.pi*4
    assert abs(float(q8.sum())-analytic)/analytic<.04
    assert abs(float(q8.sum())-float(q4.sum()))/analytic<.04
    a=np.full(specimen.shape,.6,np.float32);zero=np.zeros_like(a)
    assert np.array_equal(render(a,specimen,zero,.1,.8,.7),a)
    for alpha in [.35,.85]:assert np.isfinite(render(a,specimen,q4,.1,alpha,.7)).all()
    low=render(a,specimen,q4,.1,.35,.7);high=render(a,specimen,q4,.1,.85,.7);assert np.all(low>=high-1e-7)
    holed=raster([path],specimen.shape,specimen,[dict(center=[64,64],radii=[1,1])]);assert holed.sum()<q4.sum()
    js(OUT/'GEOMETRY_CHECKS.json',dict(result='passed',zero_opening_identity=True,straight_area_analytic=analytic,area_s4=float(q4.sum()),area_s8=float(q8.sum()),same_geometry_for_paired_grey=True,monotone_contrast=True,retained_material_reduces_occupancy=True))

def audit():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=old.read(old.ANNOT/'evaluation_manifest.csv');rows=[]
    for m in old.MS:
        base=old.read(BASE/m/'dataset/manifest.csv');origins=sorted({r['source'] for r in base if r['split']=='train'});real=sorted({r['source_volume'] for r in manifest if r['material']==m})
        rows.append(dict(material=m,synthetic_sources=origins,real_sources=real,exact_source_path_overlap=sorted(set(origins)&set(real)),note='Same material/specimen can remain correlated across loading states; matching slice number alone is not a matching source image.'))
    js(OUT/'SOURCE_AUDIT.json',dict(materials=rows,negative_policy='Archived Nov10 masks from fitting14 only; label-assisted, not label-free. Existing injected-geometry targets do not comprehensively label source defects.',old_study_sha256=sha(old.OUT/'FINAL_COMPLETE.json')))
    js(OUT/'PROTECTED.json',{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'output/manuscript/CURRENT_RELEASE.json',ROOT/'output/manuscript/XCT_virtual_crack_manuscript_v45.docx',ROOT/'output/manuscript/XCT_virtual_crack_supplement_v45.docx',old.OUT/'FINAL_COMPLETE.json']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--material',choices=old.MS);a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.material:prepare_material(a.material)
    else:audit();checks()
