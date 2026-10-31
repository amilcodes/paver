"""Freeze per-frame ground, ego and nuisance masks without scoring any method.

Previous annotation/compiler versions remain unchanged. All masks are source
annotations; there is no algorithmic truth below the visible capture vehicle.
"""
import argparse,json,hashlib
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from benchmark import ROOT,digest,save_json
from repair import polygon_mask,nadir
from dataset_admission import effective_splits

CATEGORIES=('ego','ground','moving','shadow','privacy','uncertain')


def compile_masks(annotation):
    size=annotation['projection']['size'];shape=(size,size,3);compiled={};counts={}
    margin=annotation.get('boundary_margin_px',0)
    if not isinstance(margin,int) or margin<0 or margin>64:
        raise ValueError('Boundary margin must be an integer in [0,64]')
    for name in annotation['frame_order']:
        frame=annotation['frames'][name];parts={}
        for category in CATEGORIES:
            for poly in frame[category]:
                xy=np.asarray(poly)
                if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<3 or not np.isfinite(xy).all() or (xy<0).any() or (xy>1).any():
                    raise ValueError('Invalid normalized polygon')
            parts[category]=polygon_mask(shape,frame[category])
            compiled[name+'_'+category]=parts[category]
        nuisance=parts['moving']|parts['shadow']|parts['privacy']|parts['uncertain']
        excluded=parts['ego']|nuisance
        if margin:
            excluded=cv2.dilate(excluded.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*margin+1,2*margin+1)))>0
        compiled[name+'_excluded']=excluded
        compiled[name+'_allowed']=parts['ground']&~excluded
        assert compiled[name+'_allowed'].sum()>10000
        counts[name]={c:int(m.sum()) for c,m in parts.items()}
        counts[name].update(allowed=int(compiled[name+'_allowed'].sum()),boundary_margin=margin,
                            nuisance_inside_ground=int((nuisance&parts['ground']&~parts['ego']).sum()))
    footprint=compiled['target_ego'];compiled['region_footprint']=footprint
    assert footprint.any()
    region_counts={'footprint':dict(pixels=int(footprint.sum()),ratio_to_footprint=1.)}
    for name,polys in annotation['regions'].items():
        if name=='footprint':
            raise ValueError('Cannot replace the measured ego silhouette')
        for poly in polys:
            xy=np.asarray(poly)
            if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<3 or not np.isfinite(xy).all() or (xy<0).any() or (xy>1).any():
                raise ValueError('Invalid hidden-region polygon')
        raw=polygon_mask(shape,polys);mask=raw&compiled['target_allowed']
        assert mask.sum()>1000
        compiled['region_'+name]=mask
        region_counts[name]=dict(pixels=int(mask.sum()),ratio_to_footprint=float(mask.sum()/footprint.sum()),
                                 fraction_clipped=float(1-mask.sum()/raw.sum()))
    return compiled,counts,region_counts


def prepare(path,preview_dir=None,freeze=False):
    path=Path(path).resolve();a=json.loads(path.read_text())
    effective=effective_splits(ROOT/'dataset_v1');membership=effective['collections'][a['collection']]
    assert membership['split']==a['split']=='train'
    assert a['collection'] not in effective['quarantined_collections']
    active_quarantine=ROOT/'dataset_v1/active_quarantine.json'
    if active_quarantine.exists():
        assert a['collection'] not in json.loads(active_quarantine.read_text())['collections']
    assert a['group']==effective['effective_groups'][a['collection']]
    data=ROOT/'dataset_v1';cohort=json.loads((data/'cohorts/initial_100_v1.json').read_text())
    assert any(t['collection']==a['collection'] and t['id']==a['target_id'] and t['split']=='train' for t in cohort['targets'])
    assert len(a['frame_order'])==len(set(a['frame_order']))==len(a['frames'])
    ids=set()
    for name in a['frame_order']:
        frame=a['frames'][name];ids.add(frame['id'])
        assert digest(ROOT/frame['image'])==frame['sample_sha256']
        metadata=json.loads((data/'metadata'/(frame['id']+'.json')).read_text())
        assert metadata['collection']==a['collection'] and metadata['properties']['license']=='etalab-2.0'
        assert digest(data/'metadata'/(frame['id']+'.json'))==frame['metadata_sha256']
    assert len(ids)==len(a['frames']) and a['frames']['target']['id']==a['target_id']
    folder=ROOT/'benchmark_v3/prepared'/a['scene_id'];existing=folder/'manifest.json'
    if existing.exists() and json.loads(existing.read_text())['annotation_sha256']!=digest(path):
        raise RuntimeError('Frozen annotation changed; create a new scene version')
    masks,counts,regions=compile_masks(a)
    if preview_dir:
        preview_dir=Path(preview_dir);preview_dir.mkdir(parents=True,exist_ok=True)
        # Internal annotation QA, not a public visual report or a result image.
        for name in a['frame_order']:
            source=ROOT/a['frames'][name]['image']
            im=nadir(np.asarray(Image.open(source)),a['projection']['size'],a['projection']['fov'])
            overlay=im.copy()
            for suffix,color in [('allowed',[0,85,0]),('ego',[100,0,0]),('moving',[80,0,80]),('shadow',[90,50,0])]:
                m=masks[name+'_'+suffix];overlay[m]=(im[m]*.5+color).clip(0,255)
            if name=='target':
                for region in a['regions']:
                    m=masks['region_'+region];overlay[m]=(im[m]*.4+[0,70,115]).clip(0,255)
            Image.fromarray(overlay).resize((1200,1200)).save(preview_dir/(name+'.jpg'),quality=92)
    if freeze:
        assert a['status']=='reviewed_frozen_before_scoring'
        if existing.exists():
            prior=json.loads(existing.read_text());assert digest(folder/'masks.npz')==prior['mask_sha256']
            with np.load(folder/'masks.npz') as z:
                assert set(z.files)==set(masks)
                for name,m in masks.items():assert np.array_equal(m,z[name])
            return prior
        folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/'masks.npz',**masks)
        manifest=dict(scene_id=a['scene_id'],collection=a['collection'],split='train',
                      annotation_path=str(path.relative_to(ROOT)),annotation_sha256=digest(path),
                      compiler='prepare_annotated_case_v2.py',compiler_sha256=digest(Path(__file__)),
                      mask_sha256=digest(folder/'masks.npz'),frames=counts,regions=regions,
                      cohort_sha256=digest(data/'cohorts/initial_100_v1.json'),
                      evaluation_status='not_run',no_under_car_ground_truth=True,
                      note='Moving traffic, vehicle shadows, privacy regions and uncertain surfaces are excluded from registration and hidden ground truth, but never counted as ego footprint.')
        save_json(existing,manifest)
    else:manifest=dict(scene_id=a['scene_id'],frames=counts,regions=regions,frozen=False)
    print(json.dumps(manifest,indent=2));return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('annotation');p.add_argument('--preview-dir');p.add_argument('--freeze',action='store_true')
    a=p.parse_args();prepare(a.annotation,a.preview_dir,a.freeze)
