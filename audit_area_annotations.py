"""Independent source-mask, angular area, split and condition audit. No matcher."""
import json,math,gc
from pathlib import Path
from datetime import datetime,timezone
import cv2
import numpy as np
from PIL import Image
from validate_dense_run_v2 import ROOT,checked_ref,sha,require,array_digest,load_npz
from dense_experiment import immutable_json
from repair import nadir
from dataset_admission import effective_splits
from run_dense import heavy_job

OUT=ROOT/'controlled_v2';CATEGORIES=['ego','ground','moving','shadow','privacy','uncertain']


def raster(polygons,size):
    out=np.zeros((size,size),np.uint8)
    for polygon in polygons:
        xy=np.array(polygon,np.float64)
        require(xy.ndim==2 and xy.shape[1]==2 and len(xy)>=3 and np.isfinite(xy).all() and (xy>=0).all() and (xy<=1).all(),'invalid polygon')
        cv2.fillPoly(out,[np.rint(xy*(size-1)).astype(np.int32)],255)
    return out>0


def verify_source_masks(a,parts):
    size=a['projection']['size'];margin=a['boundary_margin_px'];gm=a['ground_margin_px']
    for name in a['frame_order']:
        masks={c:raster(a['frames'][name][c],size) for c in CATEGORIES}
        for c,m in masks.items():require(np.array_equal(parts[name+'_'+c],m),'raw category mismatch')
        excluded=np.logical_or.reduce([masks[c] for c in ['ego','moving','shadow','privacy','uncertain']])
        if margin:excluded=cv2.dilate(excluded.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*margin+1,2*margin+1)))>0
        ground=cv2.erode(masks['ground'].astype(np.uint8),np.ones((2*gm+1,2*gm+1),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
        require(np.array_equal(parts[name+'_excluded'],excluded),'excluded mask mismatch')
        require(np.array_equal(parts[name+'_allowed'],ground&~excluded),'allowed mask admits excluded or unreviewed pixels')
        require(parts[name+'_allowed'].sum()>10000,'insufficient eligible surface')
    require(np.array_equal(parts['region_footprint'],parts['target_ego']),'footprint mask differs')
    for location,polys in a['regions'].items():require(np.array_equal(parts['region_'+location],raster(polys,size)&parts['target_allowed']),'basis mask differs')


def quadrature(size,fov):
    # Independent 2x2 Gauss integration of the pinhole solid-angle Jacobian.
    f=size/(2*math.tan(math.radians(fov)/2));c=np.arange(size,dtype=np.float64)-(size-1)/2;w=np.zeros((size,size),np.float64)
    for dx in [-1/math.sqrt(12),1/math.sqrt(12)]:
        for dy in [-1/math.sqrt(12),1/math.sqrt(12)]:w+=f/((c[None,:]+dx)**2+(c[:,None]+dy)**2+f*f)**1.5/4
    return w


def audit():
    pp=OUT/'benchmark_protocol.json';protocol=json.loads(pp.read_text());selection=json.loads(checked_ref(ROOT,protocol['selection']).read_text());effective=effective_splits(ROOT/'dataset_v1')
    graph=json.loads(checked_ref(ROOT,protocol['graph']).read_text());links={(x['target'],x['donor']) for x in graph['edges']}
    selected={f"{t['area']}_{t['index']:03d}":t for t in selection['targets']};reports=[];seen=set();weight_cache=None
    for record in protocol['prepared']:
        manifest=json.loads(checked_ref(ROOT,record).read_text());a=json.loads(checked_ref(ROOT,manifest['annotation']).read_text());name=a['scene_id'];sel=selected[name]
        for key in ['compiler','base_compiler','corruption_source']:checked_ref(ROOT,manifest[key])
        for key in ['source','source_manifest','selection']:checked_ref(ROOT,a[key])
        require(a['status']=='reviewed_frozen_before_scoring' and a['no_model_outputs_used'] is True,'review not frozen')
        require(a['target_id']==sel['target_id'] and a['split']==sel['split']==effective['collections'][a['collection']]['split'],'selection/split differs')
        require(a['split'] in ['development','train'] and a['collection'] not in effective['quarantined_collections'],'held-out or quarantined source')
        require([a['frames'][n]['id'] for n in a['frame_order']]==[r['id'] for r in sel['frames']],'selected frames differ')
        for n,r in a['frames'].items():
            checked_ref(ROOT,dict(path=r['image'],sha256=r['sample_sha256']));meta=json.loads(checked_ref(ROOT,dict(path=r['metadata_path'],sha256=r['metadata_sha256'])).read_text())
            require(meta['id']==r['id'] and meta['collection']==a['collection'] and meta['properties']['license']==r['license']=='etalab-2.0','source provenance mismatch')
            require(meta['properties']['geovisio:producer']==r['producer'] and meta['properties']['datetime']==r['capture_time'],'source attribution/time differs')
            if n!='target':require((a['target_id'],r['id']) in links,'donor is not in candidate graph')
            seen.add(r['id'])
        parts=load_npz(checked_ref(ROOT,manifest['masks']));verify_source_masks(a,parts)
        for k,v in parts.items():require(v.dtype==bool and v.shape==(a['projection']['size'],)*2,'mask shape/dtype differs')
        if weight_cache is None:weight_cache=quadrature(**a['projection'])
        weights=weight_cache;foot=parts['region_footprint'];foot_sr=float(weights[foot].sum())
        require(math.isclose(foot_sr,manifest['obstruction']['solid_angle_sr'],rel_tol=1e-8,abs_tol=1e-9),'footprint angular area differs')
        clean=nadir(np.asarray(Image.open(ROOT/a['frames']['target']['image']).convert('RGB')),**a['projection']);gray=cv2.cvtColor(clean,cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3,scale=1/8);gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3,scale=1/8);texture=gx*gx+gy*gy>=16
        strata=load_npz(checked_ref(ROOT,manifest['scoring_only_strata']));require(set(strata)=={'textured'} and np.array_equal(texture,strata['textured']),'scoring-only texture strata differ')
        require(len(manifest['mask_specs'])==12 and len(manifest['conditions'])==24,'condition population differs')
        specifications={x['key']:x for x in manifest['mask_specs']}
        for spec in specifications.values():
            mask=parts['hidden_'+spec['key']];basis=parts['region_'+spec['location']]
            require(not (mask&~basis).any() and not (mask&~parts['target_allowed']).any() and mask.sum()>=500,'invalid controlled mask')
            require(array_digest(mask)==spec['sha256'] and int(mask.sum())==spec['pixels'],'mask hash/count differs')
            sr=float(weights[mask].sum())
            for key,value in [('solid_angle_sr',sr),('sphere_fraction',sr/(4*math.pi)),('projected_area_ratio_to_obstruction',mask.sum()/foot.sum()),('angular_area_ratio_to_obstruction',sr/foot_sr),('scoring_only_textured_fraction',texture[mask].mean())]:
                require(math.isclose(value,spec[key],rel_tol=1e-8,abs_tol=1e-9),'measurement differs: '+key)
            if spec['severity']=='mild':
                other=parts['hidden_'+spec['location']+'_'+spec['shape']+'_severe'];require(not(mask&~other).any(),'severity masks not nested')
        expected={(loc,family,severity) for loc in a['regions'] for family in ['opaque_disc','opaque_vehicle','missing_strip','blur','clipping','jpeg'] for severity in ['mild','severe']}
        require({(c['location'],c['family'],c['severity']) for c in manifest['conditions']}==expected,'corruption conditions differ')
        for c in manifest['conditions']:
            shape='vehicle' if c['family']=='opaque_vehicle' else 'strip' if c['family']=='missing_strip' else 'disc'
            require(c['mask_key']==f"hidden_{c['location']}_{shape}_{c['severity']}",'corruption mask mapping differs')
        reports.append(dict(scene_id=name,source_frames=5,conditions=24,mask_specs=12,minimum_allowed_source_pixels=min(int(parts[n+'_allowed'].sum()) for n in a['frame_order']),
            projected_area_ratio_range=[min(s['projected_area_ratio_to_obstruction'] for s in specifications.values()),max(s['projected_area_ratio_to_obstruction'] for s in specifications.values())],
            solid_angle_sr_range=[min(s['solid_angle_sr'] for s in specifications.values()),max(s['solid_angle_sr'] for s in specifications.values())],
            scoring_only_texture_fraction_range=[min(s['scoring_only_textured_fraction'] for s in specifications.values()),max(s['scoring_only_textured_fraction'] for s in specifications.values())],
            manifest_sha256=record['sha256']))
        del parts,clean,gray,gx,gy,texture,strata;gc.collect()
    require(len(reports)==6 and len(seen)==30,'anchor/source population differs')
    old=json.loads(checked_ref(ROOT,protocol['existing_controlled_v1']).read_text())
    for p,h in old['signature'].items():checked_ref(ROOT,dict(path=p,sha256=h))
    result=dict(passed=True,created_at_utc=datetime.now(timezone.utc).isoformat(),validator_sha256=sha(Path(__file__)),protocol_sha256=sha(pp),
        anchors=6,reviewed_source_frames=30,conditions=144,distinct_masks=72,model_executed=False,new_recovery_runs=0,prior_benchmark_sources_unchanged=True,results=reports,
        limitations=['Manual conservative road/obstruction envelopes, not a pixel-perfect semantic ground truth.','Six additional reused-route anchors are development/train, not independent test.','Real occlusion truth remains unavailable; prepared synthetic tests are not executed recovery evidence.'])
    immutable_json(OUT/'annotation_audit_001.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    with heavy_job():audit()
