"""Independent saved-result audit of the expanded baseline. No matching."""
import gc,json,math
from pathlib import Path
from datetime import datetime,timezone
import cv2
import numpy as np
from PIL import Image
from repair import nadir
from run_dense import heavy_job
from dense_experiment import immutable_json
from audit_area_annotations import quadrature
from audit_controlled_benchmark import bands,independent_damage
from validate_dense_run_v2 import (ROOT,checked_ref,sha,require,compare,accepted,rgb_metrics,gradient_metrics,load_npz,
    verify_certificate,verify_pairs,verify_evidence,verify_selected_donors,array_digest,METHODS)

RUN=ROOT/'controlled_v2/baseline_v1'


def expected_extended(pred,truth,take,mask,w,texture=None):
    n=int(np.count_nonzero(mask));k=int(np.count_nonzero(take));total=float(np.sum(w[mask]));used=float(np.sum(w[take]))
    out=dict(mask_pixels=n,accepted_pixels=k,mask_solid_angle_sr=total,accepted_solid_angle_sr=used,angular_coverage=used/total,angular_abstention=1-used/total)
    if truth is None:return out
    summed=np.abs(pred[take].astype(np.int32)-truth[take].astype(np.int32)).sum(axis=1);weights=w[take]
    out.update(angular_rgb_mae_255=float(np.sum(summed*weights)/(3*used)) if used else None,
        angular_rgb_mae_le_20_fraction_of_mask=float(weights[summed<=60].sum()/total),
        accepted_rgb_mae_le_threshold_pixels={str(t):int(np.count_nonzero(summed<=3*t)) for t in [10,20]},strata={})
    for name,partition in [('textured',texture),('low_texture',~texture)]:
        population=int(np.count_nonzero(mask&partition));chosen=take&partition;m=rgb_metrics(pred,truth,chosen)
        out['strata'][name]=dict(mask_pixels=population,accepted_pixels=m['pixels'],coverage=m['pixels']/population if population else None,
            rgb_mae_255=m['mae_255'],over_20_fraction=m['over_20_fraction'])
    return out


def verify_extended(reported,expected):
    compare(reported,expected,'extended')
    for key in ['mask_solid_angle_sr','accepted_solid_angle_sr','angular_coverage','angular_abstention']:
        require(math.isclose(reported[key],expected[key],rel_tol=1e-7,abs_tol=1e-8),'angular metric differs')


def expected_same(pred,control,truth,take):
    a=np.abs(pred[take].astype(np.int32)-truth[take].astype(np.int32)).sum(1);b=np.abs(control[take].astype(np.int32)-truth[take].astype(np.int32)).sum(1);n=len(a)
    return dict(pixels=n,source_mae_255=float(a.sum()/(3*n)) if n else None,control_mae_255=float(b.sum()/(3*n)) if n else None,
        source_better_pixels=int(np.count_nonzero(a<b)),source_worse_pixels=int(np.count_nonzero(a>b)),equal_pixels=int(np.count_nonzero(a==b)))


def audit():
    pp=RUN/'execution_plan.json';plan=json.loads(pp.read_text());ph=sha(pp)
    for p,h in plan['signature'].items():checked_ref(ROOT,dict(path=p,sha256=h))
    for key in ['protocol','annotation_audit','candidate_graph']:checked_ref(ROOT,plan[key])
    for r in plan['comparison_controls'].values():checked_ref(ROOT,r)
    require(plan['policies']==json.loads((ROOT/'benchmark_v2/protocol.json').read_text())['policies'],'policy changed')
    require(len(plan['scenes'])==6 and plan['controlled_masks']==72 and plan['conditions']==144 and plan['natural_masks']==6,'plan population')
    graph=json.loads(checked_ref(ROOT,plan['candidate_graph']).read_text());links={(e['target'],e['donor']) for e in graph['edges']}
    groups=[];conditions=[];missing=[];visual=[];w=None;projection=None
    for scene in plan['scenes']:
        require(scene['split'] in ['development','train'],'test leakage');m=json.loads(checked_ref(ROOT,scene['manifest']).read_text());a=json.loads(checked_ref(ROOT,scene['annotation']).read_text())
        require(scene['conditions']==m['conditions'] and scene['projection']==m['projection'],'frozen conditions differ')
        parts=load_npz(checked_ref(ROOT,scene['masks']));als=[parts[r['name']+'_allowed'] for r in scene['frames']];target=scene['target'];donors=[i for i in range(5) if i!=target]
        require(len(scene['frames'])==5 and len({r['id'] for r in scene['frames']})==5,'source population')
        for r in scene['frames']:
            expected=a['frames'][r['name']];require(r['id']==expected['id'] and r['sha256']==expected['sample_sha256'],'frame identity mismatch')
        for i in donors:require((scene['frames'][target]['id'],scene['frames'][i]['id']) in links,'not a metadata candidate')
        images=[nadir(np.asarray(Image.open(checked_ref(ROOT,r)).convert('RGB')),**scene['projection']) for r in scene['frames']];truth=images[target]
        texture=load_npz(checked_ref(ROOT,scene['scoring_only_strata']))['textured']
        if w is None:w=quadrature(**scene['projection']);projection=scene['projection']
        require(scene['projection']==projection,'mixed projection unsupported in current angular cache')
        folder=RUN/scene['scene'];require(len(scene['jobs'])==13,'job population')
        expected_jobs=[dict(key=s['key'],mask_key='hidden_'+s['key'],kind='controlled',mask_sha256=s['sha256']) for s in m['mask_specs']]+[dict(key='footprint',mask_key='region_footprint',kind='natural')]
        require(scene['jobs']==expected_jobs,'job masks differ from prepared manifest')
        for job in scene['jobs']:
            path=folder/'groups'/(job['key']+'.json')
            if not path.exists():missing.append(str(path.relative_to(ROOT)));continue
            record=json.loads(path.read_text());require(record['scene']==scene['scene'] and record['area']==scene['area'] and record['job']==job,'group identity')
            require(record['signature']['plan_sha256']==ph,'plan signature');verify_certificate(ROOT,record['signature']['certificate'],plan)
            require(set(record['methods'])==METHODS and set(record['graphs'])==METHODS,'method population')
            mask=parts[job['mask_key']];natural=job['kind']=='natural'
            require(not ((mask&als[target]).any() if natural else (mask&~als[target]).any()),'truth visibility')
            if not natural:require(array_digest(mask)==job['mask_sha256'],'mask hash')
            count=verify_pairs(ROOT,record,images,als,mask,target,plan['model']);methods={}
            for method,data in record['methods'].items():
                e=load_npz(checked_ref(ROOT,data['evidence']));regs=data['registrations'];g=record['graphs'][method]
                checked=verify_evidence(e,data['policies'],mask,plan['policies'],4,footprint=natural,truth=None if natural else truth,graph=g,registrations=regs)
                checked['provenance']=verify_selected_donors(e,[images[i] for i in donors],[als[i] for i in donors],regs)
                pred=np.zeros_like(truth);pred[mask]=e['prediction_on_mask']
                for name,p in plan['policies'].items():verify_extended(data['extended'][name],expected_extended(pred,None if natural else truth,accepted(e,mask,p),mask,w,None if natural else texture))
                for i,r in zip(donors,regs):
                    if r.get('accepted') is True:require(r['homography']==g['poses'].get(str(i)),'accepted pose differs')
                    visual.append(dict(scene=scene['scene'],target=scene['frames'][target]['id'],donor=scene['frames'][i]['id'],job=job['key'],kind=job['kind'],method=method,
                        geometry_accepted=r.get('accepted') is True,selected_pixels=int((mask&(e['provenance']==donors.index(i))).sum()),record_sha256=sha(path)))
                methods[method]=checked;del pred,e;gc.collect()
            groups.append(dict(scene=scene['scene'],job=job['key'],kind=job['kind'],record_sha256=sha(path),pair_caches_verified=count,methods=methods))
        for spec in scene['conditions']:
            path=folder/'conditions'/(spec['key']+'.json')
            if not path.exists():missing.append(str(path.relative_to(ROOT)));continue
            record=json.loads(path.read_text());require(record['signature']['plan_sha256']==ph and record['scene']==scene['scene'] and record['condition']==spec,'condition identity')
            mask=parts[spec['mask_key']];gp=checked_ref(ROOT,record['geometry']);require(gp==folder/'groups'/(spec['mask_key'].removeprefix('hidden_')+'.json'),'wrong geometry reference')
            geometry=json.loads(gp.read_text());damaged=independent_damage(truth,mask,spec['family'],spec['severity']);blank=damaged.copy();blank[mask]=0
            controls=dict(as_damaged=damaged,telea_r3=cv2.inpaint(blank,mask.astype(np.uint8),3,cv2.INPAINT_TELEA))
            require(set(record['baselines'])==set(controls),'baseline population')
            for name,pred in controls.items():
                data=record['baselines'][name];saved=load_npz(checked_ref(ROOT,data['evidence']))
                require(set(saved)=={'prediction_on_mask'} and np.array_equal(saved['prediction_on_mask'],pred[mask]),'baseline bytes differ')
                compare(data['metrics'],dict(coverage=1.,abstention=0.,accepted_pixels=int(mask.sum()),error=rgb_metrics(pred,truth,mask),gradient=gradient_metrics(pred,truth,mask),appearance=bands(pred,truth,mask,mask)))
                verify_extended(data['extended'],expected_extended(pred,truth,mask,mask,w,texture))
            require(set(record['same_pixel_controls'])==METHODS,'same-pixel method population')
            for method,data in geometry['methods'].items():
                e=load_npz(checked_ref(ROOT,data['evidence']));pred=np.zeros_like(truth);pred[mask]=e['prediction_on_mask']
                expected={name:{control:expected_same(pred,baseline,truth,accepted(e,mask,p)) for control,baseline in controls.items()} for name,p in plan['policies'].items()}
                compare(record['same_pixel_controls'][method],expected,'same-pixel controls');del e,pred
            conditions.append(dict(scene=scene['scene'],condition=spec['key'],record_sha256=sha(path)));del damaged,blank,controls;gc.collect()
        print('AUDITED',scene['scene'],flush=True);del images,truth,parts,als,texture;gc.collect()
    complete=not missing and len(groups)==78 and len(conditions)==144
    result=dict(passed=True if complete else None,status='complete' if complete else 'incomplete',created_at_utc=datetime.now(timezone.utc).isoformat(),
        plan_sha256=ph,validator_sha256=sha(Path(__file__)),checked_geometry_groups=len(groups),checked_conditions=len(conditions),missing=missing,groups=groups,conditions=conditions,
        matcher_executed=False,limitation='Evidence/provenance/metric audit, not calibrated geometric correctness. No target truth for natural occlusion; reused routes are development/train.')
    if complete:
        immutable_json(RUN/'audit_001.json',result)
        immutable_json(ROOT/'network_v1/visual_evidence_v2.json',dict(plan_sha256=ph,audit_sha256=sha(RUN/'audit_001.json'),edges=visual,
            scope='Six additional annotated anchors and query-specific decisions; metadata candidates elsewhere remain unverified.'))
    print(json.dumps({k:v for k,v in result.items() if k not in ['groups','conditions']}));return result


if __name__=='__main__':
    with heavy_job():audit()
