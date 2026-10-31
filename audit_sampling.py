"""Independent saved-evidence checks; no learned inference or pose fitting."""
import gc,json
from datetime import datetime,timezone
from pathlib import Path
import cv2
import numpy as np
from run_dense import heavy_job
from dense_experiment import immutable_json
from run_sampling import inputs
from audit_area_annotations import quadrature
from audit_expanded_baseline import expected_extended,verify_extended,expected_same
from validate_dense_run_v2 import (ROOT,checked_ref,sha,require,compare,accepted,rgb_metrics,load_npz,
    verify_pairs,verify_evidence,verify_selected_donors,METHODS)

OUT=ROOT/'sampling_v1'


def expected_partitions(new,old,truth,policies):
    mask=new['mask'];pred=np.zeros((*mask.shape,3),np.uint8);pred[mask]=new['prediction_on_mask'];control=pred.copy();control[mask]=old['prediction_on_mask'];out={}
    for name,p in policies.items():
        a=accepted(new,mask,p);b=accepted(old,mask,p);out[name]={}
        for label,take in [('common',a&b),('gained',a&~b),('lost',b&~a)]:
            row=dict(pixels=int(take.sum()),fraction_of_mask=float(take.sum()/mask.sum()))
            if truth is not None:
                for key,image in [('new',pred),('control',control)]:
                    if (key=='new' and label=='lost') or (key=='control' and label=='gained'):continue
                    m=rgb_metrics(image,truth,take);row[key]={k:m[k] for k in ['pixels','mae_255','p90_rgb','over_20_fraction']}
            out[name][label]=row
    return out


def audit():
    pp=OUT/'execution_plan.json';plan=json.loads(pp.read_text());ph=sha(pp)
    for f,h in plan['signature'].items():checked_ref(ROOT,dict(path=f,sha256=h))
    for r in plan['baseline_plans']+plan['baseline_audits']+[plan['protocol']]:checked_ref(ROOT,r)
    require(plan['groups']==126 and plan['conditions']==216,'population changed')
    require(plan['policies']==json.loads((ROOT/'benchmark_v2/protocol.json').read_text())['policies'],'policy changed')
    cert=json.loads(checked_ref(ROOT,plan['certificate']).read_text())
    require(cert['signature']==plan['model'] and cert['validator_sha256']==plan['signature']['validate_sampling_model.py'],'certificate signature')
    require(cert['passed'] is True and cert['actual_model_executed'] is True and cert['device']=='mps' and cert['excluded_rgb_correspondence_invariance'] is True,'model preflight missing')
    require(len(cert['transforms'])==2 and all(t['matches']>=32 and t['median_error_px']<1.5 and t['p90_error_px']<3 and t['hidden_rgb_invariant'] is True for t in cert['transforms']),'model preflight thresholds')
    base_model=json.loads(checked_ref(ROOT,plan['baseline_plans'][0]).read_text())['model'];require(plan['model']['checkpoint_sha256']==base_model['checkpoint_sha256'],'weights changed')
    weights=None;projection=None;groups=[];conditions=[];missing=[]
    for scene in plan['scenes']:
        require(scene['split'] in ['development','train'],'held-out leak');images,allowed=inputs(scene);target=scene['target'];donors=[i for i in range(5) if i!=target];truth=images[target]
        if scene['loader']=='expanded':texture=load_npz(checked_ref(ROOT,scene['source']['scoring_only_strata']))['textured']
        else:
            gray=cv2.cvtColor(truth,cv2.COLOR_RGB2GRAY).astype(np.float32);x=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3,scale=1/8);y=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3,scale=1/8);texture=x*x+y*y>=16
        if weights is None:weights=quadrature(**scene['projection']);projection=scene['projection']
        require(scene['projection']==projection,'projection differs')
        for job in scene['jobs']:
            path=OUT/scene['scene']/'groups'/(job['key']+'.json')
            if not path.exists():missing.append(str(path.relative_to(ROOT)));continue
            r=json.loads(path.read_text());require(r['scene']==scene['scene'] and r['job']==job and r['signature']==dict(plan_sha256=ph,certificate=plan['certificate']),'group identity/signature')
            mask=load_npz(checked_ref(ROOT,job['mask']))[job['mask_key']];natural=job['kind']=='natural';reference=None if natural else truth
            require(not ((mask&allowed[target]).any() if natural else (mask&~allowed[target]).any()),'target visibility changed')
            require(set(r['methods'])==METHODS and set(r['graphs'])==METHODS,'methods changed')
            control=json.loads(checked_ref(ROOT,job['control']).read_text())
            count=verify_pairs(ROOT,r,images,allowed,mask,target,plan['model'])
            verify_pairs(ROOT,control,images,allowed,mask,target,base_model) # Prove identical input/mask controls despite heterogeneous old loaders.
            for pair in r['correspondences']:
                if pair['matches']:require(pair['matcher']['max_side']==960 and pair['matcher']['method']=='masked_loftr_outdoor_960','incorrect sampling treatment')
            results={}
            for method,d in r['methods'].items():
                e=load_npz(checked_ref(ROOT,d['evidence']));old=load_npz(checked_ref(ROOT,control['methods'][method]['evidence']));regs=d['registrations'];graph=r['graphs'][method]
                checked=verify_evidence(e,d['policies'],mask,plan['policies'],4,footprint=natural,truth=reference,graph=graph,registrations=regs)
                verify_evidence(old,control['methods'][method]['policies'],mask,plan['policies'],4,footprint=natural,truth=reference)
                checked['provenance']=verify_selected_donors(e,[images[i] for i in donors],[allowed[i] for i in donors],regs)
                compare(d['control_policies'],control['methods'][method]['policies'],'control policies')
                for evidence,key in [(e,'extended'),(old,'control_extended')]:
                    pred=np.zeros_like(truth);pred[mask]=evidence['prediction_on_mask']
                    for name,p in plan['policies'].items():verify_extended(d[key][name],expected_extended(pred,reference,accepted(evidence,mask,p),mask,weights,None if natural else texture))
                    del pred
                compare(d['partitions'],expected_partitions(e,old,reference,plan['policies']),'common/gained/lost')
                for i,reg in zip(donors,regs):
                    if reg.get('accepted') is True:require(reg['homography']==graph['poses'].get(str(i)),'pose differs')
                results[method]=checked;del e,old;gc.collect()
            groups.append(dict(scene=scene['scene'],job=job['key'],record_sha256=sha(path),pair_caches_verified=count,methods=results))
        for c in scene['conditions']:
            path=OUT/scene['scene']/'conditions'/(c['spec']['key']+'.json')
            if not path.exists():missing.append(str(path.relative_to(ROOT)));continue
            r=json.loads(path.read_text());require(r['condition']==c and r['scene']==scene['scene'] and r['signature']==dict(plan_sha256=ph,certificate=plan['certificate']),'condition identity')
            gp=checked_ref(ROOT,r['geometry']);require(gp==OUT/scene['scene']/'groups'/(c['job']+'.json'),'geometry link')
            geometry=json.loads(gp.read_text());baseline=json.loads(checked_ref(ROOT,c['baseline']).read_text());require(baseline['geometry']==geometry['job']['control'],'wrong shared baseline')
            expected={}
            for method,d in geometry['methods'].items():
                e=load_npz(checked_ref(ROOT,d['evidence']));mask=e['mask'];expected[method]={}
                for name,p in plan['policies'].items():
                    take=accepted(e,mask,p)[mask];expected[method][name]={}
                    for control,d in baseline['baselines'].items():
                        original=load_npz(checked_ref(ROOT,d['evidence']))['prediction_on_mask']
                        expected[method][name][control]=expected_same(e['prediction_on_mask'],original,truth[mask],take)
                del e
            compare(r['same_pixel_controls'],expected,'condition same-pixel control');conditions.append(dict(scene=scene['scene'],condition=c['spec']['key'],record_sha256=sha(path)))
        print('AUDITED',scene['scene'],flush=True);del images,allowed,texture,truth;gc.collect()
    complete=not missing and len(groups)==126 and len(conditions)==216
    result=dict(passed=True if complete else None,status='complete' if complete else 'incomplete',created_at_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=ph,validator_sha256=sha(Path(__file__)),groups=groups,conditions=conditions,missing=missing,matcher_executed=False)
    if complete:immutable_json(OUT/'audit_001.json',result)
    print(json.dumps(dict(passed=result['passed'],groups=len(groups),conditions=len(conditions),missing=len(missing))))


if __name__=='__main__':
    with heavy_job():audit()
