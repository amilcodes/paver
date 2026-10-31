"""Independent metric/provenance audit of the frozen controlled battery.

Does not execute matching. Reuses the previously tested independent RGB/gradient
and selected-donor validator, and rejects incomplete batteries as incomplete.
"""
import gc,json
from pathlib import Path
from datetime import datetime,timezone
import cv2
import numpy as np
from PIL import Image
from repair import nadir
from dense_experiment import immutable_json
from validate_dense_run_v2 import (ROOT,sha,checked_ref,require,compare,accepted,rgb_metrics,gradient_metrics,
    load_npz,verify_evidence,verify_selected_donors,verify_pairs,verify_certificate,array_digest,METHODS)

OUT=ROOT/'controlled_v1'


def bands(pred,truth,take,mask):
    errors=np.abs(pred[take].astype(np.int32)-truth[take].astype(np.int32)).sum(1)
    counts={str(t):int(np.count_nonzero(errors<=3*t)) for t in [10,20]}
    gray=cv2.cvtColor(truth,cv2.COLOR_RGB2GRAY).astype(np.float32)
    dx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3,scale=1/8);dy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3,scale=1/8)
    textured=(dx*dx+dy*dy)>=16
    return dict(mask_pixels=int(mask.sum()),accepted_pixels=int(take.sum()),accepted_rgb_mae_le_threshold_pixels=counts,
        fraction_of_mask_with_rgb_mae_le_threshold={k:n/max(1,int(mask.sum())) for k,n in counts.items()},
        accepted_textured_pixels=int((take&textured).sum()),accepted_low_texture_pixels=int((take&~textured).sum()))


def independent_damage(truth,mask,family,severity):
    pred=truth.copy();severe=severity=='severe'
    if family in ['opaque_disc','missing_strip']:pred[mask]=0
    elif family=='opaque_vehicle':pred[mask]=32
    elif family=='blur':pred[mask]=cv2.GaussianBlur(truth,(0,0),32 if severe else 12)[mask]
    elif family=='clipping':pred[mask]=np.minimum(255,truth[mask].astype(np.float64)*(3 if severe else 1.8)+(100 if severe else 60)).astype(np.uint8)
    elif family=='jpeg':
        ok,data=cv2.imencode('.jpg',truth[:,:,::-1],[cv2.IMWRITE_JPEG_QUALITY,5 if severe else 20]);require(ok,'JPEG encoding failed')
        pred[mask]=cv2.imdecode(data,cv2.IMREAD_COLOR)[:,:,::-1][mask]
    else:raise ValueError('Unknown condition')
    return pred


def audit():
    pp=OUT/'execution_plan.json';plan=json.loads(pp.read_text());planhash=sha(pp)
    for p,h in plan['signature'].items():checked_ref(ROOT,dict(path=p,sha256=h))
    dataset=json.loads(checked_ref(ROOT,plan['dataset_audit']).read_text());require(dataset.get('passed') is True,'dataset audit failed')
    graph=json.loads(checked_ref(ROOT,plan['graph']).read_text());links={(e['target'],e['donor']) for e in graph['edges']}
    require(len(graph['nodes'])==247,'graph node population')
    require(plan['policies']==json.loads((ROOT/'benchmark_v2/protocol.json').read_text())['policies'],'policies changed')
    require(plan['scenario_count']==72 and plan['distinct_geometry_masks']==36,'population changed')
    groups=[];scenarios=[];missing=[];visual=[]
    for area in plan['areas']:
        folder=OUT/area['area'];target=area['target'];donors=[i for i in range(len(area['frames'])) if i!=target]
        require(area['split'] in ['train','development'],'held-out area used')
        for r in area['inputs']:checked_ref(ROOT,r)
        for i in donors:require((area['frames'][target]['id'],area['frames'][i]['id']) in links,'non-candidate donor')
        bundle=load_npz(checked_ref(ROOT,area['masks']));allowed=[bundle[f'allowed_{i}'] for i in range(len(area['frames']))]
        images=[nadir(np.asarray(Image.open(checked_ref(ROOT,r)).convert('RGB')),**area['projection']) for r in area['frames']];truth=images[target]
        for spec in area['mask_specs']:
            mask=bundle[spec['key']];require(mask.dtype==bool and not (mask&~allowed[target]).any(),'mask outside reviewed ground')
            require(array_digest(mask)==spec['sha256'] and int(mask.sum())==spec['pixels'],'mask digest/count')
            path=folder/'groups'/(spec['key']+'.json')
            if not path.exists():missing.append(str(path.relative_to(ROOT)));continue
            record=json.loads(path.read_text());require(record['mask_key']==spec['key'] and record['area']==area['area'],'group identity')
            require(record['signature']['plan_sha256']==planhash,'plan signature');verify_certificate(ROOT,record['signature']['certificate'],plan)
            require(set(record['methods'])==METHODS and set(record['graphs'])==METHODS,'method population')
            count=verify_pairs(ROOT,record,images,allowed,mask,target,plan['model']);methods={}
            for method,data in record['methods'].items():
                e=load_npz(checked_ref(ROOT,data['evidence']));regs=data['registrations'];g=record['graphs'][method]
                result=verify_evidence(e,data['policies'],mask,plan['policies'],len(donors),footprint=False,truth=truth,graph=g,registrations=regs)
                result['provenance']=verify_selected_donors(e,[images[i] for i in donors],[allowed[i] for i in donors],regs)
                pred=np.zeros_like(truth);pred[mask]=e['prediction_on_mask']
                for name,p in plan['policies'].items():compare(data['appearance'][name],bands(pred,truth,accepted(e,mask,p),mask),'appearance')
                for i,r in zip(donors,regs):
                    if r.get('accepted') is True:require(r['homography']==g['poses'].get(str(i)),'accepted pose differs from graph')
                    visual.append(dict(area=area['area'],target=area['frames'][target]['id'],donor=area['frames'][i]['id'],
                        mask_key=spec['key'],method=method,geometry_accepted=r.get('accepted') is True,
                        selected_pixels=int((mask&(e['provenance']==donors.index(i))).sum()),
                        status='query_specific_geometry_evidence_not_accuracy',record=dict(path=str(path.relative_to(ROOT)),sha256=sha(path))))
                methods[method]=result;del e,pred;gc.collect()
            groups.append(dict(area=area['area'],mask_key=spec['key'],sha256=sha(path),pair_caches_verified=count,methods=methods))
        for spec in area['scenarios']:
            path=folder/'scenarios'/(spec['key']+'.json')
            if not path.exists():missing.append(str(path.relative_to(ROOT)));continue
            record=json.loads(path.read_text());mask=bundle[spec['mask_key']]
            require(all(record[k]==v for k,v in spec.items()) and record['area']==area['area'],'scenario identity')
            require(record['signature']['plan_sha256']==planhash,'scenario plan signature')
            require(checked_ref(ROOT,record['geometry'])==folder/'groups'/(spec['mask_key']+'.json'),'wrong geometry reference')
            damaged=independent_damage(truth,mask,spec['family'],spec['severity']);empty=damaged.copy();empty[mask]=0
            telea=cv2.inpaint(empty,mask.astype(np.uint8),3,cv2.INPAINT_TELEA)
            require(set(record['baselines'])=={'as_damaged','telea_r3'},'baseline population')
            for name,expected in [('as_damaged',damaged),('telea_r3',telea)]:
                data=record['baselines'][name];e=load_npz(checked_ref(ROOT,data['evidence']))
                require(set(e)=={'prediction_on_mask'} and np.array_equal(e['prediction_on_mask'],expected[mask]),'baseline pixels differ')
                measured=dict(coverage=1.,abstention=0.,accepted_pixels=int(mask.sum()),error=rgb_metrics(expected,truth,mask),
                    gradient=gradient_metrics(expected,truth,mask),appearance=bands(expected,truth,mask,mask))
                compare(data['metrics'],measured,'baseline')
            scenarios.append(dict(area=area['area'],key=spec['key'],sha256=sha(path)))
            del damaged,empty,telea;gc.collect()
        del images,truth,bundle;gc.collect();print('AUDITED',area['area'],flush=True)
    complete=not missing and len(groups)==36 and len(scenarios)==72
    result=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),passed=True if complete else None,status='complete' if complete else 'incomplete',
        plan_sha256=planhash,validator_sha256=sha(Path(__file__)),checked_geometry_masks=len(groups),checked_scenarios=len(scenarios),
        missing=missing,geometry=groups,scenarios=scenarios,matcher_executed=False,
        limitation='Checks stored evidence, masked inputs and selected donor pixels; does not establish real under-vehicle truth or calibrated confidence.')
    if complete:
        immutable_json(OUT/'audit_001.json',result)
        immutable_json(ROOT/'network_v1/visual_evidence_v1.json',dict(plan_sha256=planhash,audit_sha256=sha(OUT/'audit_001.json'),edges=visual,
            scope='Three annotated anchors, four donors each, query/method-specific decisions. Metadata candidates elsewhere remain unverified.'))
    print(json.dumps({k:v for k,v in result.items() if k not in ['geometry','scenarios']}))
    return result


if __name__=='__main__':audit()
