"""Resumable three-area source-only baseline; clean hidden RGB enters scoring only."""
import argparse,gc,json,shutil,time
from datetime import datetime,timezone
import cv2
import numpy as np
from PIL import Image
from benchmark import ROOT,digest,policy_mask
from repair import nadir
from resource_guard import require_storage
from run_dense import heavy_job,verify_installed,verify_certificate,ref,read_ref,save_evidence
from dense_experiment import immutable_json,masked_inputs,register_sequence,score_transfer
from controlled_occlusions import damage,appearance_bands,baseline_stats

OUT=ROOT/'controlled_v1'


def checked_plan():
    plan=json.loads((OUT/'execution_plan.json').read_text())
    for p,h in plan['signature'].items():read_ref(dict(path=p,sha256=h))
    read_ref(plan['graph']);audit=json.loads(read_ref(plan['dataset_audit']).read_text())
    if audit.get('passed') is not True:raise RuntimeError('Area audit has not passed')
    return plan


def load_area(area):
    for r in area['inputs']:read_ref(r)
    images=[nadir(np.asarray(Image.open(read_ref(r)).convert('RGB')),**area['projection']) for r in area['frames']]
    with np.load(read_ref(area['masks']),allow_pickle=False) as z:bundle={k:z[k] for k in z.files}
    return images,[bundle[f'allowed_{i}'] for i in range(len(images))],bundle


def load_evidence(record):
    with np.load(read_ref(record),allow_pickle=False) as z:return {k:z[k] for k in z.files}


def prediction(evidence):
    mask=evidence['mask'];out=np.zeros((*mask.shape,3),np.uint8);out[mask]=evidence['prediction_on_mask'];return out


def run(args):
    plan=checked_plan();certificate=ROOT/'sequence_v6/model_validation_001.json'
    signature=dict(plan_sha256=digest(OUT/'execution_plan.json'),certificate=ref(certificate))
    with heavy_job():
        verify_installed();verify_certificate(certificate)
        from dense_matching import MaskedLoFTR
        matcher=None;done=0
        for area in plan['areas']:
            folder=OUT/area['area'];groups=folder/'groups';groups.mkdir(exist_ok=True);pairs=folder/'pairs';pairs.mkdir(exist_ok=True)
            for r in area['reuse_donor_pairs']:
                source=read_ref(r);dest=pairs/f"donor_only_{r['i']}_{r['j']}.npz"
                if not dest.exists():shutil.copyfile(source,dest)
                if digest(dest)!=r['sha256']:raise RuntimeError('Reused donor cache changed')
            images=None
            for spec in area['mask_specs']:
                key=spec['key'];dest=groups/(key+'.json')
                if dest.exists():
                    old=json.loads(dest.read_text())
                    if old['signature']!=signature:raise RuntimeError('Run signature changed')
                    for data in old['methods'].values():read_ref(data['evidence'])
                    continue
                if done>=args.limit:return
                require_storage(ROOT);start=time.monotonic()
                if images is None:images,allowed,bundle=load_area(area)
                mask=bundle[key];target=area['target'];clean,als=masked_inputs(images,allowed,mask,target)
                if matcher is None:matcher=MaskedLoFTR(args.weights,plan['model']['checkpoint_sha256'],root=ROOT)
                def match(i,j,a,b,ma,mb):return matcher.correspondences(a,b,ma,mb)
                regs,graphs,corr=register_sequence(clean,als,target,area['projection']['fov'],key,pairs,match,plan['model'])
                record=dict(area=area['area'],case=area['case'],mask_key=key,signature=signature,
                    created_at_utc=datetime.now(timezone.utc).isoformat(),correspondences=corr,graphs=graphs,methods={},
                    graph_candidates=area['graph_donor_candidates'],annotated_donors_used=area['annotated_donors_used'])
                for method,r in regs.items():
                    stats,evidence=score_transfer(clean,als,mask,target,r,plan['policies'],images[target])
                    ep=groups/(key+'_'+method+'.npz');save_evidence(ep,evidence)
                    pred=prediction(evidence)
                    appearance={name:appearance_bands(pred,images[target],policy_mask(evidence,mask,p),mask) for name,p in plan['policies'].items()}
                    record['methods'][method]=dict(policies=stats,appearance=appearance,registrations=[v.diagnostics for v in r.values()],evidence=ref(ep))
                    del evidence,pred;gc.collect()
                record['runtime_seconds']=time.monotonic()-start;immutable_json(dest,record);done+=1
                print('GEOMETRY',area['area'],key,{m:dict(coverage=v['policies']['candidate']['coverage'],mae=v['policies']['candidate']['error']['mae_255']) for m,v in record['methods'].items()},flush=True)
                del clean,als,regs;gc.collect()
            scenarios=folder/'scenarios';scenarios.mkdir(exist_ok=True)
            for spec in area['scenarios']:
                dest=scenarios/(spec['key']+'.json')
                if dest.exists():
                    if json.loads(dest.read_text())['signature']!=signature:raise RuntimeError('Scenario signature changed')
                    continue
                require_storage(ROOT)
                if images is None:images,allowed,bundle=load_area(area)
                mask=bundle[spec['mask_key']];truth=images[area['target']]
                damaged=damage(truth,mask,spec['family'],spec['severity']);sanitized=damaged.copy();sanitized[mask]=0
                filled=cv2.inpaint(sanitized,mask.astype(np.uint8),3,cv2.INPAINT_TELEA)
                base={}
                for name,pred in [('as_damaged',damaged),('telea_r3',filled)]:
                    ep=scenarios/(spec['key']+'_'+name+'.npz');save_evidence(ep,dict(prediction_on_mask=pred[mask]))
                    base[name]=dict(metrics=baseline_stats(pred,truth,mask),evidence=ref(ep))
                immutable_json(dest,dict(**spec,area=area['area'],signature=signature,baselines=base,
                    geometry=ref(groups/(spec['mask_key']+'.json')),
                    interpretation='Source-only reconstruction excludes all damaged pixels. Shared masks reuse identical geometry; as_damaged/Telea are controls, not accepted recovery.'))
                del damaged,sanitized,filled,pred;gc.collect()
            print('AREA_COMPLETE',area['area'],flush=True)
            del images;gc.collect()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--limit',type=int,default=36);args=p.parse_args()
    if not 1<=args.limit<=36:p.error('limit must be1..36')
    run(args)
