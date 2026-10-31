"""Unscored real-footprint track. Reuse identical old runs; add Nantes only.

Outputs transfer support and abstention, never a correctness rate for real holes.
"""
import argparse,json
from pathlib import Path
import numpy as np
from benchmark import ROOT,digest
from dense_experiment import immutable_json,masked_inputs,register_sequence,score_transfer
from run_dense import heavy_job,ref,read_ref,save_evidence,verify_installed,verify_certificate
from run_controlled_benchmark import checked_plan,load_area,load_evidence
from validate_dense_run_v2 import verify_pairs,verify_evidence,verify_selected_donors

OUT=ROOT/'natural_track_v1'


def run(weights):
    base=checked_plan();controls={a:ref(ROOT/'sequence_v6/results'/case/'footprint.json') for a,case in [('gironde','gironde_0'),('bourges','bourges_turn_001')]}
    plan=dict(controlled_plan=ref(ROOT/'controlled_v1/execution_plan.json'),source=ref(Path(__file__)),reused_records=controls,
        task='Real-footprint donor support only; no target ground truth.',new_model_runs='Nantes only; Gironde/Bourges reused unchanged five-frame controls.')
    immutable_json(OUT/'execution_plan.json',plan);results=[]
    with heavy_job():
        verify_installed();cert=ROOT/'sequence_v6/model_validation_001.json';verify_certificate(cert)
        for area in base['areas']:
            images,allowed,bundle=load_area(area);target=area['target'];mask=bundle['footprint'];donors=[i for i in range(len(images)) if i!=target]
            if (mask&allowed[target]).any():raise RuntimeError('Natural footprint enters allowed target')
            reused=area['area'] in controls
            if reused:rp=read_ref(controls[area['area']]);record=json.loads(rp.read_text())
            else:
                rp=OUT/(area['area']+'.json')
                if rp.exists():record=json.loads(rp.read_text())
                else:
                    from dense_matching import MaskedLoFTR
                    matcher=MaskedLoFTR(weights,base['model']['checkpoint_sha256'],root=ROOT)
                    clean,als=masked_inputs(images,allowed,mask,target)
                    def match(i,j,a,b,ma,mb):return matcher.correspondences(a,b,ma,mb)
                    regs,graphs,corr=register_sequence(clean,als,target,area['projection']['fov'],'natural_footprint',ROOT/'controlled_v1'/area['area']/'pairs',match,base['model'])
                    record=dict(signature=dict(plan_sha256=digest(OUT/'execution_plan.json'),certificate=ref(cert)),area=area['area'],methods={},graphs=graphs,correspondences=corr)
                    for method,r in regs.items():
                        stats,evidence=score_transfer(clean,als,mask,target,r,base['policies'],truth=None)
                        ep=OUT/(area['area']+'_'+method+'.npz');save_evidence(ep,evidence)
                        record['methods'][method]=dict(policies=stats,evidence=ref(ep),registrations=[v.diagnostics for v in r.values()])
                    immutable_json(rp,record);del matcher
            count=verify_pairs(ROOT,record,images,allowed,mask,target,base['model']);methods={}
            for method,data in record['methods'].items():
                e=load_evidence(data['evidence']);regs=data['registrations'];g=record['graphs'][method]
                checked=verify_evidence(e,data['policies'],mask,base['policies'],len(donors),footprint=True,truth=None,graph=g,registrations=regs)
                checked['provenance']=verify_selected_donors(e,[images[i] for i in donors],[allowed[i] for i in donors],regs)
                methods[method]=checked
            results.append(dict(area=area['area'],reused_previous_run=reused,record=ref(rp),mask_pixels=int(mask.sum()),pair_caches_verified=count,methods=methods))
    summary=dict(plan_sha256=digest(OUT/'execution_plan.json'),passed=True,truth_available=False,results=results,
        warning='Coverage is geometrically accepted source support, not measured correctness. Single-view or agreeing donors can be wrong. Read controlled_v1 for measured withheld-visible-ground error.')
    immutable_json(OUT/'summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True);args=p.parse_args();run(args.weights)
