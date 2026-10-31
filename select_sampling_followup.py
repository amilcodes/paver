"""Freeze follow-up eligibility from solver diagnostics alone; no scoring RGB."""
import json
from pathlib import Path
from benchmark import ROOT,digest
from run_dense import ref
from dense_experiment import immutable_json


def select():
    folder=ROOT/'sampling_v1';plan=json.loads((folder/'execution_plan.json').read_text());audit=json.loads((folder/'audit_001.json').read_text())
    if audit.get('passed') is not True or audit['plan_sha256']!=digest(folder/'execution_plan.json'):raise RuntimeError('Complete audit required')
    hashes={(r['scene'],r['job']):r['record_sha256'] for r in audit['groups']};selected=[];reused=[]
    for scene in plan['scenes']:
        for job in scene['jobs']:
            path=folder/scene['scene']/'groups'/(job['key']+'.json')
            if digest(path)!=hashes[(scene['scene'],job['key'])]:raise RuntimeError('Audited source changed')
            record=json.loads(path.read_text());g=record['graphs']['loftr_metric_free']
            eligible=(g.get('connected_frames',0)>1 and g.get('solver_converged') is False and g.get('nfev')==160 and g.get('boundary_hits')==[])
            row=dict(scene=scene['scene'],job=job['key'],record=ref(path),solver_converged=g.get('solver_converged'),nfev=g.get('nfev'),boundary_hits=g.get('boundary_hits'))
            (selected if eligible else reused).append(row)
    out=dict(status='proposed_frozen_not_executed',source=ref(Path(__file__).resolve()),baseline_plan=ref(folder/'execution_plan.json'),audit=ref(folder/'audit_001.json'),
        selection_rule='Metric-free fit has connected_frames>1, solver_converged=false, nfev=160 and no boundary hits. No coverage/error/scoring values used.',
        intervention=dict(old_max_nfev=160,new_max_nfev=640,initialization='identical',bounds='identical',tolerances='identical',convergence_required=True,acceptance_and_support_gates='identical'),
        selected=selected,reuse_unchanged=reused,learned_inference_needed=False,
        evaluation='Reuse and hash-verify all unaffected results, including full urban failures and physical-bound failures. Compare affected regions with both640px and960px baselines, then reconstruct full216-condition and original hard-road comparisons. Require source/metric audit before any claim. No promotion from coverage alone.')
    immutable_json(folder/'solver_budget_followup_selection.json',out)
    print(json.dumps(dict(selected=len(selected),reused=len(reused),road_fits_executed=0)))


if __name__=='__main__':select()
