"""Read-only control audit and small oracle experiment; no learned inference."""
import json,sys,tempfile
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from benchmark import ROOT,digest,policy_mask
from dense_experiment import immutable_json,masked_inputs,register_sequence,score_transfer
from run_dense import build_plan,read_ref,verify_installed


def run():
    verify_installed();plan=build_plan();frozen=ROOT/'sequence_v6/execution_plan.json'
    if plan!=json.loads(frozen.read_text()):raise RuntimeError('Execution plan has changed')
    checks=[]
    for case in plan['cases']:
        for job in case['jobs']:
            with np.load(read_ref(job['mask']),allow_pickle=False) as z:mask=z[job['mask_key']]
            for method,refs in job['controls'].items():
                record=json.loads(read_ref(refs['record']).read_text())
                with np.load(read_ref(refs['evidence']),allow_pickle=False) as z:
                    ev={k:z[k] for k in z.files}
                assert np.array_equal(ev['mask'],mask)
                assert mask.sum()==record['mask_pixels']
                assert ('absolute_rgb_error_sum_on_mask' in ev)==(job['region']!='footprint')
                if job['region']!='footprint':assert ev['absolute_rgb_error_sum_on_mask'].shape==(int(mask.sum()),)
                counts={}
                for policy,params in plan['policies'].items():
                    counts[policy]=int(policy_mask(ev,mask,params).sum())
                    old=record['methods'][method]['policies'][policy]
                    assert counts[policy]==old['accepted_pixels']
                    assert abs(old['coverage']-counts[policy]/max(1,int(mask.sum())))<1e-12
                checks.append(dict(case=case['case'],region=job['region'],control=method,mask_pixels=int(mask.sum()),accepted_pixels=counts))
    # This is a controlled oracle experiment, explicitly separate from real roads.
    sys.path.insert(0,str(ROOT/'tests'))
    from test_dense_experiment import fixture,oracle,POLICIES
    images,allowed,hidden,offsets=fixture();clean,als=masked_inputs(images,allowed,hidden,1);results={}
    with tempfile.TemporaryDirectory(prefix='dense-oracle-') as temp:
        for name,bias in [('correct_translation',0),('coherent_wrong_translation',24)]:
            regs,graphs,_=register_sequence(clean,als,1,140,name,Path(temp)/name,oracle(offsets,bias=bias),{'synthetic_oracle':name})
            results[name]={}
            for method,r in regs.items():
                stats,ev=score_transfer(clean,als,hidden,1,r,POLICIES,images[1])
                results[name][method]=dict(graph=graphs[method],policies=stats,registrations=[v.diagnostics for v in r.values()])
                if bias==0:
                    wrong_truth=images[1].copy();wrong_truth[hidden]=255
                    changed,_=score_transfer(clean,als,hidden,1,r,POLICIES,wrong_truth)
                    results[name][method]['changed_hidden_truth_policies']=changed
    assert 'torch' not in sys.modules,'This audit must not load model runtime'
    report=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),passed=True,
                kind='runner_control_audit_and_synthetic_oracle_only',actual_model_executed=False,
                real_road_recovery_runs=0,execution_plan_sha256=digest(frozen),
                source_sha256={n:digest(ROOT/n) for n in ['audit_dense_runner.py','tests/test_dense_experiment.py',*plan['signature']]},
                frozen_controls=checks,control_masks_checked=len(checks),policy_count_checks=len(checks)*len(plan['policies']),
                synthetic_oracle=results,
                limitations=['Oracle correspondences are exact synthetic geometry, not learned matching performance.',
                             'Zero error and full coverage on the toy fixture are not estimates of real-road recovery.',
                             'No new road pixels scored; cached control scores only.',
                             'Actual MPS inference, coordinate calibration and fixed-road comparisons remain pending.'])
    immutable_json(ROOT/'sequence_v6/runner_validation.json',report)
    print(json.dumps(dict(passed=True,control_masks_checked=len(checks),policy_count_checks=report['policy_count_checks'],real_road_recovery_runs=0,model_loaded=False,
                         synthetic={name:{m:d['policies']['candidate'] for m,d in methods.items()} for name,methods in results.items()})))


if __name__=='__main__':run()
