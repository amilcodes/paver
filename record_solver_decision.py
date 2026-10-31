"""Record the bounded experiment's conclusion from audited full-suite results."""
import json
from datetime import datetime, timezone
from pathlib import Path
from benchmark import ROOT, digest
from run_dense import ref
from dense_experiment import immutable_json


def record():
    out = ROOT/'solver_budget_v1'; summary = json.loads((out/'summary.json').read_text()); audit = json.loads((out/'audit_001.json').read_text())
    assert audit['passed'] is True and audit['plan_sha256'] == digest(out/'execution_plan.json')
    rural = [r for r in summary['aggregates'] if r['area']=='gironde' and r['severity']=='both' and r['method']=='loftr_metric_free' and r['family'] in ['opaque_disc', 'opaque_vehicle', 'missing_strip'] and r['policy'] in ['candidate', 'corroborated']]
    rear = next(r for r in summary['hard_controls'] if r['scene']=='gironde_hard' and r['region']=='hidden_rear')['methods']['loftr_metric_free']
    vehicle = next(r for r in rural if r['family']=='opaque_vehicle' and r['policy']=='candidate')
    strip = next(r for r in rural if r['family']=='missing_strip' and r['policy']=='candidate')
    urban = [r for r in summary['aggregates'] if r['area'] in ['nantes', 'bourges']]
    assert all(r['treatment']['coverage']==0 for r in urban)
    assert rear['partitions640']['candidate']['common']['new']['mae_255'] > rear['partitions640']['candidate']['common']['control']['mae_255']
    assert vehicle['treatment']['strata']['textured']['accepted_over_20_fraction'] > vehicle['baseline960']['strata']['textured']['accepted_over_20_fraction']
    assert strip['treatment']['coverage'] > strip['baseline960']['coverage'] and strip['treatment']['rgb_mae_255'] < strip['baseline960']['rgb_mae_255']
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), source=ref(Path(__file__).resolve()), plan=ref(out/'execution_plan.json'),
                  summary=ref(out/'summary.json'), audit=ref(out/'audit_001.json'), graph=ref(ROOT/'network_v1/visual_evidence_solver_budget.json'),
                  status='completed_audited_rejected_as_default', promoted=False, implemented='Only metric solver max_nfev160->640 for six previously frozen interior failures. No new learned inference or changed correspondences, initialization, bounds, tolerances, support or acceptance gates.',
                  counts=dict(groups=126, method_evaluations=252, conditions=216, new_fits=6, converged_new_fits=5, remaining_budget_failures=1, reused_groups=120, reused_conditions=208, unit_checks=3, hash_verified_files=audit['files_hash_verified']),
                  result='Longer budget resolves five termination failures, but convergence alone does not establish reliable transfer. Strip coverage/error improve; vehicle textured-error tails and original rear same-pixel error prevent default adoption.',
                  rural_family_results=rural, original_rear=rear,
                  reasons=[
                      'Gironde original hidden_rear recovers51.12% after the960 baseline abstained; original640 covered46.55% atMAE4.20, while treatmentMAE5.78. Oncommonpixels error4.12->5.90, so additional support is not a defensible default improvement.',
                      'Pooled Gironde vehicle coverage55.13->68.97% versus960, butMAE4.97->5.14 and accepted textured>20 tail7.44->10.01%. No predeclared error tradeoff permits promotion.',
                      'Pooled strip coverage73.77->76.29%,MAE5.01->4.98, overall>20tail0.498->0.483%,texturedtail3.20->2.99%; retain as a measured positive observation, not a post-hoc family-specific policy.',
                      'One disc fit remains unconverged at640 and fully abstains. Bourges physical-bound case and all disconnected failures are reused unchanged.',
                      'Nantes/Bourges/Alsace stay atzero; Gironde local_rear still14.62% versus640 control99.03%. More solver iterations cannot create absent observations.',
                      'Actual ego-vehicle masks were not treated in this ablation and have no clean target truth. Shared masks and reused routes are correlated development evidence.'
                  ], next_action='Stop budget sweeps. Predeclare a versioned native/perspective ground-crop matcher with pinned MPS weights, correct coordinate transforms, deduplication and hidden-RGB invariance. Compare full original640,960 and budget-overlay controls; focus on missing urban observations and local-rear connectivity. Do not relax geometry gates.',
                  infrastructure_record=ref(out/'infrastructure_failure_001.json'), resource_scope='One heavy job; cached observations only; all prior signed sources/results preserved.')
    immutable_json(out/'recovery_decision.json', result)
    print(json.dumps(dict(promoted=False, converged=5, experiments_complete=True)))


if __name__=='__main__': record()
