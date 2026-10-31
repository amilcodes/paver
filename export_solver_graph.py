"""Update only audited changed donor decisions in the full experimental graph."""
import copy, json
from pathlib import Path
from benchmark import ROOT, digest, policy_mask
from run_dense import ref, read_ref, heavy_job
from run_controlled_benchmark import load_evidence
from dense_experiment import immutable_json


def export():
    folder = ROOT/'solver_budget_v1'
    plan = json.loads((folder/'execution_plan.json').read_text())
    audit = json.loads((folder/'audit_001.json').read_text())
    assert audit['passed'] is True and audit['plan_sha256'] == digest(folder/'execution_plan.json')
    assert audit['registry_sha256'] == digest(folder/'registry.json')
    registry = json.loads((folder/'registry.json').read_text())
    parent_ref = ref(ROOT/'network_v1/visual_evidence_sampling_960.json')
    base = json.loads(read_ref(parent_ref).read_text())
    parent_plan = json.loads(read_ref(plan['parent_plan']).read_text())
    scenes = {s['scene']: s for s in parent_plan['scenes']}
    def key(r): return r['scene'], r['job'], r['method'], r['target'], r['donor']
    rows = {key(r): copy.deepcopy(r) for r in base['decisions']}; changed = 0
    assert len(rows) == len(base['decisions']) == 968
    for link in registry['groups']:
        if not link['changed']: continue
        s = scenes[link['scene']]; r = json.loads(read_ref(link['record']).read_text()); method = 'loftr_metric_free'
        d = r['methods'][method]; e = load_evidence(d['evidence'])
        accepted = {name: policy_mask(e, e['mask'], p) for name, p in plan['policies'].items()}
        donors = [i for i in range(len(s['frames'])) if i != s['target']]
        for index, (i, reg) in enumerate(zip(donors, d['registrations'])):
            k = (s['scene'], link['key'], method, s['frames'][s['target']]['id'], s['frames'][i]['id'])
            assert k in rows
            rows[k].update(geometry_accepted=reg.get('accepted') is True, selected_pixels={name: int((take & (e['provenance'] == index)).sum()) for name, take in accepted.items()}, record=link['record'])
            changed += 1
    assert changed == 24
    result = dict(plan=ref(folder/'execution_plan.json'), audit=ref(folder/'audit_001.json'), source=ref(Path(__file__).resolve()), parent_graph=parent_ref,
                  retrieval_graph=base['retrieval_graph'], model=plan['model'], decisions=list(rows.values()), changed_decisions=changed,
                  outside_area_graph_controls=base['outside_area_graph_controls'], status='experimental_not_promoted', scope=base['scope'])
    immutable_json(ROOT/'network_v1/visual_evidence_solver_budget.json', result)
    print(json.dumps(dict(decisions=len(rows), changed_decisions=changed, distinct_target_donor_pairs=len({(r['target'], r['donor']) for r in rows.values()}))))


if __name__ == '__main__':
    with heavy_job(): export()
