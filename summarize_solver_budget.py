"""Full-population, paired 640/960/solver overlay comparison after audit."""
import json
from collections import defaultdict
from benchmark import ROOT, digest
from run_dense import ref, read_ref
from run_solver_budget import OUT, METHOD
from dense_experiment import immutable_json
from summarize_expanded_baseline import aggregate
from summarize_sampling import partition_aggregate


def identity_partitions(d, policy, natural=False):
    n = d['policies'][policy]['accepted_pixels']; total = d['extended'][policy]['mask_pixels']
    result = {k: dict(pixels=n if k == 'common' else 0, fraction_of_mask=n/total if k == 'common' else 0.) for k in ['common', 'gained', 'lost']}
    if not natural:
        value = {k: d['policies'][policy]['error'][k] for k in ['pixels', 'mae_255', 'p90_rgb', 'over_20_fraction']}
        empty = dict(pixels=0, mae_255=None, p90_rgb=None, over_20_fraction=None)
        result['common'].update(new=value, control=value)
        result['gained']['new'] = empty; result['lost']['control'] = empty
    return result


def summarize():
    p = json.loads((OUT/'execution_plan.json').read_text()); audit = json.loads((OUT/'audit_001.json').read_text())
    if audit.get('passed') is not True or audit['plan_sha256'] != digest(OUT/'execution_plan.json') or audit['registry_sha256'] != digest(OUT/'registry.json'):
        raise RuntimeError('Complete matching audit required')
    for f, h in p['signature'].items(): read_ref(dict(path=f, sha256=h))
    parent = json.loads(read_ref(p['parent_plan']).read_text()); registry = json.loads((OUT/'registry.json').read_text())
    groups = {(r['scene'], r['key']): r for r in registry['groups']}; conditions = {(r['scene'], r['key']): r for r in registry['conditions']}
    rows = defaultdict(list); hard = []; natural = []; changed = []
    for scene in parent['scenes']:
        records = {}
        for job in scene['jobs']:
            link = groups[(scene['scene'], job['key'])]; g = json.loads(read_ref(link['record']).read_text()); old = json.loads(read_ref(link['parent']).read_text()); records[job['key']] = (g, old)
            if link['changed']:
                changed.append(dict(scene=scene['scene'], job=job['key'], record=link['record'], old_solver=old['graphs'][METHOD], new_solver=g['graphs'][METHOD],
                                    policies=g['methods'][METHOD]['policies'], extended=g['methods'][METHOD]['extended'], comparison_to_960=g['methods'][METHOD]['comparison_to_960'],
                                    partitions_to_640=g['methods'][METHOD]['partitions'], gradient_partitions_to_640=g['methods'][METHOD]['partition_gradients']))
            if job['kind'] in ['natural', 'hard_control']:
                entry = dict(scene=scene['scene'], area=scene['area'], region=job['key'], record=link['record'], changed=link['changed'], methods={})
                for m, d in g['methods'].items():
                    before = old['methods'][m]
                    entry['methods'][m] = dict(treatment=dict(policies=d['policies'], extended=d['extended']), baseline640=dict(policies=d['control_policies'], extended=d['control_extended']),
                                              baseline960=dict(policies=before['policies'], extended=before['extended']), partitions640=d['partitions'],
                                              partitions960=d['comparison_to_960']['partitions'] if 'comparison_to_960' in d else {name: identity_partitions(d, name, job['kind']=='natural') for name in p['policies']})
                (natural if job['kind'] == 'natural' else hard).append(entry)
        for c in scene['conditions']:
            link = conditions[(scene['scene'], c['spec']['key'])]; record = json.loads(read_ref(link['record']).read_text()); old_condition = json.loads(read_ref(link['parent']).read_text()); g, old = records[c['job']]
            for m, d in g['methods'].items():
                before = old['methods'][m]
                for policy, stats in d['policies'].items():
                    row = dict(scene=scene['scene'], metrics=stats, extended=d['extended'][policy], same_pixel_controls=record['same_pixel_controls'][m][policy], partition=d['partitions'][policy],
                               partition960=d['comparison_to_960']['partitions'][policy] if 'comparison_to_960' in d else identity_partitions(d, policy),
                               baseline640=dict(scene=scene['scene'], metrics=d['control_policies'][policy], extended=d['control_extended'][policy]),
                               baseline960=dict(scene=scene['scene'], metrics=before['policies'][policy], extended=before['extended'][policy], same_pixel_controls=old_condition['same_pixel_controls'][m][policy]))
                    for severity in [c['spec']['severity'], 'both']:
                        rows[(scene['area'], c['spec']['family'], severity, m, policy)].append(row)
    aggregates = [dict(area=a, family=f, severity=s, method=m, policy=po, treatment=aggregate(v), baseline640=aggregate([r['baseline640'] for r in v]), baseline960=aggregate([r['baseline960'] for r in v]),
                       partitions640=partition_aggregate(v), partitions960=partition_aggregate([dict(r, partition=r['partition960']) for r in v])) for (a, f, s, m, po), v in sorted(rows.items())]
    result = dict(status='audited_budget_ablation_no_automatic_promotion', plan=ref(OUT/'execution_plan.json'), audit=ref(OUT/'audit_001.json'), registry=ref(OUT/'registry.json'),
                  groups=126, conditions=216, method_evaluations=252, new_solver_fits=6, new_learned_inference=0, aggregates=aggregates, changed=changed, hard_controls=hard, natural=natural,
                  limitations=['Nine area targets and Alsace are reused development/train; not independent tests.', 'Shared-mask corruption conditions are correlated.',
                               'Only six interior budget-limited metric fits change; all other methods/groups/failures are exact inherited outputs.', 'Natural car footprints have no clean target truth; unchanged support is not measured correctness.',
                               'Appearance scores do not calibrate geometric accuracy. Small controlled masks differ from full car footprints.'])
    immutable_json(OUT/'summary.json', result)
    print(json.dumps(dict(groups=126, conditions=216, new_fits=6, converged=sum(r['new_solver']['optimizer_success'] for r in changed))))


if __name__ == '__main__': summarize()
