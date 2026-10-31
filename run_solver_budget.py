"""Frozen six-fit solver-budget ablation; observations and all gates stay fixed.

An overlay registry retains the complete 126-group/216-condition population.
Only six metric fits are run. No matcher is instantiated or called.
"""
import argparse, gc, json, platform, time
from datetime import datetime, timezone
from importlib.metadata import version
import cv2
import numpy as np
from benchmark import ROOT, digest, policy_mask
from run_dense import ref, read_ref, save_evidence, heavy_job
from dense_experiment import immutable_json, array_hash, masked_inputs, score_transfer
from sequence_geometry import propose_edge
from metric_plane import fit_metric_graph, gradient_error
from constrained_sequence import audit_poses
from run_sampling import inputs, texture_for_scoring, prediction
from run_controlled_benchmark import load_evidence
from prepare_area_benchmark import solid_angles
from area_recovery_metrics import extended_metrics, same_pixels
from sampling_metrics import partitions
from resource_guard import require_storage

OUT = ROOT / 'solver_budget_v1'
METHOD = 'loftr_metric_free'
OTHER = 'loftr_homography'


def environment():
    return dict(python=platform.python_version(), packages={p: version(p) for p in ['numpy', 'scipy']}, opencv=cv2.__version__)


def cached_edges(record):
    """Replay deterministic edge selection from immutable matched coordinates."""
    edges, certificate = [], []
    for pair in record['correspondences']:
        z = load_evidence(dict(path=pair['path'], sha256=pair['sha256']))
        edge, proposal = propose_edge(pair['i'], pair['j'], z['p'], z['q'])
        if proposal != pair['proposal']:
            raise RuntimeError('Cached edge selection differs from original run')
        row = dict(i=pair['i'], j=pair['j'], pair_sha256=pair['sha256'], accepted=edge is not None)
        if edge is not None:
            edges.append(edge)
            row['arrays'] = {k: array_hash(edge[k]) for k in ['p', 'q', 'vp', 'vq', 'H']}
        certificate.append(row)
    return edges, certificate


def build_plan():
    selection_ref = ref(ROOT / 'sampling_v1/solver_budget_followup_selection.json')
    selection = json.loads(read_ref(selection_ref).read_text())
    parent = json.loads(read_ref(selection['baseline_plan']).read_text())
    audit = json.loads(read_ref(selection['audit']).read_text())
    if audit.get('passed') is not True:
        raise RuntimeError('Parent audit required')
    selected = {(r['scene'], r['job']): r for r in selection['selected']}
    audited = {(r['scene'], r['job']): r['record_sha256'] for r in audit['groups']}
    ca = {(r['scene'], r['condition']): r['record_sha256'] for r in audit['conditions']}
    groups, conditions = [], []
    for scene in parent['scenes']:
        for job in scene['jobs']:
            key = (scene['scene'], job['key'])
            r = ref(ROOT / 'sampling_v1' / key[0] / 'groups' / (key[1] + '.json'))
            if r['sha256'] != audited[key]:
                raise RuntimeError('Parent group changed since audit')
            row = dict(scene=key[0], job=key[1], parent=r, changed=key in selected)
            if row['changed']:
                if selected[key]['record'] != r:
                    raise RuntimeError('Frozen selection differs')
                _, row['edge_certificate'] = cached_edges(json.loads(read_ref(r).read_text()))
            groups.append(row)
        for c in scene['conditions']:
            r = ref(ROOT / 'sampling_v1' / scene['scene'] / 'conditions' / (c['spec']['key'] + '.json'))
            if r['sha256'] != ca[(scene['scene'], c['spec']['key'])]:
                raise RuntimeError('Parent condition changed since audit')
            conditions.append(dict(scene=scene['scene'], condition=c['spec']['key'], job=c['job'], parent=r, changed=(scene['scene'], c['job']) in selected))
    assert len(groups) == 126 and sum(r['changed'] for r in groups) == 6
    assert len(conditions) == 216 and sum(r['changed'] for r in conditions) == 8
    sources = dict(parent['signature'])
    for f in ['run_solver_budget.py', 'audit_solver_budget.py', 'summarize_solver_budget.py', 'test_solver_budget.py', 'summarize_expanded_baseline.py', 'summarize_sampling.py']:
        sources[f] = digest(ROOT / f)
    plan = dict(selection=selection_ref, parent_plan=selection['baseline_plan'], parent_audit=selection['audit'],
                intervention=selection['intervention'], model=parent['model'], certificate=parent['certificate'],
                environment=environment(), signature=sources, policies=parent['policies'], groups=groups, conditions=conditions,
                created_at_utc=datetime.now(timezone.utc).isoformat(), learned_inference=False, independent_test=False,
                evaluation='All126groups/216conditions and retained hard roads. Compare640 and960, RGB/gradient/angular/texture tails, abstention and same-pixel controls. No new truth or confidence tuning.',
                promotion_rule='No automatic promotion. Reject a default if retained hard-road coverage/error regresses against640, new textured-error tails worsen, urban failures persist, or gains are merely unsupported coverage. No relaxed acceptance gates.')
    immutable_json(OUT / 'execution_plan.json', plan)
    return plan


def check_plan():
    plan = json.loads((OUT / 'execution_plan.json').read_text())
    for f, h in plan['signature'].items():
        read_ref(dict(path=f, sha256=h))
    for k in ['selection', 'parent_plan', 'parent_audit', 'certificate']:
        read_ref(plan[k])
    if plan['environment'] != environment():
        raise RuntimeError('Numerical environment changed')
    return plan


def gradient_partitions(new, old, truth, policies):
    a, b = prediction(new), prediction(old)
    result = {}
    for name, p in policies.items():
        x, y = policy_mask(new, new['mask'], p), policy_mask(old, old['mask'], p)
        result[name] = {}
        for label, take in [('common', x & y), ('gained', x & ~y), ('lost', y & ~x)]:
            row = {}
            if label != 'lost': row['new'] = gradient_error(a, truth, take)
            if label != 'gained': row['control'] = gradient_error(b, truth, take)
            result[name][label] = row
    return result


def registry(plan):
    result = dict(plan=ref(OUT / 'execution_plan.json'), groups=[], conditions=[])
    for kind in ['groups', 'conditions']:
        for row in plan[kind]:
            key = row['job'] if kind == 'groups' else row['condition']
            path = OUT / row['scene'] / kind / (key + '.json')
            r = ref(path) if row['changed'] else row['parent']
            read_ref(r)
            result[kind].append(dict(scene=row['scene'], key=key, changed=row['changed'], record=r, parent=row['parent']))
    immutable_json(OUT / 'registry.json', result)


def run(limit):
    plan = check_plan()
    parent_plan = json.loads(read_ref(plan['parent_plan']).read_text())
    sig = dict(plan_sha256=digest(OUT / 'execution_plan.json'), certificate=plan['certificate'])
    rows = {(r['scene'], r['job']): r for r in plan['groups']}
    changed_conditions = {(r['scene'], r['condition']) for r in plan['conditions'] if r['changed']}
    completed = 0
    with heavy_job():
        for scene in parent_plan['scenes']:
            selected = [j for j in scene['jobs'] if rows[(scene['scene'], j['key'])]['changed']]
            if not selected: continue
            images = None
            for job in selected:
                row = rows[(scene['scene'], job['key'])]
                dest = OUT / scene['scene'] / 'groups' / (job['key'] + '.json')
                if dest.exists():
                    previous = json.loads(dest.read_text())
                    if previous['signature'] != sig: raise RuntimeError('Resume signature differs')
                    for m in previous['methods'].values(): read_ref(m['evidence'])
                    continue
                if completed >= limit: return
                require_storage(ROOT)
                started = time.monotonic()
                if images is None: images, allowed = inputs(scene)
                mask = load_evidence(job['mask'])[job['mask_key']]
                target = scene['target']
                if job['kind'] == 'natural' or (mask & ~allowed[target]).any():
                    raise RuntimeError('Unexpected selection or hidden visibility')
                clean, permitted = masked_inputs(images, allowed, mask, target)
                parent = json.loads(read_ref(row['parent']).read_text())
                edges, certificate = cached_edges(parent)
                if certificate != row['edge_certificate']: raise RuntimeError('Edge partition changed')
                poses, graph, active = fit_metric_graph(len(clean), target, edges, len(mask), scene['projection']['fov'], max_nfev=640)
                graph['poses'] = {str(i): h.tolist() for i, h in poses.items()}
                regs = audit_poses(clean, permitted, poses, graph, active, target)
                # All geometry and source acceptance is complete before truth/texture is consulted.
                truth = images[target]
                stats, evidence = score_transfer(clean, permitted, mask, target, regs, plan['policies'], truth)
                ep = dest.with_name(job['key'] + '_' + METHOD + '.npz')
                save_evidence(ep, evidence)
                weights = solid_angles(**scene['projection'])
                texture = texture_for_scoring(scene, truth)
                pred = prediction(evidence)
                extra = {n: extended_metrics(pred, truth, policy_mask(evidence, mask, p), mask, weights, texture) for n, p in plan['policies'].items()}
                old640 = json.loads(read_ref(job['control']).read_text())['methods'][METHOD]
                data = dict(policies=stats, extended=extra, evidence=ref(ep), registrations=[r.diagnostics for r in regs.values()])
                for label, old in [('640', old640), ('960', parent['methods'][METHOD])]:
                    e = load_evidence(old['evidence'])
                    comparison = dict(policies=old['policies'], extended=parent['methods'][METHOD]['control_extended'] if label == '640' else old['extended'],
                                      partitions=partitions(evidence, e, plan['policies']), gradient_partitions=gradient_partitions(evidence, e, truth, plan['policies']))
                    if label == '640':
                        data.update(control_policies=comparison['policies'], control_extended=comparison['extended'], partitions=comparison['partitions'], partition_gradients=comparison['gradient_partitions'])
                    else: data['comparison_to_960'] = comparison
                result = dict(scene=scene['scene'], area=scene['area'], job=job, signature=sig, parent_960=row['parent'],
                              edge_certificate=certificate, correspondences=parent['correspondences'], graphs={OTHER: parent['graphs'][OTHER], METHOD: graph},
                              methods={OTHER: parent['methods'][OTHER], METHOD: data}, reused_methods=[OTHER], learned_model_executed=False,
                              runtime_seconds=time.monotonic()-started, created_at_utc=datetime.now(timezone.utc).isoformat())
                immutable_json(dest, result)
                completed += 1
                print('GROUP', scene['scene'], job['key'], dict(converged=graph['optimizer_success'], nfev=graph['nfev'], coverage=stats['candidate']['coverage'], mae=stats['candidate']['error']['mae_255']), flush=True)
                del clean, permitted, evidence, pred, e, regs, truth, weights, texture
                gc.collect()
            for c in scene['conditions']:
                if (scene['scene'], c['spec']['key']) not in changed_conditions: continue
                dest = OUT / scene['scene'] / 'conditions' / (c['spec']['key'] + '.json')
                if dest.exists():
                    if json.loads(dest.read_text())['signature'] != sig: raise RuntimeError('Condition signature differs')
                    continue
                if images is None: images, allowed = inputs(scene)
                gp = OUT / scene['scene'] / 'groups' / (c['job'] + '.json')
                g = json.loads(gp.read_text())
                baseline = json.loads(read_ref(c['baseline']).read_text())
                prior = json.loads((ROOT / 'sampling_v1' / scene['scene'] / 'conditions' / (c['spec']['key'] + '.json')).read_text())
                comparisons = {OTHER: prior['same_pixel_controls'][OTHER], METHOD: {}}
                e = load_evidence(g['methods'][METHOD]['evidence'])
                for name, p in plan['policies'].items():
                    take = policy_mask(e, e['mask'], p)[e['mask']]
                    comparisons[METHOD][name] = {b: same_pixels(e['prediction_on_mask'], load_evidence(v['evidence'])['prediction_on_mask'], images[scene['target']][e['mask']], take) for b, v in baseline['baselines'].items()}
                immutable_json(dest, dict(scene=scene['scene'], condition=c, signature=sig, geometry=ref(gp), same_pixel_controls=comparisons))
            if images is not None: del images, allowed
            gc.collect()
        registry(plan)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--limit', type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.limit <= 6: parser.error('limit must be1..6')
    if args.prepare:
        p = build_plan()
        print(json.dumps(dict(groups=len(p['groups']), changed=6, conditions=len(p['conditions']), changed_conditions=8)))
    else: run(args.limit)
