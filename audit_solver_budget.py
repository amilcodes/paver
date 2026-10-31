"""Audit changed fits, exact cached partitions and every immutable reuse link."""
import gc, json
from datetime import datetime, timezone
import cv2
import numpy as np
from run_dense import heavy_job
from run_solver_budget import OUT, METHOD, OTHER, cached_edges
from run_sampling import inputs
from dense_experiment import immutable_json, masked_inputs, array_hash, score_transfer
from constrained_sequence import audit_poses
from audit_area_annotations import quadrature
from audit_expanded_baseline import expected_extended, verify_extended, expected_same
from audit_sampling import expected_partitions
from validate_dense_run_v2 import (ROOT, checked_ref, sha, require, compare, accepted,
    load_npz, verify_pairs, verify_evidence, verify_selected_donors, gradient_metrics)


def compare_structure(actual, expected, label):
    if isinstance(expected, (str, bool)):
        require(actual == expected and type(actual) is type(expected), label+' differs')
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label+' length differs')
        for i, (a, b) in enumerate(zip(actual, expected)): compare_structure(a, b, label+str(i))
    elif isinstance(expected, dict):
        require(set(actual) == set(expected), label+' fields differ')
        for k, v in expected.items(): compare_structure(actual[k], v, label+'.'+k)
    else: compare(actual, expected, label)


def audit():
    pp = OUT / 'execution_plan.json'
    p = json.loads(pp.read_text())
    for f, h in p['signature'].items(): checked_ref(ROOT, dict(path=f, sha256=h))
    parent = json.loads(checked_ref(ROOT, p['parent_plan']).read_text())
    pa = json.loads(checked_ref(ROOT, p['parent_audit']).read_text())
    selection = json.loads(checked_ref(ROOT, p['selection']).read_text())
    checked_ref(ROOT, p['certificate'])
    require(pa['passed'] is True and pa['plan_sha256'] == p['parent_plan']['sha256'], 'parent audit')
    require(p['model'] == parent['model'] and p['policies'] == parent['policies'], 'model/gates changed')
    require(p['intervention'] == selection['intervention'] and p['learned_inference'] is False, 'intervention changed')
    registry = json.loads((OUT / 'registry.json').read_text())
    require(registry['plan']['sha256'] == sha(pp), 'registry signature')
    require(len(registry['groups']) == 126 and len(registry['conditions']) == 216, 'population')
    selected = {(r['scene'], r['job']) for r in selection['selected']}
    require(len(selected) == 6, 'selection count')
    pa_groups = {(r['scene'], r['job']): r['record_sha256'] for r in pa['groups']}
    pa_conditions = {(r['scene'], r['condition']): r['record_sha256'] for r in pa['conditions']}
    audited, seen_files = {}, set()
    def verify_ref(r):
        tag = (r['path'], r['sha256'])
        if tag not in seen_files:
            checked_ref(ROOT, r); seen_files.add(tag)
    for kind, old_index in [('groups', pa_groups), ('conditions', pa_conditions)]:
        rows = registry[kind]
        require(len({(r['scene'], r['key']) for r in rows}) == len(rows), 'duplicate registry row')
        require({(r['scene'], r['key']) for r in rows} == set(old_index), 'missing frozen query')
        for r in rows:
            require(r['parent']['sha256'] == old_index[(r['scene'], r['key'])], 'parent no longer audited')
            old = json.loads(checked_ref(ROOT, r['parent']).read_text())
            verify_ref(r['record'])
            if kind == 'groups':
                require(r['changed'] == ((r['scene'], r['key']) in selected), 'unselected fit changed')
                for m in old['methods'].values(): verify_ref(m['evidence'])
                for pair in old['correspondences']: verify_ref(dict(path=pair['path'], sha256=pair['sha256']))
            else:
                require(r['changed'] == ((r['scene'], old['condition']['job']) in selected), 'condition selection')
            if not r['changed']: require(r['record'] == r['parent'], 'unchanged output not exactly reused')
    groups = {(r['scene'], r['key']): r for r in registry['groups']}
    conditions = {(r['scene'], r['key']): r for r in registry['conditions']}
    sig = dict(plan_sha256=sha(pp), certificate=p['certificate'])
    for scene in parent['scenes']:
        jobs = [j for j in scene['jobs'] if (scene['scene'], j['key']) in selected]
        if not jobs: continue
        images, allowed = inputs(scene); target = scene['target']; truth = images[target]
        donors = [i for i in range(len(images)) if i != target]
        weights = quadrature(**scene['projection'])
        if scene['loader'] == 'expanded': texture = load_npz(checked_ref(ROOT, scene['source']['scoring_only_strata']))['textured']
        else:
            gray = cv2.cvtColor(truth, cv2.COLOR_RGB2GRAY).astype(np.float32)
            x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=1/8)
            y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=1/8)
            texture = x*x+y*y >= 16
        for job in jobs:
            row = groups[(scene['scene'], job['key'])]
            r = json.loads(checked_ref(ROOT, row['record']).read_text())
            old = json.loads(checked_ref(ROOT, row['parent']).read_text())
            control = json.loads(checked_ref(ROOT, job['control']).read_text())
            require(r['signature'] == sig and r['job'] == job and r['parent_960'] == row['parent'], 'result identity')
            require(r['learned_model_executed'] is False and r['reused_methods'] == [OTHER], 'method scope')
            require(r['correspondences'] == old['correspondences'], 'observations changed')
            require(r['graphs'][OTHER] == old['graphs'][OTHER] and r['methods'][OTHER] == old['methods'][OTHER], 'homography changed')
            mask = load_npz(checked_ref(ROOT, job['mask']))[job['mask_key']]
            require(job['kind'] != 'natural' and not (mask & ~allowed[target]).any(), 'hidden visibility')
            verify_pairs(ROOT, r, images, allowed, mask, target, p['model'])
            edges, cert = cached_edges(old)
            require(cert == r['edge_certificate'], 'edge hashes')
            frozen = next(x for x in p['groups'] if x['scene'] == scene['scene'] and x['job'] == job['key'])
            require(cert == frozen['edge_certificate'], 'unfrozen partition')
            # Check the holdout coordinates against the original raw-cache permutation.
            for e in edges:
                pair = next(x for x in r['correspondences'] if x['i'] == e['i'] and x['j'] == e['j'])
                z = load_npz(checked_ref(ROOT, dict(path=pair['path'], sha256=pair['sha256'])))
                order = np.random.default_rng(29+17*e['i']+31*e['j']).permutation(len(z['p']))
                cut = int(.75*len(order)); tr, va = order[:cut], order[cut:]
                require(np.array_equal(e['vp'], z['p'][va]) and np.array_equal(e['vq'], z['q'][va]), 'heldout points changed')
                training = {tuple(np.r_[a, b]) for a, b in zip(z['p'][tr], z['q'][tr])}
                require(all(tuple(np.r_[a, b]) in training for a, b in zip(e['p'], e['q'])), 'heldout leaked into fit')
            graph = r['graphs'][METHOD]; prior_graph = old['graphs'][METHOD]
            require(prior_graph['connected_frames'] > 1 and prior_graph['solver_converged'] is False and prior_graph['nfev'] == 160 and prior_graph['boundary_hits'] == [], 'selection not interior budget failure')
            require(graph['optimizer_success'] == (graph['solver_converged'] and not graph['boundary_hits']), 'convergence gate changed')
            require(160 < graph['nfev'] <= 640, 'unexpected numerical path/budget')
            for key in ['initial_plane_tilt', 'initial_median_fit_px', 'connected_frames', 'edges', 'parameters']:
                compare_structure(graph[key], prior_graph[key], 'initialization.'+key)
            clean, permitted = masked_inputs(images, allowed, mask, target)
            poses = {int(i): np.array(h) for i, h in graph['poses'].items()}
            active = [e for e in edges if e['i'] in poses and e['j'] in poses]
            regs = audit_poses(clean, permitted, poses, graph, active, target)
            d = r['methods'][METHOD]
            compare_structure(d['registrations'], [v.diagnostics for v in regs.values()], 'frozen support/audit gates')
            e = load_npz(checked_ref(ROOT, d['evidence']))
            checks = verify_evidence(e, d['policies'], mask, p['policies'], 4, footprint=False, truth=truth, graph=graph, registrations=d['registrations'])
            checks['provenance'] = verify_selected_donors(e, [images[i] for i in donors], [allowed[i] for i in donors], d['registrations'])
            # Replay transfer with audited support hulls, never a new fit or match.
            _, replay = score_transfer(clean, permitted, mask, target, regs, p['policies'], truth)
            require(all(array_hash(e[k]) == array_hash(replay[k]) for k in e), 'transfer/support replay differs')
            pred = np.zeros_like(truth); pred[mask] = e['prediction_on_mask']
            for name, policy in p['policies'].items():
                verify_extended(d['extended'][name], expected_extended(pred, truth, accepted(e, mask, policy), mask, weights, texture))
            for label, original in [('640', control['methods'][METHOD]), ('960', old['methods'][METHOD])]:
                old_e = load_npz(checked_ref(ROOT, original['evidence']))
                reported = d if label == '640' else d['comparison_to_960']
                compare(reported['control_policies'] if label == '640' else reported['policies'], original['policies'], 'control stats')
                compare(reported['control_extended'] if label == '640' else reported['extended'], old['methods'][METHOD]['control_extended'] if label == '640' else original['extended'], 'control extended')
                compare(reported['partitions'], expected_partitions(e, old_e, truth, p['policies']), 'partitions')
                old_pred = np.zeros_like(truth); old_pred[mask] = old_e['prediction_on_mask']; expected = {}
                for name, policy in p['policies'].items():
                    a, b = accepted(e, mask, policy), accepted(old_e, mask, policy); expected[name] = {}
                    for tag, take in [('common', a&b), ('gained', a&~b), ('lost', b&~a)]:
                        expected[name][tag] = {}
                        if tag != 'lost': expected[name][tag]['new'] = gradient_metrics(pred, truth, take)
                        if tag != 'gained': expected[name][tag]['control'] = gradient_metrics(old_pred, truth, take)
                compare(reported['partition_gradients'] if label == '640' else reported['gradient_partitions'], expected, 'gradient partitions')
            audited[(scene['scene'], job['key'])] = checks
            del e, replay, old_e, pred, old_pred, clean, permitted, regs; gc.collect()
        for c in scene['conditions']:
            row = conditions[(scene['scene'], c['spec']['key'])]
            if not row['changed']: continue
            r = json.loads(checked_ref(ROOT, row['record']).read_text())
            require(r['signature'] == sig and r['condition'] == c and r['scene'] == scene['scene'], 'condition identity')
            require(r['geometry'] == groups[(scene['scene'], c['job'])]['record'], 'condition geometry')
            g = json.loads(checked_ref(ROOT, r['geometry']).read_text())
            prior = json.loads(checked_ref(ROOT, row['parent']).read_text())
            require(r['same_pixel_controls'][OTHER] == prior['same_pixel_controls'][OTHER], 'homography controls changed')
            baseline = json.loads(checked_ref(ROOT, c['baseline']).read_text()); e = load_npz(checked_ref(ROOT, g['methods'][METHOD]['evidence'])); expected = {}
            require(baseline['geometry'] == g['job']['control'], 'baseline correspondence')
            for name, policy in p['policies'].items():
                take = accepted(e, e['mask'], policy)[e['mask']]
                expected[name] = {b: expected_same(e['prediction_on_mask'], load_npz(checked_ref(ROOT, d['evidence']))['prediction_on_mask'], truth[e['mask']], take) for b, d in baseline['baselines'].items()}
            compare(r['same_pixel_controls'][METHOD], expected, 'same-pixel damaged/Telea controls')
        print('AUDITED', scene['scene'], len(jobs), flush=True)
        del images, allowed, truth, weights, texture; gc.collect()
    require(len(audited) == 6, 'changed groups incomplete')
    result = dict(passed=True, created_at_utc=datetime.now(timezone.utc).isoformat(), plan_sha256=sha(pp), registry_sha256=sha(OUT/'registry.json'),
                  validator_sha256=sha(__file__), groups=126, conditions=216, changed_groups=6, reused_groups=120, changed_conditions=8, reused_conditions=208,
                  changed_method_evaluations=6, inherited_audit=p['parent_audit'], files_hash_verified=len(seen_files), matcher_executed=False, solver_executed=False,
                  changed=[dict(scene=s, job=j, checks=v) for (s, j), v in audited.items()])
    immutable_json(OUT/'audit_001.json', result)
    print(json.dumps({k: result[k] for k in ['passed', 'groups', 'conditions', 'changed_groups', 'files_hash_verified']}))


if __name__ == '__main__':
    with heavy_job(): audit()
