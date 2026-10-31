"""Low-memory real-road input-isolation experiment, without model inference.

No learned features, recovery estimates or model-performance claims are made.
The deliberately unsafe control masks only after resizing to expose leakage.
"""
import gc, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from benchmark import ROOT, digest, masks, save_json
from repair import nadir
from dense_matching import prepare_masked
from resource_guard import admission


def unsafe_resize_then_mask(image, allowed, prepared):
    x0, y0 = prepared['origin']
    ys, xs = np.nonzero(allowed)
    x1, y1 = min(allowed.shape[1], int(xs.max())+33), min(allowed.shape[0], int(ys.max())+33)
    rw = round((x1-x0)*prepared['scale'][0]); rh = round((y1-y0)*prepared['scale'][1])
    gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    gray = cv2.resize(gray, (rw, rh), interpolation=cv2.INTER_AREA)
    post_mask = cv2.resize(allowed[y0:y1, x0:x1].astype(np.uint8), (rw, rh), interpolation=cv2.INTER_NEAREST) > 0
    gray[~post_mask] = 0
    return gray


def main():
    root = ROOT/'sequence_v6'; protocol = root/'protocol.json'
    signature = {n: digest(ROOT/n) for n in ['preflight_dense.py', 'dense_matching.py', 'resource_guard.py', 'repair.py', 'benchmark.py']}
    signature['protocol'] = digest(protocol)
    output = root/'preflight.json'
    if output.exists():
        previous = json.loads(output.read_text())
        assert previous['signature'] == signature, 'Frozen preflight inputs changed'
        print('RESUME', output); return
    cfg = json.loads((ROOT/'benchmark_v2/protocol.json').read_text())
    results = []
    for case in ['gironde_0', 'alsace_fields_0', 'bourges_turn_001']:
        if case == 'bourges_turn_001':
            annotation = ROOT/'benchmark_v3/annotations_train_001.json'
            a = json.loads(annotation.read_text()); frame = a['frames']['target']
            source = ROOT/frame['image']; expected = frame['sample_sha256']
            manifest = ROOT/'benchmark_v3/prepared'/case/'manifest.json'
            m = json.loads(manifest.read_text()); assert digest(annotation) == m['annotation_sha256']
            frozen = ROOT/'benchmark_v3/prepared'/case/'masks.npz'; assert digest(frozen) == m['mask_sha256']
            with np.load(frozen) as z:
                allowed = z['target_allowed']; regions = {k.removeprefix('region_'): z[k] for k in z.files if k.startswith('region_')}
            size, fov = a['projection']['size'], a['projection']['fov']
        else:
            data = ROOT/'benchmark_v2/data'/case; source = data/'target.jpg'; manifest = data/'manifest.json'
            frame = next(r for r in json.loads(manifest.read_text())['images'] if r['name']=='target'); expected = frame['sample_sha256']
            _, als, _ = masks(cfg, cfg['cases'][case]); allowed = als['target']
            regions = {}
            for name in ['footprint', 'hidden_front', 'hidden_rear', 'local_front', 'local_rear']:
                with np.load(ROOT/'benchmark_v2/results'/case/'disk_lightglue'/(name+'_evidence.npz')) as z:
                    regions[name] = z['mask']
            size, fov = cfg['size'], cfg['fov']
        assert digest(source) == expected
        im = nadir(np.asarray(Image.open(source)), size, fov)
        for name, hidden in regions.items():
            als = allowed & ~hidden
            safe = prepare_masked(im, als)
            bad = unsafe_resize_then_mask(im, als, safe)
            pattern_rows = []
            for value in [0, 255, 'invert']:
                changed = im.copy()
                changed[~als] = 255 - im[~als] if value == 'invert' else value
                got = prepare_masked(changed, als)
                assert np.array_equal(got['gray'], safe['gray']) and np.array_equal(got['mask'], safe['mask'])
                unsafe = unsafe_resize_then_mask(changed, als, got)
                pattern_rows.append(dict(pattern=value, safe_changed_pixels=int(np.count_nonzero(got['gray'] != safe['gray'])),
                                         unsafe_changed_pixels=int(np.count_nonzero(unsafe != bad))))
            control = ROOT/'benchmark_v3/results'/case/(name+'.json') if case=='bourges_turn_001' else ROOT/'sequence_v5/results/sequence_v1'/case/(name+'.json')
            c = json.loads(control.read_text())
            methods = {k:v['policies']['candidate'] for k,v in c['methods'].items() if k in ['metric_free','joint_homography','homography_converged']}
            results.append(dict(case=case, region=name, source_path=str(source.relative_to(ROOT)), source_sha256=expected,
                                manifest_path=str(manifest.relative_to(ROOT)), manifest_sha256=digest(manifest),
                                mask_sha256=hashlib.sha256(hidden.tobytes()).hexdigest(),
                                allowed_sha256=hashlib.sha256(als.tobytes()).hexdigest(),
                                prepared_gray_sha256=hashlib.sha256(safe['gray'].tobytes()).hexdigest(),
                                prepared_shape=list(safe['gray'].shape), eligible_input_pixels=int(safe['mask'].sum()),
                                checks=pattern_rows, prior_control_path=str(control.relative_to(ROOT)),
                                prior_control_sha256=digest(control), prior_control_candidate_metrics=methods))
            del changed, got, unsafe, safe, bad
        del im, regions; gc.collect()
    out = dict(created=datetime.now(timezone.utc).isoformat(), signature=signature, regions=len(results),
               perturbation_checks=sum(len(r['checks']) for r in results),
               all_safe_inputs_bit_identical=True,
               unsafe_control_detected_in_regions=sum(any(p['unsafe_changed_pixels'] for p in r['checks']) for r in results),
               model_inference_performed=False, new_recovery_measurements=None,
               status='input_isolation_passed_model_experiment_blocked_by_storage',
               resources=admission(ROOT, reserve_bytes=512*1024**2), results=results,
               limitation='Preprocessing invariance only; actual learned matches, MPS operation, registration and recovery still need validation. Previous controls are references, not reruns.')
    save_json(output, out)
    print(json.dumps({k:v for k,v in out.items() if k not in ['signature','results']}))


if __name__=='__main__':
    main()
