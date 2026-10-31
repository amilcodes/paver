"""Run the real-image planar baseline and hidden-patch development checks."""
import argparse
import json
from pathlib import Path
import platform
import hashlib
import cv2
import numpy as np
from PIL import Image
from repair import nadir, polygon_mask, register, fuse, score

ROOT = Path(__file__).resolve().parent


def save(path, image):
    Image.fromarray(image).save(path)


def run(sample, config_path, output):
    cv2.setNumThreads(1)
    config = json.loads(config_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    names = ['target', 'prev', 'next']
    manifest=json.loads((sample/'manifest.json').read_text())
    for record in manifest['images']:
        digest=hashlib.sha256((sample/f'{record["name"]}.jpg').read_bytes()).hexdigest()
        if digest != record['sample_sha256']:
            raise ValueError(f'Sample checksum mismatch: {record["name"]}')
    images = {name: nadir(np.array(Image.open(sample/f'{name}.jpg').convert('RGB')),
                          config['size'], config['fov']) for name in names}
    target = images['target']
    vehicle = polygon_mask(target.shape, config['vehicle'])
    allowed = {name: polygon_mask(target.shape, config['ground'][name]) & ~vehicle
               for name in names}
    sources = [images[n] for n in names[1:]]
    source_masks = [allowed[n] for n in names[1:]]
    for name in names:
        save(output/f'{name}_nadir.jpg', images[name])
        save(output/f'{name}_allowed.png', allowed[name].astype(np.uint8)*255)
    masks = {name: polygon_mask(target.shape, poly) & allowed['target']
             for name, poly in config['holdouts'].items()}
    masks['vehicle_footprint'] = vehicle
    report = dict(dataset='One Panoramax sequence, three real panoramas',
                  status='Development checks; neither independent test set nor Google Street View evaluation',
                  configuration=config, source_manifest=manifest, versions=dict(python=platform.python_version(),
                  numpy=np.__version__, opencv=cv2.__version__), cases=[])
    for case_name, mask in masks.items():
        if not mask.any():
            raise ValueError(f'Empty mask: {case_name}')
        truth_available = case_name != 'vehicle_footprint'
        case_dir = output/case_name
        case_dir.mkdir(exist_ok=True)
        corrupted = target.copy()
        corrupted[mask] = [32, 37, 45]
        # No target ground truth in the missing region reaches registration.
        regs = [register(src, corrupted, sm, allowed['target'] & ~mask)
                for src, sm in zip(sources, source_masks)]
        result = fuse(corrupted, mask, sources, source_masks, regs, **config['fusion'])
        telea = cv2.inpaint(corrupted, mask.astype(np.uint8)*255, 3, cv2.INPAINT_TELEA)
        provenance = np.full_like(target, [22, 30, 39])
        provenance[mask] = [242, 156, 74]
        provenance[result['accepted'] & (result['provenance']==0)] = [73, 190, 221]
        provenance[result['accepted'] & (result['provenance']==1)] = [176, 140, 244]
        evidence = np.zeros_like(target)
        evidence[mask] = [242, 156, 74]
        evidence[result['accepted'] & (result['count']==1)] = [73, 190, 221]
        evidence[result['accepted'] & (result['count']>=2)] = [101, 212, 158]
        for name, array in [('input',corrupted),('repaired',result['image']),
                            ('telea',telea),('provenance',provenance),('evidence',evidence)]:
            save(case_dir/f'{name}.png',array)
        save(case_dir/'mask.png',mask.astype(np.uint8)*255)
        np.savez_compressed(case_dir/'evidence.npz',provenance=result['provenance'],
                            observation_count=result['count'],accepted=result['accepted'],
                            disagreement=result['disagreement'])
        accepted = result['accepted']
        record = dict(name=case_name, ground_truth_available=truth_available,
                      mask_pixels=int(mask.sum()), repaired_pixels=int(accepted.sum()),
                      coverage=float(accepted.sum()/mask.sum()),
                      single_view_pixels=int((accepted & (result['count']==1)).sum()),
                      agreeing_multi_view_pixels=int((accepted & (result['count']>=2)).sum()),
                      unresolved_pixels=int((mask & ~accepted).sum()),
                      registrations={n:r.diagnostics for n,r in zip(names[1:],regs)})
        if truth_available:
            record['metrics'] = dict(
                multiview_on_accepted=score(result['image'],target,accepted),
                telea_on_same_accepted=score(telea,target,accepted),
                multiview_full_mask_including_abstentions=score(result['image'],target,mask),
                telea_full_mask=score(telea,target,mask),
                no_repair_full_mask=score(corrupted,target,mask))
        record['threshold_sweep'] = []
        for minimum in [1,2]:
            for threshold in [8.,18.,32.]:
                variant = fuse(corrupted,mask,sources,source_masks,regs,threshold,minimum)
                sweep = dict(min_views=minimum,threshold=threshold,
                             coverage=float(variant['accepted'].sum()/mask.sum()))
                if truth_available:
                    sweep['error'] = score(variant['image'],target,variant['accepted'])
                record['threshold_sweep'].append(sweep)
        report['cases'].append(record)
        print(case_name, f'coverage={record["coverage"]:.1%}',
              [(n,r.diagnostics['inliers'],r.diagnostics.get('reason')) for n,r in zip(names[1:],regs)],flush=True)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample',type=Path,default=ROOT/'sample')
    parser.add_argument('--config',type=Path,default=ROOT/'sample_config.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results')
    args=parser.parse_args()
    run(args.sample,args.config,args.output)
