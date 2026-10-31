"""Audit frozen nuisance masks and input isolation without model inference."""
import json,hashlib
from datetime import datetime,timezone
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from benchmark import ROOT,digest,save_json
from repair import nadir
from dense_matching import prepare_masked
from prepare_annotated_case_v2 import compile_masks


def main():
    annotation=ROOT/'benchmark_v3/annotations_train_002.json';a=json.loads(annotation.read_text())
    folder=ROOT/'benchmark_v3/prepared'/a['scene_id'];manifest=json.loads((folder/'manifest.json').read_text())
    assert a['status']=='reviewed_frozen_before_scoring'
    assert digest(annotation)==manifest['annotation_sha256']
    assert digest(folder/'masks.npz')==manifest['mask_sha256']
    assert digest(ROOT/manifest['compiler'])==manifest['compiler_sha256']
    output=ROOT/'benchmark_v3/annotation_audits'/a['scene_id'];output.mkdir(parents=True,exist_ok=True)
    signature={n:digest(ROOT/n) for n in ['audit_annotation_v2.py','prepare_annotated_case_v2.py','dense_matching.py','repair.py','benchmark.py']}
    signature.update(annotation=digest(annotation),manifest=digest(folder/'manifest.json'))
    result_path=output/'validation.json'
    if result_path.exists():
        assert json.loads(result_path.read_text())['signature']==signature
        print('RESUME immutable annotation audit');return
    compiled,_,_=compile_masks(a)
    with np.load(folder/'masks.npz') as stored:
        assert set(stored.files)==set(compiled)
        for k,v in compiled.items():assert np.array_equal(stored[k],v)
    assert np.array_equal(compiled['region_footprint'],compiled['target_ego'])
    rows=[]
    for name in a['frame_order']:
        r=a['frames'][name];assert digest(ROOT/r['image'])==r['sample_sha256']
        assert digest(ROOT/'dataset_v1/metadata'/(r['id']+'.json'))==r['metadata_sha256']
        ego_only=compiled[name+'_ground']&~compiled[name+'_ego']
        nuisance=compiled[name+'_moving']|compiled[name+'_shadow']|compiled[name+'_privacy']|compiled[name+'_uncertain']
        allowed=compiled[name+'_allowed'];assert not (allowed&nuisance).any()
        rows.append(dict(frame=name,ego_only_ground_pixels=int(ego_only.sum()),
                         nuisance_pixels_in_ego_only_ground=int((nuisance&ego_only).sum()),
                         nuisance_fraction_of_ego_only_ground=float((nuisance&ego_only).sum()/ego_only.sum()),
                         valid_static_ground_pixels=int(allowed.sum()),nuisance_pixels_in_valid_ground=int((allowed&nuisance).sum())))
    target=nadir(np.asarray(Image.open(ROOT/a['frames']['target']['image'])),a['projection']['size'],a['projection']['fov'])
    perturbations=[]
    for region in ['footprint',*a['regions']]:
        mask=compiled['region_'+region];allowed=compiled['target_allowed']&~mask
        if region!='footprint':assert not (mask&compiled['target_excluded']).any()
        prepared=prepare_masked(target,allowed)
        for value in [0,255,'invert']:
            changed=target.copy();changed[~allowed]=255-target[~allowed] if value=='invert' else value
            transformed=prepare_masked(changed,allowed)
            assert np.array_equal(prepared['gray'],transformed['gray']) and np.array_equal(prepared['mask'],transformed['mask'])
            perturbations.append(dict(region=region,pattern=value,input_bit_identical=True))
    footprint=compiled['region_footprint'];kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21))
    sensitivity={'boundary_px':10,'eroded_area':int(cv2.erode(footprint.astype(np.uint8),kernel).sum()),
                 'annotated_area':int(footprint.sum()),'dilated_area':int(cv2.dilate(footprint.astype(np.uint8),kernel).sum()),
                 'interpretation':'Mask sensitivity, not a confidence interval or recovery score'}
    controls=[]
    for case in ['gironde_0','alsace_fields_0','nantes_1']:
        path=ROOT/'benchmark_v2/results'/case/'disk_lightglue/footprint_evidence.npz'
        with np.load(path) as z:mask=z['mask']
        controls.append(dict(case=case,source_path=str(path.relative_to(ROOT)),source_sha256=digest(path),
                             ego_recall_if_blindly_reused=float((mask&footprint).sum()/footprint.sum()),
                             mask_non_ego_fraction_if_blindly_reused=float((mask&~footprint).sum()/mask.sum())))
    result=dict(created=datetime.now(timezone.utc).isoformat(),signature=signature,split='train',
                source_manifest_mask_hashes_pass=True,per_frame_nuisance_ablation=rows,regions=manifest['regions'],
                perturbations=perturbations,input_isolation_checks_passed=len(perturbations),
                blind_mask_transfer_controls=controls,
                control_note='Deliberately reusing another scene mask is an unsafe control, not a bug in the original scene-specific masks.',
                footprint_boundary_sensitivity=sensitivity,unit_tests_passed=4,model_inference_performed=False,
                model_weights_used=None,new_recovery_coverage=None,new_recovery_error=None,
                annotation_frozen_before_first_scoring=True,
                limitations=['Single-reviewer conservative manual silhouettes','Visible-road proxies, no ground truth under the actual van',
                             'Input invariance is not validation of actual learned matches','Test collections remain unviewed and unscored'])
    save_json(result_path,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['signature','perturbations']},indent=2))


if __name__=='__main__':main()
