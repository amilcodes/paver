"""Validate frozen annotations and every new-scene transfer record."""
import json,importlib.metadata,subprocess
from pathlib import Path
import numpy as np
from benchmark import ROOT,digest,save_json

def main():
 annopath=ROOT/'benchmark_v3/annotations_train_001.json';a=json.loads(annopath.read_text());folder=ROOT/'benchmark_v3/results'/a['scene_id'];prepared=ROOT/'benchmark_v3/prepared'/a['scene_id'];manifest=json.loads((prepared/'manifest.json').read_text());masks=np.load(prepared/'masks.npz');cfg=json.loads((ROOT/'benchmark_v2/protocol.json').read_text());checked=0
 assert digest(annopath)==manifest['annotation_sha256'];assert digest(prepared/'masks.npz')==manifest['mask_sha256']
 for frame in a['frames'].values():assert digest(ROOT/frame['image'])==frame['sample_sha256']
 for region in ['footprint',*a['regions']]:
  d=json.loads((folder/(region+'.json')).read_text())
  for key,value in d['signature'].items():
   path=annopath if key=='annotation' else prepared/'manifest.json' if key=='prepared' else ROOT/'benchmark_v2/protocol.json' if key=='policies' else ROOT/key
   assert digest(path)==value,(region,key)
  for ref in d['correspondences']:assert digest(ROOT/ref['path'])==ref['sha256']
  for method,result in d['methods'].items():
   e=np.load(folder/(region+'_'+method+'_evidence.npz'));mask=masks['region_'+region];assert np.array_equal(e['mask'],mask);assert not e['count'][~mask].any()
   if method in d['graphs'] and not d['graphs'][method].get('optimizer_success',False):assert not e['count'].any()
   for name,p in cfg['policies'].items():
    accepted=mask&(e['count']>=p['min_views'])&(e['disagreement']<=p['max_disagreement']);v=result['policies'][name];assert int(accepted.sum())==v['accepted_pixels']
    assert np.all((e['provenance'][accepted]>=0)&(e['provenance'][accepted]<4))
    if region!='footprint' and accepted.any():
     error=e['absolute_rgb_error_sum_on_mask'][accepted[mask]].astype(float)/3;assert abs(error.mean()-v['error']['mae_255'])<1e-6
   checked+=1
 model=json.loads((ROOT/'benchmark_v2/model_manifest.json').read_text());workspace=ROOT.parent.parent
 for name,value in model['models'].items():assert digest(workspace/'work/model-cache/hub/checkpoints'/name)==value
 versions={name:importlib.metadata.version(name) for name in model['packages']}
 assert versions==model['packages']
 commit=subprocess.check_output(['git','-C',str(workspace/'work/LightGlue'),'rev-parse','HEAD'],text=True).strip();assert commit==model['lightglue_commit']
 save_json(folder/'validation.json',dict(region_method_runs=checked,source_annotation_model_code_mask_provenance_pass=True,all_target_regions_abstained=True,reason='Only zero to two matches on every frame pair; target has no accepted edge',split='train',frozen_before_first_scoring=True,not_a_heldout_test_group=True))
 save_json(folder/'model_provenance.json',dict(model_manifest_sha256=digest(ROOT/'benchmark_v2/model_manifest.json'),models=model['models'],packages=versions,lightglue_commit=commit,device='mps',training_overlap='Pretrained model training overlap is unknown; no fine-tuning used.'))
 print('Verified',checked,'new-scene method/region runs, immutable masks, inputs and model provenance.')
if __name__=='__main__':main()
