"""Validate v4 artifacts and summarize error/coverage with matched-pixel controls."""
import json
from pathlib import Path
import numpy as np
from benchmark import ROOT,digest,save_json

METHODS=['homography_converged','affine','affine_stable','similarity','similarity_stable']
CASES=['strasbourg_outer_0','alsace_fields_0','bordeaux_0','gironde_0','nantes_1']

def main():
 root=ROOT/'sequence_v4';cfg=json.loads((ROOT/'benchmark_v2/protocol.json').read_text());files=sorted((root/'results').glob('*/*/*.json'));assert len(files)==55,len(files)
 rows=[];checks=0;nonconverged=0
 for path in files:
  d=json.loads(path.read_text());rows.append(d);case,region,source=d['case'],d['region'],d['source_experiment']
  manifest=ROOT/'dataset_v1/windows'/(case+'.json') if source=='sequence_v3' else ROOT/'benchmark_v2/data'/case/'manifest.json'
  for key,value in d['signature'].items():
   p=root/'protocol.json' if key=='protocol' else ROOT/'benchmark_v2/protocol.json' if key=='baseline_protocol' else ROOT/source/'results'/case/(region+'.json') if key=='upstream' else manifest if key=='manifest' else ROOT/key
   assert digest(p)==value,(path,key)
  mask=np.load(ROOT/'benchmark_v2/results'/case/'disk_lightglue'/(region+'_evidence.npz'))['mask']
  for method,m in d['methods'].items():
   e=np.load(path.parent/(region+'_'+method+'_evidence.npz'));assert np.array_equal(e['mask'],mask);assert not e['count'][~mask].any()
   graph=d['graphs'][method.removesuffix('_stable')]
   if not graph.get('optimizer_success',False):assert not e['count'].any();nonconverged+=1
   if method.endswith('_stable'):
    base=np.load(path.parent/(region+'_'+method.removesuffix('_stable')+'_evidence.npz'));assert np.all(e['count']<=base['count'])
   for name,p in cfg['policies'].items():
    accept=mask&(e['count']>=p['min_views'])&(e['disagreement']<=p['max_disagreement']);v=m['policies'][name]
    assert accept.sum()==v['accepted_pixels'];assert (e['provenance'][accept]>=0).all();assert (e['provenance'][accept]<len(d['source_order'])).all()
    if region!='footprint' and accept.any():
     errors=e['absolute_rgb_error_sum_on_mask'][accept[mask]].astype(float)/3
     assert abs(errors.mean()-v['error']['mae_255'])<1e-6
     assert abs((errors>20).mean()-v['error']['over_20_fraction'])<1e-10
   checks+=1
 result=dict(scope='Same five car roads, now development; no hidden actual vehicle-ground truth and no independently held-out test performance.',methods={},per_case={})
 cohort=[d for d in rows if d['source_experiment']=='sequence_v1' and d['case'] in CASES]
 for method in METHODS:
  out={}
  for group,regions in [('footprint',['footprint']),('large_hidden',['hidden_front','hidden_rear']),('local_hidden',['local_front','local_rear'])]:
   samples=[d for d in cohort if d['region'] in regions];out[group]={}
   for policy in cfg['policies']:
    vals=[d['methods'][method]['policies'][policy] for d in samples];total=sum(d['mask_pixels'] for d in samples);n=sum(v['accepted_pixels'] for v in vals)
    s=dict(mask_pixels=total,accepted_pixels=n,pooled_coverage=n/max(total,1),macro_coverage=float(np.mean([v['coverage'] for v in vals])))
    if group!='footprint':
     for field in ['mae_255','over_20_fraction']:
      s[field]=sum(v['accepted_pixels']*v['error'][field] for v in vals if v['accepted_pixels'])/n if n else None
     s['telea_mae_same_pixels']=sum(v['accepted_pixels']*v['telea_same_pixels']['mae_255'] for v in vals if v['accepted_pixels'])/n if n else None
     if method!='homography_converged':
      common=[d['same_pixel_control'][method][policy] for d in samples];cn=sum(v['new']['pixels'] for v in common)
      s['common_pixels']=cn;s['common_new_mae']=sum(v['new']['pixels']*v['new']['mae_255'] for v in common if v['new']['pixels'])/cn if cn else None
      s['common_control_mae']=sum(v['control']['pixels']*v['control']['mae_255'] for v in common if v['control']['pixels'])/cn if cn else None
    out[group][policy]=s
  result['methods'][method]=out
 for d in rows:
  result['per_case'].setdefault(d['source_experiment']+'/'+d['case'],{})[d['region']]={m:{p:v['coverage'] for p,v in d['methods'][m]['policies'].items()} for m in METHODS}
 save_json(root/'summary.json',result);save_json(root/'validation.json',dict(regions=len(rows),region_method_runs=checks,hashes_masks_provenance_errors_pass=True,nonconverged_or_disconnected_method_runs_forced_to_abstain=nonconverged))
 for m,d in result['methods'].items():print(m,'footprint',round(d['footprint']['candidate']['macro_coverage'],4),'large',d['large_hidden']['candidate'],'local',d['local_hidden']['candidate'])
if __name__=='__main__':main()
