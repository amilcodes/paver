"""Verify exact experiment code/data hashes and saved coverage evidence."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 config=json.loads((ROOT/'benchmark_v2/protocol.json').read_text());runs=0;regions=0;warnings=[]
 for exp in ['sequence_v1','sequence_v2','sequence_v3']:
  root=ROOT/exp;files=list((root/'results').glob('*/*.json'))
  assert len(files)==(10 if exp=='sequence_v3' else 45),(exp,len(files))
  for path in files:
   d=json.loads(path.read_text());case=d['case'];region=d['region'];regions+=1
   manifest=ROOT/'dataset_v1/windows'/(case+'.json') if exp=='sequence_v3' else ROOT/'benchmark_v2/data'/case/'manifest.json'
   for name,value in d['signature'].items():
    p=root/'protocol.json' if name=='sequence_protocol' else ROOT/'benchmark_v2/protocol.json' if name=='baseline_protocol' else manifest if name=='manifest' else ROOT/name
    assert sha(p)==value,(exp,case,region,name)
   for pair in d['correspondences']:assert sha(root/pair['path'])==pair['sha256']
   original=np.load(ROOT/'benchmark_v2/results'/case/'disk_lightglue'/(region+'_evidence.npz'))['mask']
   for method,result in d['methods'].items():
    e=np.load(path.parent/(region+'_'+method+'_evidence.npz'));mask=e['mask'];assert np.array_equal(mask,original)
    assert mask.sum()==d['mask_pixels'];assert not e['count'][~mask].any()
    for name,p in config['policies'].items():
     accept=mask&(e['count']>=p['min_views'])&(e['disagreement']<=p['max_disagreement']);m=result['policies'][name]
     assert int(accept.sum())==m['accepted_pixels'];assert abs(m['coverage']-accept.sum()/max(mask.sum(),1))<1e-10
     assert np.all((e['provenance'][accept]>=0)&(e['provenance'][accept]<len(d['source_order'])))
     assert m['single_view_pixels']+m['multi_view_pixels']==m['accepted_pixels']
     if region!='footprint':assert m['error']['pixels']==m['accepted_pixels']
    runs+=1
   if d['graph'].get('optimizer_success') is False:warnings.append([exp,case,region,'optimizer did not converge; do not promote'])
 out={'region_method_runs':runs,'regions':regions,'all_hashes_masks_provenance_and_policy_counts_pass':True,'warnings':warnings}
 (ROOT/'autoresearch/validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
