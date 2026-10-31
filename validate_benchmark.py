import json,hashlib,sys
from pathlib import Path
import numpy as np
r=Path(__file__).resolve().parent;root=r/'benchmark_v2';config=json.loads((root/'protocol.json').read_text());checks=0
for case in config['cases']:
 manifest=json.loads((root/'data'/case/'manifest.json').read_text())
 assert len({x['id'] for x in manifest['images']})==5
 for row in manifest['images']:
  assert hashlib.sha256((root/'data'/case/(row['name']+'.jpg')).read_bytes()).hexdigest()==row['sample_sha256']
 for method in ['sift','disk_lightglue']:
  folder=root/'results'/case/method
  for file in ['metrics.json','local_probes.json']:
   d=json.loads((folder/file).read_text())
   for name,sig in d['signature'].items():
    path=root/('protocol.json' if name=='protocol' else 'local_probe_protocol.json') if name in ['protocol','probe_protocol'] else root/'data'/case/'manifest.json' if name=='manifest' else r/name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==sig,(case,method,name)
   for region,v in d['regions'].items():
    e=np.load(folder/(region+'_evidence.npz'));mask=e['mask'];assert mask.sum()==v['pixels'];assert not e['count'][~mask].any()
    for name,p in config['policies'].items():
     accept=mask&(e['count']>=p['min_views'])&(e['disagreement']<=p['max_disagreement'])
     m=v['policies'][name];assert accept.sum()==m['accepted_pixels'];assert np.all((e['provenance'][accept]>=0)&(e['provenance'][accept]<4))
     assert m['single_view_pixels']+m['multi_view_pixels']==m['accepted_pixels']
     assert (folder/(region+'_'+name+'.jpg')).exists();assert (folder/(region+'_'+name+'_support.jpg')).exists()
    checks+=1
print('Verified',checks,'region-method runs; source, protocol, implementation hashes; masks, provenance and all policy counts.')
