"""Supplemental small hidden patches, added after large-footprint abstentions.

Exploratory diagnostic only: neither full-car validation nor calibrated risk.
Fixed rectangles immediately ahead/behind the annotated rig, clipped to ground.
"""
import argparse,json,time,gc
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import cv2
from PIL import Image
from benchmark import ROOT,digest,save_json,save_image,masks,policy_mask,error_metrics
from repair import nadir,polygon_mask,fuse
from matching_v2 import LearnedMatcher,SIFTMatcher,register_learned

def main(method,device):
 root=ROOT/'benchmark_v2';config=json.loads((root/'protocol.json').read_text())
 protocol_path=root/'local_probe_protocol.json'
 if not protocol_path.exists():
  save_json(protocol_path,dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
   added_after_large_holdout_results=True,note='Exploratory local accuracy check. Fixed 0.08-wide by 0.06-high boxes centered at x=0.49, 0.06 before/after padded rig bounds. Intersect with annotated visible ground. No matcher/gate changes.',width=.08,height=.06,x=.49,offset=.06))
 signature={n:digest(ROOT/n) for n in ['repair.py','matching_v2.py','benchmark.py','local_probes.py']}
 signature['protocol']=digest(root/'protocol.json');signature['probe_protocol']=digest(protocol_path)
 matcher=LearnedMatcher(device) if method=='disk_lightglue' else SIFTMatcher()
 for key,case in config['cases'].items():
  out=root/'results'/key/method;out.mkdir(parents=True,exist_ok=True);data=root/'data'/key
  manifest=json.loads((data/'manifest.json').read_text());sig=dict(signature,manifest=digest(data/'manifest.json'))
  if (out/'local_probes.json').exists():
   old=json.loads((out/'local_probes.json').read_text())
   if old['signature']==sig:print('RESUME',key,method,flush=True);continue
   raise RuntimeError('Existing probes have different signature')
  for row in manifest['images']:
   if digest(data/(row['name']+'.jpg'))!=row['sample_sha256']:raise ValueError('Source checksum changed')
  if len({x['id'] for x in manifest['images']})!=5:raise ValueError('Five distinct observations required')
  images={n:nadir(np.array(Image.open(data/(n+'.jpg'))),config['size'],config['fov']) for n in ['target']+config['source_names']}
  vehicle,allowed,_=masks(config,case);target=images['target'];sources=[images[n] for n in config['source_names']];sa=[allowed[n] for n in config['source_names']]
  ys=[p[1] for poly in case['vehicle'] for p in poly];regions={}
  for region,cy in [('local_front',min(ys)-.06),('local_rear',max(ys)+.06)]:
   mask=polygon_mask(target.shape,[[[.45,cy-.03],[.53,cy-.03],[.53,cy+.03],[.45,cy+.03]]])&allowed['target']
   clean=target.copy();clean[vehicle|mask]=0
   regs=[register_learned(matcher,s,clean,a,allowed['target']&~mask) for s,a in zip(sources,sa)]
   base=fuse(clean,mask,sources,sa,regs,agreement_threshold=255)
   telea=cv2.inpaint(clean,mask.astype(np.uint8)*255,3,cv2.INPAINT_TELEA)
   m=dict(pixels=int(mask.sum()),relative_to_footprint=float(mask.sum()/vehicle.sum()),registrations=[r.diagnostics for r in regs],policies={},telea_full=error_metrics(telea,target,mask))
   np.savez_compressed(out/(region+'_evidence.npz'),mask=mask,provenance=base['provenance'],count=base['count'],disagreement=base['disagreement'])
   save_image(out/(region+'_input.jpg'),clean);save_image(out/(region+'_telea.jpg'),telea)
   for name,p in config['policies'].items():
    accepted=policy_mask(base,mask,p);error=np.abs(base['image'].astype(np.float32)-target).mean(2)
    m['policies'][name]=dict(accepted_pixels=int(accepted.sum()),coverage=float(accepted.sum()/max(1,mask.sum())),single_view_pixels=int((accepted&(base['count']==1)).sum()),multi_view_pixels=int((accepted&(base['count']>=2)).sum()),accepted_error=error_metrics(base['image'],target,accepted),telea_same_pixels=error_metrics(telea,target,accepted),covered_with_error_le_20_fraction=float((error[accepted]<=20).sum()/max(1,mask.sum())))
    preview=target.copy();preview[accepted]=base['image'][accepted];save_image(out/(region+'_'+name+'.jpg'),preview)
    evidence=target.copy();evidence[mask]=[222,83,109];evidence[accepted&(base['count']==1)]=[244,178,76];evidence[accepted&(base['count']>=2)]=[43,199,174]
    save_image(out/(region+'_'+name+'_support.jpg'),evidence)
   regions[region]=m
   print(key,method,region,{k:(round(v['coverage'],3),None if v['accepted_error']['mae_255'] is None else round(v['accepted_error']['mae_255'],2)) for k,v in m['policies'].items()},flush=True)
   del base,regs,telea,error;gc.collect()
  save_json(out/'local_probes.json',dict(signature=sig,case=key,method=method,regions=regions))
  matcher.cache.clear();del images,sources,target;gc.collect()
  if hasattr(matcher,'torch') and matcher.device=='mps':matcher.torch.mps.empty_cache()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--method',required=True,choices=['sift','disk_lightglue']);p.add_argument('--device',default='auto');a=p.parse_args();main(a.method,a.device)
