"""Compare whole-neighborhood geometry against a same-feature pairwise control.

Writes compact numeric results and inspectable evidence, no formatted reports.
"""
import argparse,gc,hashlib,json,time
from pathlib import Path
from datetime import datetime,timezone
import cv2
import numpy as np
from PIL import Image
from benchmark import ROOT,digest,masks,save_json,save_image,policy_mask,error_metrics
from repair import nadir,polygon_mask,fuse
from matching_v2 import fit_registration
from sequence_geometry import TiledMatcher,propose_edge,sequence_registrations


def run(args):
 root=ROOT/'sequence_v1';oldroot=ROOT/'benchmark_v2';config=json.loads((oldroot/'protocol.json').read_text());spec=json.loads((root/'protocol.json').read_text());names=spec['frames'];center=names.index('target')
 signature={name:digest(ROOT/name) for name in ['run_sequence.py','sequence_geometry.py','matching_v2.py','repair.py','benchmark.py']}
 signature.update(sequence_protocol=digest(root/'protocol.json'),baseline_protocol=digest(oldroot/'protocol.json'))
 matcher=TiledMatcher(args.device)
 for key,case in config['cases'].items():
  if args.case and key not in args.case:continue
  folder=root/'results'/key;folder.mkdir(parents=True,exist_ok=True);data=oldroot/'data'/key
  manifest=json.loads((data/'manifest.json').read_text());sig=dict(signature,manifest=digest(data/'manifest.json'))
  assert len(set(x['id'] for x in manifest['images']))==5
  for row in manifest['images']:assert digest(data/(row['name']+'.jpg'))==row['sample_sha256']
  images=[nadir(np.array(Image.open(data/(n+'.jpg'))),config['size'],config['fov']) for n in names]
  vehicle,allowed_by_name,holes=masks(config,case);allowed=[allowed_by_name[n] for n in names]
  ys=[p[1] for poly in case['vehicle'] for p in poly]
  for region,cy in [('local_front',min(ys)-.06),('local_rear',max(ys)+.06)]:
   holes[region]=polygon_mask(images[center].shape,[[[.45,cy-.03],[.53,cy-.03],[.53,cy+.03],[.45,cy+.03]]])&allowed[center]
  corr_cache={};start=time.time();source_indices=[i for i in range(len(names)) if i!=center]
  print('SCENE',key,flush=True)
  for region,mask in holes.items():
   if args.region and region not in args.region:continue
   result_path=folder/(region+'.json')
   if result_path.exists():
    previous=json.loads(result_path.read_text())
    if previous['signature']==sig:print('RESUME',key,region,flush=True);continue
    raise RuntimeError('Experiment changed; use a new version/output directory instead of replacing results')
   target=images[center].copy();target[vehicle|mask]=0
   ims=images.copy();ims[center]=target;als=allowed.copy();als[center]=allowed[center]&~mask
   edges=[];edge_info=[];direct={};pair_meta=[]
   for i in range(len(names)):
    for j in range(i+1,len(names)):
     if j-i>2 and center not in [i,j]:continue
     cachekey=(i,j,region if center in [i,j] else 'donor_only')
     if cachekey not in corr_cache:
      p,q,extra=matcher.correspondences(ims[i],ims[j],als[i],als[j]);corr_cache[cachekey]=(p,q,extra)
     p,q,extra=corr_cache[cachekey]
     edge,info=propose_edge(i,j,p,q);edge_info.append(info)
     if edge is not None:edges.append(edge)
     if center in [i,j]:
      source=j if i==center else i
      sp,tp=(q,p) if i==center else (p,q)
      direct[source]=fit_registration(sp,tp,ims[source],target,als[source],als[center],dict(extra,method='tiled_pairwise_control'))
     cachefolder=root/'correspondences'/key;cachefolder.mkdir(parents=True,exist_ok=True)
     cachefile=cachefolder/(f'{cachekey[2]}_{i}_{j}.npz')
     if not cachefile.exists():np.savez_compressed(cachefile,p=p,q=q)
     pair_meta.append(dict(i=i,j=j,path=str(cachefile.relative_to(root)),sha256=digest(cachefile)))
   joint,graph=sequence_registrations(ims,als,edges,center)
   result=dict(case=key,region=region,signature=sig,created_at_utc=datetime.now(timezone.utc).isoformat(),mask_pixels=int(mask.sum()),source_order=[names[i] for i in source_indices],pair_proposals=edge_info,correspondences=pair_meta,graph=graph,methods={})
   predictions={};acceptance={}
   if region!='footprint':telea=cv2.inpaint(target,mask.astype(np.uint8)*255,3,cv2.INPAINT_TELEA)
   for method,regs in [('tiled_pairwise',direct),('sequence_graph',joint)]:
    base=fuse(target,mask,[ims[i] for i in source_indices],[als[i] for i in source_indices],[regs[i] for i in source_indices],agreement_threshold=255)
    m=dict(registrations=[regs[i].diagnostics for i in source_indices],policies={});acceptance[method]={};predictions[method]=base['image']
    np.savez_compressed(folder/(region+'_'+method+'_evidence.npz'),mask=mask,provenance=base['provenance'],count=base['count'],disagreement=base['disagreement'])
    for name,p in config['policies'].items():
     accepted=policy_mask(base,mask,p);acceptance[method][name]=accepted
     v=dict(accepted_pixels=int(accepted.sum()),coverage=float(accepted.sum()/max(1,mask.sum())),single_view_pixels=int((accepted&(base['count']==1)).sum()),multi_view_pixels=int((accepted&(base['count']>=2)).sum()))
     if region!='footprint':
      v['error']=error_metrics(base['image'],images[center],accepted);v['telea_same_pixels']=error_metrics(telea,images[center],accepted)
     m['policies'][name]=v
    result['methods'][method]=m
    if region=='footprint':
     preview=images[center].copy();accepted=acceptance[method]['candidate'];preview[accepted]=base['image'][accepted]
     save_image(folder/(method+'_partial.jpg'),preview)
    del base;gc.collect()
   if region!='footprint':
    result['same_pixel_comparison']={}
    for p in config['policies']:
     common=acceptance['tiled_pairwise'][p]&acceptance['sequence_graph'][p]
     result['same_pixel_comparison'][p]={method:error_metrics(pred,images[center],common) for method,pred in predictions.items()}
   result['scene_elapsed_seconds']=time.time()-start;save_json(result_path,result)
   print('RESULT',key,region,'edges',len(edges),'connected',graph.get('connected_frames'),{m:{p:round(v['coverage'],4) for p,v in d['policies'].items()} for m,d in result['methods'].items()},flush=True)
   del predictions,acceptance,ims,target,als;gc.collect()
  matcher.cache.clear();del images;gc.collect()
  if matcher.device=='mps':matcher.torch.mps.empty_cache()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--case',action='append');p.add_argument('--region',action='append');p.add_argument('--device',default='auto');run(p.parse_args())
