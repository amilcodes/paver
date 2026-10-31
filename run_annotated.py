"""Apply unchanged research baselines to a newly frozen, annotated train scene."""
import argparse,json,gc,time
from pathlib import Path
from datetime import datetime,timezone
import cv2
import numpy as np
from PIL import Image
from benchmark import ROOT,digest,save_json,policy_mask,error_metrics
from repair import nadir,fuse
from matching_v2 import fit_registration
from sequence_geometry import TiledMatcher,propose_edge,fit_graph
from constrained_sequence import audit_poses
from metric_plane import fit_metric_graph,gradient_error


def run(path,device):
 a=json.loads(path.read_text());assert a['status']=='reviewed_frozen_before_scoring'
 root=ROOT/'benchmark_v3';prepared=root/'prepared'/a['scene_id'];manifest=json.loads((prepared/'manifest.json').read_text())
 assert manifest['annotation_sha256']==digest(path);assert manifest['mask_sha256']==digest(prepared/'masks.npz')
 assert a['split']==json.loads((ROOT/'dataset_v1/splits_v1.json').read_text())['collections'][a['collection']]['split']=='train'
 names=a['frame_order'];center=names.index('target');size=a['projection']['size'];fov=a['projection']['fov'];out=root/'results'/a['scene_id'];out.mkdir(parents=True,exist_ok=True)
 signature={n:digest(ROOT/n) for n in ['run_annotated.py','sequence_geometry.py','constrained_sequence.py','metric_plane.py','trajectory_matching.py','matching_v2.py','repair.py','benchmark.py']}
 signature.update(annotation=digest(path),prepared=digest(prepared/'manifest.json'),policies=digest(ROOT/'benchmark_v2/protocol.json'))
 policies=json.loads((ROOT/'benchmark_v2/protocol.json').read_text())['policies'];masks=np.load(prepared/'masks.npz');ims=[];allowed=[];metadata=[]
 for name in names:
  rec=a['frames'][name];source=ROOT/rec['image'];assert digest(source)==rec['sample_sha256']
  ims.append(nadir(np.array(Image.open(source)),size,fov));allowed.append(masks[name+'_allowed']);metadata.append(json.loads((ROOT/'dataset_v1/metadata'/(rec['id']+'.json')).read_text()))
 assert len(set(a['frames'][n]['id'] for n in names))==len(names)
 matcher=TiledMatcher(device);corr_cache={};donors=[i for i in range(len(names)) if i!=center]
 for region in ['footprint',*a['regions']]:
  output=out/(region+'.json')
  if output.exists():assert json.loads(output.read_text())['signature']==signature;print('RESUME',region,flush=True);continue
  start=time.time();mask=masks['region_'+region];target=ims[center].copy();target[mask|masks['target_ego']]=0;images=ims.copy();images[center]=target;als=allowed.copy();als[center]=allowed[center]&~mask
  assert not (mask&allowed[center]).any() if region=='footprint' else not (mask&~allowed[center]).any()
  edges=[];refs=[];direct={};proposals=[]
  for i in range(len(names)):
   for j in range(i+1,len(names)):
    if j-i>2 and center not in [i,j]:continue
    key=(i,j,region if center in [i,j] else 'donor_only')
    if key not in corr_cache:corr_cache[key]=matcher.correspondences(images[i],images[j],als[i],als[j])
    p,q,extra=corr_cache[key];edge,info=propose_edge(i,j,p,q);proposals.append(info)
    if edge is not None:edges.append(edge)
    if center in [i,j]:
     index=j if i==center else i;sp,tp=(q,p) if i==center else (p,q)
     direct[index]=fit_registration(sp,tp,images[index],target,als[index],als[center],extra)
    cache=out/(f'{key[2]}_matches_{i}_{j}.npz')
    if not cache.exists():np.savez_compressed(cache,p=p,q=q)
    refs.append(dict(i=i,j=j,path=str(cache.relative_to(ROOT)),sha256=digest(cache)))
  result=dict(case=a['scene_id'],split='train',region=region,signature=signature,mask_pixels=int(mask.sum()),mask_ratio_to_footprint=float(mask.sum()/masks['region_footprint'].sum()),source_order=[names[i] for i in donors],correspondences=refs,pair_proposals=proposals,created_at_utc=datetime.now(timezone.utc).isoformat(),graphs={},methods={},same_pixel_control={})
  if region!='footprint':telea=cv2.inpaint(target,mask.astype(np.uint8)*255,3,cv2.INPAINT_TELEA)
  control=None;control_accept={}
  def score(method,regs):
   nonlocal control
   base=fuse(target,mask,[images[i] for i in donors],[als[i] for i in donors],[regs[i] for i in donors],agreement_threshold=255)
   evidence=dict(mask=mask,count=base['count'],provenance=base['provenance'],disagreement=base['disagreement'])
   if region!='footprint':evidence['absolute_rgb_error_sum_on_mask']=np.abs(base['image'].astype(np.int16)-ims[center].astype(np.int16)).sum(2)[mask].astype(np.uint16)
   np.savez_compressed(out/(region+'_'+method+'_evidence.npz'),**evidence);m=dict(registrations=[regs[i].diagnostics for i in donors],policies={})
   if method=='joint_homography':control=base['image'].copy()
   for policy,p in policies.items():
    accept=policy_mask(base,mask,p);v=dict(accepted_pixels=int(accept.sum()),coverage=float(accept.sum()/max(1,mask.sum())),single_view_pixels=int((accept&(base['count']==1)).sum()),multi_view_pixels=int((accept&(base['count']>=2)).sum()))
    if region!='footprint':v.update(error=error_metrics(base['image'],ims[center],accept),gradient=gradient_error(base['image'],ims[center],accept),telea_same_pixels=error_metrics(telea,ims[center],accept))
    if method=='joint_homography':control_accept[policy]=accept
    if method=='metric_free' and region!='footprint':
     common=accept&control_accept[policy];result['same_pixel_control'][policy]=dict(new=error_metrics(base['image'],ims[center],common),control=error_metrics(control,ims[center],common),new_gradient=gradient_error(base['image'],ims[center],common),control_gradient=gradient_error(control,ims[center],common))
    m['policies'][policy]=v
   result['methods'][method]=m;del base,evidence;gc.collect()
  score('tiled_pairwise',direct)
  for model in ['joint_homography','metric_free']:
   if model=='joint_homography':P,diag,active=fit_graph(len(names),center,edges,size)
   else:P,diag,active=fit_metric_graph(len(names),center,edges,size,fov)
   diag['poses']={str(k):v.tolist() for k,v in P.items()};result['graphs'][model]=diag
   score(model,audit_poses(images,als,P,diag,active,center))
  result['runtime_seconds']=time.time()-start;save_json(output,result);print('RESULT',a['scene_id'],region,{m:round(d['policies']['candidate']['coverage'],4) for m,d in result['methods'].items()},flush=True)
  del control,control_accept,images,als,target;gc.collect()
 matcher.cache.clear()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('annotation');p.add_argument('--device',default='mps');args=p.parse_args();run(Path(args.annotation).resolve(),args.device)
