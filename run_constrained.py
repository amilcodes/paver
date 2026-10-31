"""Replay fixed sequence matches with reduced-degree geometry and stability gates."""
import argparse,json,gc,time
from datetime import datetime,timezone
import cv2
import numpy as np
from PIL import Image
from benchmark import ROOT,digest,masks,save_json,policy_mask,error_metrics
from repair import nadir,polygon_mask,fuse
from sequence_geometry import propose_edge
from constrained_sequence import fit_constrained_graph,audit_poses,spatial_deletions,stability_support


def run(args):
 root=ROOT/'sequence_v4';config=json.loads((ROOT/'benchmark_v2/protocol.json').read_text())
 base_signature={n:digest(ROOT/n) for n in ['run_constrained.py','constrained_sequence.py','sequence_geometry.py','matching_v2.py','repair.py','benchmark.py']}
 base_signature.update(protocol=digest(root/'protocol.json'),baseline_protocol=digest(ROOT/'benchmark_v2/protocol.json'))
 for source_experiment in args.source or ['sequence_v1','sequence_v3']:
  source_root=ROOT/source_experiment;names=json.loads((source_root/'protocol.json').read_text())['frames'];center=names.index('target')
  for case in config['cases']:
   if args.case and case not in args.case:continue
   upstream=source_root/'results'/case
   if not upstream.exists():continue
   long=source_experiment=='sequence_v3';data=ROOT/'dataset_v1' if long else ROOT/'benchmark_v2/data'/case
   manifest_path=data/'windows'/(case+'.json') if long else data/'manifest.json';manifest=json.loads(manifest_path.read_text());records={x['name']:x for x in manifest['images']}
   paths=[data/'images'/(records[n]['id']+'.jpg') if long else data/(n+'.jpg') for n in names]
   for n,p in zip(names,paths):assert digest(p)==records[n]['sample_sha256']
   assert len({records[n]['id'] for n in names})==len(names)
   images=[nadir(np.array(Image.open(p)),config['size'],config['fov']) for p in paths]
   vehicle,allowed_by_name,_=masks(config,config['cases'][case]);allowed=[allowed_by_name.get(n,polygon_mask(images[center].shape,config['cases'][case]['ground'])&~vehicle) for n in names]
   donor_indices=[i for i in range(len(names)) if i!=center];folder=root/'results'/source_experiment/case;folder.mkdir(parents=True,exist_ok=True)
   for region in ['footprint','hidden_front','hidden_rear','local_front','local_rear']:
    if args.region and region not in args.region:continue
    start=time.time();old_path=upstream/(region+'.json');old=json.loads(old_path.read_text());sig=dict(base_signature,upstream=digest(old_path),manifest=digest(manifest_path))
    dest=folder/(region+'.json')
    if dest.exists():
     assert json.loads(dest.read_text())['signature']==sig,'Immutable experiment changed';print('RESUME',source_experiment,case,region,flush=True);continue
    for name,value in old['signature'].items():
     path=source_root/'protocol.json' if name=='sequence_protocol' else ROOT/'benchmark_v2/protocol.json' if name=='baseline_protocol' else manifest_path if name=='manifest' else ROOT/name
     assert digest(path)==value,(case,region,name)
    mask=np.load(ROOT/'benchmark_v2/results'/case/'disk_lightglue'/(region+'_evidence.npz'))['mask'];target=images[center].copy();target[mask|vehicle]=0
    ims=images.copy();ims[center]=target;als=allowed.copy();als[center]=allowed[center]&~mask;edges=[]
    for ref in old['correspondences']:
     path=source_root/ref['path'];assert digest(path)==ref['sha256'];z=np.load(path)
     edge,info=propose_edge(ref['i'],ref['j'],z['p'],z['q'])
     if edge is not None:edges.append(edge)
    old_poses={int(k):np.array(v) for k,v in old['graph']['poses'].items()};active=[e for e in edges if e['i'] in old_poses and e['j'] in old_poses]
    old_summary=dict(old['graph'],model='homography');control=audit_poses(ims,als,old_poses,old_summary,active,center)
    result=dict(case=case,region=region,source_experiment=source_experiment,signature=sig,mask_pixels=int(mask.sum()),source_order=[names[i] for i in donor_indices],created_at_utc=datetime.now(timezone.utc).isoformat(),graphs={'homography_converged':old_summary},methods={},same_pixel_control={})
    if region!='footprint':telea=cv2.inpaint(target,mask.astype(np.uint8)*255,3,cv2.INPAINT_TELEA)
    control_pred=None;control_accept={}
    def score(method,regs):
     nonlocal control_pred
     base=fuse(target,mask,[ims[i] for i in donor_indices],[als[i] for i in donor_indices],[regs[i] for i in donor_indices],agreement_threshold=255)
     evidence=dict(mask=mask,provenance=base['provenance'],count=base['count'],disagreement=base['disagreement'])
     if region!='footprint':evidence['absolute_rgb_error_sum_on_mask']=np.abs(base['image'].astype(np.int16)-images[center].astype(np.int16)).sum(2)[mask].astype(np.uint16)
     np.savez_compressed(folder/(region+'_'+method+'_evidence.npz'),**evidence)
     m=dict(registrations=[regs[i].diagnostics for i in donor_indices],policies={})
     if method=='homography_converged':control_pred=base['image'].copy()
     for name,p in config['policies'].items():
      accept=policy_mask(base,mask,p);v=dict(accepted_pixels=int(accept.sum()),coverage=float(accept.sum()/max(1,mask.sum())),single_view_pixels=int((accept&(base['count']==1)).sum()),multi_view_pixels=int((accept&(base['count']>=2)).sum()))
      if region!='footprint':v.update(error=error_metrics(base['image'],images[center],accept),telea_same_pixels=error_metrics(telea,images[center],accept))
      if method=='homography_converged':
       control_accept[name]=accept
       expected=old['methods']['sequence_graph']['policies'][name]['accepted_pixels'] if old['graph'].get('optimizer_success') else 0
       assert v['accepted_pixels']==expected,(source_experiment,case,region,name,v['accepted_pixels'],expected)
      elif region!='footprint':
       common=accept&control_accept[name];result['same_pixel_control'].setdefault(method,{})[name]={'new':error_metrics(base['image'],images[center],common),'control':error_metrics(control_pred,images[center],common)}
      m['policies'][name]=v
     result['methods'][method]=m
     del base,evidence;gc.collect()
    score('homography_converged',control)
    for model in ['affine','similarity']:
     poses,diag,active=fit_constrained_graph(len(names),center,edges,config['size'],model)
     diag['poses']={str(k):v.tolist() for k,v in poses.items()};regs=audit_poses(ims,als,poses,diag,active,center);score(model,regs)
     if diag['optimizer_success'] and any(r.homography is not None for r in regs.values()):
      replicates,detail=spatial_deletions(len(names),center,active,config['size'],model,poses);stable,sensitivity=stability_support(regs,poses,replicates,config['size']);diag['spatial_deletion']=detail
      del sensitivity
     else:stable=regs;diag['spatial_deletion']={'reason':'base geometry already abstains'}
     score(model+'_stable',stable);result['graphs'][model]=diag
    result['runtime_seconds']=time.time()-start;save_json(dest,result)
    print('RESULT',source_experiment,case,region,{m:round(d['policies']['candidate']['coverage'],4) for m,d in result['methods'].items()},flush=True)
    del control_pred,control_accept,ims,target,als;gc.collect()
   del images;gc.collect()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',action='append',choices=['sequence_v1','sequence_v3']);p.add_argument('--case',action='append');p.add_argument('--region',action='append');run(p.parse_args())
