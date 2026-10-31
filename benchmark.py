"""Reproducible masked matching benchmark; no test target pixels used by matching.

Run with --method sift or --method disk_lightglue. Results are resumable only
when protocol, source checksums, and implementation hashes still match.
"""
import argparse,gc,hashlib,json,os,time
from pathlib import Path
from datetime import datetime,timezone
import cv2
import numpy as np
from PIL import Image
from repair import nadir,polygon_mask,fuse,score,paste_nadir
from matching_v2 import LearnedMatcher,SIFTMatcher,register_learned
ROOT=Path(__file__).resolve().parent

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save_json(p,value):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(p)
def save_image(p,a,size=1000):
    im=Image.fromarray(a)
    if im.width>size:im.thumbnail((size,size))
    im.save(p,quality=92,subsampling=0)
def masks(config,case):
    shape=(config['size'],config['size'])
    vehicle=polygon_mask(shape,case['vehicle'])
    ground={name:polygon_mask(shape,case.get('ground_by_frame',{}).get(name,case['ground']))
            for name in ['target']+config['source_names']}
    allowed={name:g&~vehicle for name,g in ground.items()}
    holes={'footprint':vehicle}
    for name,dy in [('hidden_front',-case['shift']),('hidden_rear',case['shift'])]:
        shifted=[[[x,y+dy] for x,y in p] for p in case['vehicle']]
        holes[name]=polygon_mask(shape,shifted)&allowed['target']
    return vehicle,allowed,holes

def policy_mask(base,mask,policy):
    return mask&(base['count']>=policy['min_views'])&(base['disagreement']<=policy['max_disagreement'])
def error_metrics(pred,truth,mask):
    s=score(pred,truth,mask)
    if s['pixels']:
        error=np.abs(pred[mask].astype(np.float32)-truth[mask]).mean(1)
        s.update(p90_rgb=float(np.percentile(error,90)),over_20_fraction=float((error>20).mean()))
    else:s.update(p90_rgb=None,over_20_fraction=None)
    return s

def run(args):
    root=ROOT/'benchmark_v2';protocol=root/'protocol.json';config=json.loads(protocol.read_text())
    signature={n:digest(ROOT/n) for n in ['repair.py','matching_v2.py','benchmark.py']}
    signature['protocol']=digest(protocol)
    matcher=LearnedMatcher(args.device) if args.method=='disk_lightglue' else SIFTMatcher()
    for key,case in config['cases'].items():
        if args.case and key not in args.case:continue
        data=root/'data'/key;out=root/'results'/key/args.method;out.mkdir(parents=True,exist_ok=True)
        manifest=json.loads((data/'manifest.json').read_text())
        ids=[x['id'] for x in manifest['images']]
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate donor/target IDs')
        for row in manifest['images']:
            if digest(data/(row['name']+'.jpg'))!=row['sample_sha256']:raise ValueError('Source checksum mismatch')
        run_signature=dict(signature,manifest=digest(data/'manifest.json'))
        if (out/'metrics.json').exists() and not args.force:
            old=json.loads((out/'metrics.json').read_text())
            if old['signature']==run_signature:
                print('RESUME',key,args.method,flush=True);continue
            raise RuntimeError(f'{out} signature changed; use --force to recompute')
        print('START',key,args.method,flush=True);start=time.time()
        images={n:nadir(np.array(Image.open(data/(n+'.jpg'))),config['size'],config['fov'])
                for n in ['target']+config['source_names']}
        vehicle,allowed,holes=masks(config,case)
        target=images['target'];sources=[images[n] for n in config['source_names']]
        source_allowed=[allowed[n] for n in config['source_names']]
        save_image(out/'original.jpg',target)
        annotation=target.copy();annotation[vehicle]=annotation[vehicle]//2+[90,25,25]
        annotation[allowed['target']]=(annotation[allowed['target']].astype(np.float32)*.7+[10,55,25]).clip(0,255)
        save_image(out/'annotation.jpg',annotation.astype(np.uint8))
        result=dict(case=key,method=args.method,device=getattr(matcher,'device','cpu'),signature=run_signature,
                    config=case,source_names=config['source_names'],capture=manifest['images'][0],regions={},
                    created_at_utc=datetime.now(timezone.utc).isoformat())
        for region,mask in holes.items():
            # All matching and photometric gates see a target with the hole zeroed.
            input_image=target.copy();input_image[vehicle|mask]=0
            target_allowed=allowed['target']&~mask
            regs=[register_learned(matcher,src,input_image,sa,target_allowed)
                  for src,sa in zip(sources,source_allowed)]
            base=fuse(input_image,mask,sources,source_allowed,regs,agreement_threshold=255,min_views=1)
            metrics=dict(pixels=int(mask.sum()),relative_to_footprint=float(mask.sum()/vehicle.sum()),
                         registrations=[r.diagnostics for r in regs],policies={})
            save_image(out/(region+'_input.jpg'),input_image)
            raw=target.copy();raw[mask]=base['image'][mask]
            save_image(out/(region+'_transfers.jpg'),raw)
            # Store full-resolution evidence, not only the downsampled UI previews.
            np.savez_compressed(out/(region+'_evidence.npz'),mask=mask,provenance=base['provenance'],
                                count=base['count'],disagreement=base['disagreement'])
            if region!='footprint':
                telea=cv2.inpaint(input_image,mask.astype(np.uint8)*255,3,cv2.INPAINT_TELEA)
                metrics['telea_full']=error_metrics(telea,target,mask)
                save_image(out/(region+'_telea.jpg'),telea)
            for name,p in config['policies'].items():
                accepted=policy_mask(base,mask,p)
                m=dict(accepted_pixels=int(accepted.sum()),coverage=float(accepted.sum()/max(1,mask.sum())),
                       single_view_pixels=int((accepted&(base['count']==1)).sum()),
                       multi_view_pixels=int((accepted&(base['count']>=2)).sum()))
                if region!='footprint':
                    m['accepted_error']=error_metrics(base['image'],target,accepted)
                    m['telea_same_pixels']=error_metrics(telea,target,accepted)
                    m['covered_with_error_le_20_fraction']=float((np.abs(base['image'].astype(np.float32)-target).mean(2)[accepted]<=20).sum()/max(1,mask.sum()))
                    m['full_mask_error_with_black_abstentions']=error_metrics(base['image']*accepted[:,:,None],target,mask)
                metrics['policies'][name]=m
                preview=target.copy();preview[accepted]=base['image'][accepted]
                # Unresolved pixels remain original, with a separate conspicuous evidence overlay.
                save_image(out/(region+'_'+name+'.jpg'),preview)
                evidence=target.copy();evidence[mask]=[222,83,109]
                evidence[accepted&(base['count']==1)]=[244,178,76]
                evidence[accepted&(base['count']>=2)]=[43,199,174]
                save_image(out/(region+'_'+name+'_support.jpg'),evidence)
                if region=='footprint' and args.method=='disk_lightglue':
                    # Modest 360 preview resolution; original panorama is retained on disk.
                    pano=np.array(Image.open(data/'target.jpg').resize((2048,1024)))
                    pano,sphere_mask=paste_nadir(pano,preview,accepted,config['fov'])
                    Image.fromarray(pano).save(out/('sphere_'+name+'.jpg'),quality=94,subsampling=0)
            if region=='footprint':
                # Same matcher/gates with only immediate neighbors: isolates the extra-donor effect.
                immediate=fuse(input_image,mask,sources[:2],source_allowed[:2],regs[:2],agreement_threshold=255)
                metrics['two_neighbor_ablation']={name:float(policy_mask(immediate,mask,p).sum()/max(1,mask.sum()))
                                                  for name,p in config['policies'].items()}
                del immediate
            result['regions'][region]=metrics
            print('REGION',key,region,'regs',sum(r.homography is not None for r in regs),'coverage',
                  {k:round(v['coverage'],3) for k,v in metrics['policies'].items()},flush=True)
            del base,raw,regs
            gc.collect()
        result['runtime_seconds']=time.time()-start
        save_json(out/'metrics.json',result)
        matcher.cache.clear();del images,sources,target
        gc.collect()
        if hasattr(matcher,'torch') and matcher.device=='mps':matcher.torch.mps.empty_cache()
        print('DONE',key,round(result['runtime_seconds'],1),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['sift','disk_lightglue'],required=True)
    p.add_argument('--device',default='auto');p.add_argument('--case',action='append');p.add_argument('--force',action='store_true')
    run(p.parse_args())
