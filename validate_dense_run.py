"""Independent scoring/provenance audit of saved v6 outputs; never runs a matcher.

No result is a pending experiment, not a passing model evaluation. Real footprint
pixels never receive truth metrics. This audit verifies stored evidence and
selected donor samples, not unrecorded model behavior or calibrated correctness.
"""
import argparse,hashlib,json,math
from pathlib import Path
from datetime import datetime,timezone
import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
METHODS={'loftr_homography','loftr_metric_free'}


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def checked_ref(root,record):
    root=Path(root).resolve();path=(root/record['path']).resolve()
    require(path.is_relative_to(root),'artifact path escapes research root')
    require(path.is_file() and sha(path)==record['sha256'],f'artifact hash mismatch: {path}')
    return path


def array_digest(a):
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def compare(actual,expected,label='metrics'):
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and set(actual)==set(expected),label+' fields differ')
        for key,value in expected.items():compare(actual[key],value,label+'.'+key)
    elif expected is None:require(actual is None,label+' must be null')
    elif isinstance(expected,int):require(type(actual) is int and actual==expected,label+' integer differs')
    else:
        require(type(actual) in (int,float) and math.isfinite(actual) and math.isclose(actual,expected,rel_tol=1e-6,abs_tol=2e-5),label+' numeric value differs')


def accepted(evidence,mask,policy):
    return mask&(evidence['count']>=policy['min_views'])&(evidence['disagreement']<=policy['max_disagreement'])


def rgb_metrics(pred,truth,take):
    n=int(take.sum())
    if not n:return dict(pixels=0,mae_255=None,psnr_db=None,p90_rgb=None,over_20_fraction=None)
    diff=pred[take].astype(np.float64)-truth[take].astype(np.float64)
    per_pixel=np.abs(diff.astype(np.float32)).mean(axis=1)
    return dict(pixels=n,mae_255=float(np.abs(diff).mean()),
                psnr_db=float(10*np.log10(255**2/max(float(np.square(diff).mean()),1e-12))),
                p90_rgb=float(np.percentile(per_pixel,90)),over_20_fraction=float((per_pixel>20).mean()))


def gradient_metrics(pred,truth,take):
    # Independently expand the Sobel stencil at eligible pixels, rather than
    # calling the production metric_plane.gradient_error/Sobel implementation.
    safe=cv2.erode(take.astype(np.uint8),np.ones((7,7),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    y,x=np.nonzero(safe)
    if not len(x):return dict(pixels=0,gradient_mae=None,edge_pixels=0,orientation_error=None)
    def gradients(image):
        g=cv2.cvtColor(image.astype(np.float32),cv2.COLOR_RGB2GRAY)
        dx=(g[y-1,x+1]-g[y-1,x-1]+2*(g[y,x+1]-g[y,x-1])+g[y+1,x+1]-g[y+1,x-1])/8
        dy=(g[y+1,x-1]-g[y-1,x-1]+2*(g[y+1,x]-g[y-1,x])+g[y+1,x+1]-g[y-1,x+1])/8
        return np.stack([dx,dy],axis=1)
    p,t=gradients(pred),gradients(truth);pn=np.linalg.norm(p,axis=1);tn=np.linalg.norm(t,axis=1);edge=tn>=4
    orientation=np.clip(1-np.sum(p*t,axis=1)/np.maximum(pn*tn,1e-6),0,2)
    return dict(pixels=len(x),gradient_mae=float(np.abs(p-t).mean()),edge_pixels=int(edge.sum()),
                orientation_error=float(orientation[edge].mean()) if edge.any() else None)


def verify_evidence(evidence,reported,mask,policies,donor_count,*,footprint,truth=None,graph=None,registrations=None):
    require(mask.ndim==2 and mask.dtype==bool,'expected boolean frozen mask')
    keys={'mask','count','provenance','disagreement','prediction_on_mask'}
    if footprint:require(truth is None,'real footprint cannot have evaluation truth')
    else:
        keys.add('absolute_rgb_error_sum_on_mask')
        require(truth is not None and truth.shape==(*mask.shape,3) and truth.dtype==np.uint8,'missing original hidden truth')
    require(set(evidence)==keys,'evidence schema differs; footprint truth claims are forbidden')
    require(evidence['mask'].dtype==bool and np.array_equal(evidence['mask'],mask),'frozen mask differs')
    for name,dtype in [('count',np.uint8),('provenance',np.int16),('disagreement',np.float32)]:
        require(evidence[name].shape==mask.shape and evidence[name].dtype==dtype,name+' shape/dtype differs')
    count=evidence['count'];prov=evidence['provenance'];dis=evidence['disagreement'];n=int(mask.sum())
    require(not count[~mask].any() and np.all(count<=donor_count),'count outside mask or exceeds donor population')
    require(np.isfinite(dis).all() and (dis>=0).all() and (dis<=255).all(),'invalid disagreement')
    require(not dis[count<=1].any(),'single/no-view disagreement must be zero')
    require(np.all(prov[count==0]==-1) and np.all((prov[count>0]>=0)&(prov[count>0]<donor_count)),'invalid donor provenance')
    saved=evidence['prediction_on_mask']
    require(saved.dtype==np.uint8 and saved.shape==(n,3),'saved prediction shape/dtype differs')
    require(not saved[count[mask]==0].any(),'abstained target pixels must stay sanitized')
    if graph is not None and graph.get('optimizer_success') is not True:
        require(not count.any(),'nonconverged graph produced transfers')
    if registrations is not None:
        require(len(registrations)==donor_count,'registration population differs')
        usable=[i for i,r in enumerate(registrations) if r.get('accepted') is True]
        require(np.isin(prov[count>0],usable).all() and np.all(count<=len(usable)),'rejected donor supplied evidence')
    prediction=np.zeros((*mask.shape,3),np.uint8);prediction[mask]=saved
    errors=None
    if not footprint:
        errors=np.abs(saved.astype(np.int16)-truth[mask].astype(np.int16)).sum(axis=1).astype(np.uint16)
        stored=evidence['absolute_rgb_error_sum_on_mask']
        require(stored.dtype==np.uint16 and stored.shape==(n,) and np.array_equal(errors,stored),'stored RGB error does not match original truth')
    recalculated={};joint={}
    for name,policy in policies.items():
        take=accepted(evidence,mask,policy);num=int(take.sum())
        recalculated[name]=dict(accepted_pixels=num,coverage=num/max(1,n),abstention=1-num/max(1,n),
                               single_view_pixels=int((take&(count==1)).sum()),multi_view_pixels=int((take&(count>=2)).sum()))
        joint[name]=dict(mask_pixels=n,accepted_pixels=num,abstained_pixels=n-num)
        if not footprint:
            recalculated[name].update(error=rgb_metrics(prediction,truth,take),gradient=gradient_metrics(prediction,truth,take))
            good=int((errors[take[mask]]<=60).sum())
            joint[name].update(accepted_rgb_mae_le_20_pixels=good,accepted_rgb_mae_gt_20_pixels=num-good,
                               accepted_rgb_mae_le_20_fraction_of_mask=good/max(1,n))
    compare(reported,recalculated)
    return dict(policies=recalculated,coverage_error_abstention=joint)


def verify_selected_donors(evidence,images,allowed,registrations):
    """Verify selected RGB and source visibility; no fit, matching or hidden target."""
    mask=evidence['mask'];h,w=mask.shape;prov=evidence['provenance'];count=evidence['count']
    expected=np.zeros((*mask.shape,3),np.uint8);visible_count=np.zeros(mask.shape,np.uint8)
    require(len(images)==len(allowed)==len(registrations),'donor arrays differ')
    for i,(image,ok,reg) in enumerate(zip(images,allowed,registrations)):
        if reg.get('accepted') is not True:continue
        H=np.asarray(reg['homography'],np.float64)
        require(H.shape==(3,3) and np.isfinite(H).all(),'invalid accepted pose')
        clean=image.copy();clean[~ok]=0
        safe=cv2.erode(ok.astype(np.uint8),np.ones((5,5),np.uint8))
        visible=cv2.warpPerspective(safe,H,(w,h),flags=cv2.INTER_NEAREST)>0
        visible_count+=visible.astype(np.uint8)
        chosen=mask&(prov==i)
        require(not (chosen&~visible).any(),'selected donor is outside allowed source ground')
        warped=cv2.warpPerspective(clean,H,(w,h),flags=cv2.INTER_LINEAR);expected[chosen]=warped[chosen]
    require(np.all(count<=visible_count),'count exceeds eligible warped source visibility')
    require(np.array_equal(expected[mask],evidence['prediction_on_mask']),'saved RGB is not from the selected donor')
    return dict(selected_donor_pixels=int((count>0).sum()),support_hulls_replayed=False)


def verify_common(evidence,control,policies,reported):
    mask=evidence['mask'];require(np.array_equal(control['mask'],mask),'control mask differs')
    require('absolute_rgb_error_sum_on_mask' in evidence and 'absolute_rgb_error_sum_on_mask' in control,'same-pixel comparison requires hidden truth errors')
    expected={}
    for name,p in policies.items():
        use=accepted(evidence,mask,p)&accepted(control,mask,p);select=use[mask]
        expected[name]=dict(pixels=int(use.sum()))
        for label,source in [('new',evidence),('control',control)]:
            values=source['absolute_rgb_error_sum_on_mask'][select].astype(np.float64)/3
            expected[name][label]=dict(mae_255=float(values.mean()) if len(values) else None,
                                       over_20_fraction=float((values>20).mean()) if len(values) else None)
    compare(reported,expected,'same_pixel_control')


def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def verify_certificate(root,ref,plan):
    c=json.loads(checked_ref(root,ref).read_text())
    require(c.get('passed') is True and c.get('actual_model_executed') is True and c.get('excluded_rgb_correspondence_invariance') is True,'no passed actual-model certificate')
    signature=dict(model=plan['model'],validator_sha256=plan['signature']['validate_dense_model.py'],
                   thresholds=dict(minimum_matches=32,median_error_px=1.5,p90_error_px=3.0),device='mps')
    require(c.get('signature')==signature,'certificate signature differs')
    require(type(c.get('translation_matches')) is int and c['translation_matches']>=32,'too few certificate matches')
    for key,maximum in [('translation_median_error_px',1.5),('translation_p90_error_px',3.)]:
        value=c.get(key);require(type(value) in (int,float) and math.isfinite(value) and 0<=value<maximum,'invalid certificate error')


def verify_pairs(root,record,images,allowed,mask,target,model):
    clean=[];masks=[a.copy() for a in allowed];masks[target]&=~mask
    for image,ok in zip(images,masks):
        value=image.copy();value[~ok]=0;clean.append(value)
    expected={(i,j) for i in range(len(images)) for j in range(i+1,len(images)) if j-i<=2 or target in (i,j)}
    refs=record['correspondences'];pairs=[(r['i'],r['j']) for r in refs]
    require(len(pairs)==len(expected) and set(pairs)==expected,'missing or duplicate pair records')
    for ref in refs:
        i,j=ref['i'],ref['j'];z=load_npz(checked_ref(root,ref))
        signature=dict(model=model,i=i,j=j,image0=array_digest(clean[i]),image1=array_digest(clean[j]),
                       allowed0=array_digest(masks[i]),allowed1=array_digest(masks[j]))
        require(json.loads(str(z['signature']))==signature,'pair cache inputs do not match sanitized frozen images')
        p,q=z['p'],z['q']
        require(p.shape==q.shape and p.ndim==2 and p.shape[1]==2 and len(p)==ref['matches'],'malformed pair coordinates/count')
        require(json.loads(str(z['extra']))==ref['matcher'],'pair matcher metadata differs')
        for points,k in [(p,i),(q,j)]:
            require(np.isfinite(points).all(),'nonfinite correspondence')
            xy=np.rint(points).astype(int);h,w=masks[k].shape
            require(np.all(xy>=0) and np.all(xy[:,0]<w) and np.all(xy[:,1]<h),'out-of-image correspondence')
            distance=cv2.distanceTransform(masks[k].astype(np.uint8),cv2.DIST_L2,5)
            require(np.all(distance[xy[:,1],xy[:,0]]>32),'correspondence violates excluded-pixel margin')
    return len(refs)


def load_scene(root,case):
    # Projection/mask definitions are frozen experiment inputs, not learned code.
    from repair import nadir
    from benchmark import masks
    for ref in case['inputs']:checked_ref(root,ref)
    images=[nadir(np.asarray(Image.open(checked_ref(root,r)).convert('RGB')),**case['projection']) for r in case['frames']]
    names=[r['name'] for r in case['frames']];require(len(set(names))==len(names),'duplicate frame names')
    require(len({r['id'] for r in case['frames']})==len(names),'duplicate frame IDs')
    if case['split']=='development':
        cfg=json.loads((root/'benchmark_v2/protocol.json').read_text());_,a,_=masks(cfg,cfg['cases'][case['case']]);allowed=[a[n] for n in names]
    else:
        require(case['split']=='train','validation/test data forbidden for this experiment')
        z=load_npz(root/'benchmark_v3/prepared'/case['case']/'masks.npz');allowed=[z[n+'_allowed'] for n in names]
    return images,allowed,names.index('target')


def validate_record(root,plan,plan_path,case,job,path,scene):
    record=json.loads(path.read_text());images,allowed,target=scene;donors=[i for i in range(len(images)) if i!=target]
    require(record['case']==case['case'] and record['region']==job['region'] and record['split']==case['split'],'result identity differs')
    require(record['signature']['plan_sha256']==sha(plan_path),'result plan hash differs')
    verify_certificate(root,record['signature']['certificate'],plan)
    require(record.get('promotion')=='none','experimental result cannot claim promotion')
    require(record['source_order']==[case['frames'][i]['name'] for i in donors],'donor order differs')
    mask=load_npz(checked_ref(root,job['mask']))[job['mask_key']];footprint=job['region']=='footprint'
    require(int(mask.sum())==record['mask_pixels'],'mask denominator differs')
    require(not (mask&allowed[target]).any() if footprint else not (mask&~allowed[target]).any(),'ground/footprint mask eligibility differs')
    require(set(record['methods'])==METHODS and set(record['graphs'])==METHODS,'method population differs')
    pair_count=verify_pairs(root,record,images,allowed,mask,target,plan['model'])
    controls={};require(set(record['controls'])==set(job['controls']),'control population differs')
    for name,refs in job['controls'].items():
        require(record['controls'][name]['references']==refs,'control reference differs')
        old=json.loads(checked_ref(root,refs['record']).read_text())['methods'][name]['policies']
        copied={k:dict(v,abstention=1-v['coverage']) for k,v in old.items()}
        require(record['controls'][name]['policies']==copied,'cached control report differs')
        control=load_npz(checked_ref(root,refs['evidence']));require(np.array_equal(control['mask'],mask),'control mask differs')
        for name_policy,p in plan['policies'].items():require(int(accepted(control,mask,p).sum())==old[name_policy]['accepted_pixels'],'cached control count differs')
        controls[name]=control
    require(not record['same_pixel_control'] if footprint else set(record['same_pixel_control'])==METHODS,'footprint accuracy claim or missing common-pixel comparisons')
    methods={}
    for method,data in record['methods'].items():
        evidence=load_npz(checked_ref(root,data['evidence']));regs=data['registrations'];graph=record['graphs'][method]
        methods[method]=verify_evidence(evidence,data['policies'],mask,plan['policies'],len(donors),footprint=footprint,
                                        truth=None if footprint else images[target],graph=graph,registrations=regs)
        for i,reg in zip(donors,regs):
            if reg.get('accepted') is True:require(reg['homography']==graph['poses'].get(str(i)),'accepted pose differs from graph')
        methods[method]['provenance']=verify_selected_donors(evidence,[images[i] for i in donors],[allowed[i] for i in donors],regs)
        if not footprint:
            require(set(record['same_pixel_control'][method])==set(controls),'common-pixel control population differs')
            for name,control in controls.items():verify_common(evidence,control,plan['policies'],record['same_pixel_control'][method][name])
    return dict(case=case['case'],region=job['region'],record_sha256=sha(path),pair_caches_verified=pair_count,methods=methods)


def audit(root=ROOT,limit=1,case_names=None,regions=None):
    plan_path=root/'sequence_v6/execution_plan.json';plan=json.loads(plan_path.read_text())
    for name,expected in plan['signature'].items():checked_ref(root,dict(path=name,sha256=expected))
    for ref in plan['protocols']:checked_ref(root,ref)
    policy=json.loads((root/'benchmark_v2/protocol.json').read_text())['policies'];require(plan['policies']==policy,'frozen policy differs')
    prep=json.loads((root/'sequence_v6/model_preparation.json').read_text())
    require(plan['model']['checkpoint_sha256']==prep['outdoor_file']['lfs']['sha256'],'checkpoint pin differs')
    checked=[];missing=[];available=0
    for case in plan['cases']:
        if case_names and case['case'] not in case_names:continue
        scene=None
        for job in case['jobs']:
            if regions and job['region'] not in regions:continue
            path=root/'sequence_v6/results'/case['case']/(job['region']+'.json')
            if not path.exists():missing.append(dict(case=case['case'],region=job['region']));continue
            available+=1
            if len(checked)>=limit:continue
            if scene is None:scene=load_scene(root,case)
            checked.append(validate_record(root,plan,plan_path,case,job,path,scene))
    require(bool(checked) or bool(missing),'selection contains no planned regions')
    return dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                status='pending_no_results' if not available else 'checked_subset' if missing or len(checked)<available else 'all_selected_results_checked',
                passed=None if not checked else True,checked_regions=len(checked),available_regions=available,missing_results=missing,
                matcher_executed=False,new_recovery_runs=0,plan_sha256=sha(plan_path),validator_sha256=sha(Path(__file__)),results=checked,
                limitations=['Saved artifact consistency does not prove unrecorded model behavior or training independence.',
                             'Checks selected donor bytes and allowed visibility, but does not replay convex support hulls or fit geometry.',
                             'RGB error <=20 is a measured proxy criterion, not a calibrated geometric correctness probability.',
                             'Real car footprint has no ground truth; no accuracy is assigned to it.'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',action='append');p.add_argument('--region',action='append');p.add_argument('--limit',type=int,default=1);p.add_argument('--output');args=p.parse_args()
    if not 1<=args.limit<=15:p.error('--limit must be between1 and15 regions')
    result=audit(limit=args.limit,case_names=args.case,regions=args.region)
    if args.output:
        out=Path(args.output)
        require(not out.exists(),'refusing to overwrite an earlier validation record');out.parent.mkdir(parents=True,exist_ok=True)
        temp=out.with_suffix(out.suffix+'.tmp');temp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temp.replace(out)
    print(json.dumps(result,indent=2,allow_nan=False))
    raise SystemExit(2 if result['status']=='pending_no_results' else 0)
