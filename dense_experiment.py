"""Truth-isolated sequence experiment core, reusable with controlled fixtures."""
import hashlib,json
from pathlib import Path
import cv2
import numpy as np
from benchmark import policy_mask,error_metrics
from repair import fuse
from sequence_geometry import propose_edge,fit_graph,transform
from constrained_sequence import audit_poses
from metric_plane import fit_metric_graph,gradient_error


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def array_hash(a):
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def immutable_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if canonical(json.loads(path.read_text()))!=canonical(value):
            raise RuntimeError(f'Immutable record differs: {path}')
        return
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def masked_inputs(images,allowed,hidden,target):
    masks=[a.copy() for a in allowed];masks[target]&=~hidden
    clean=[]
    for im,mask in zip(images,masks):
        assert mask.dtype==np.bool_ and im.shape[:2]==mask.shape
        x=im.copy();x[~mask]=0;clean.append(x)
    return clean,masks


def pair_indices(n,target):
    return [(i,j) for i in range(n) for j in range(i+1,n) if j-i<=2 or target in (i,j)]


def cached_pair(path,i,j,clean,allowed,matcher,model_signature):
    signature=dict(model=model_signature,i=i,j=j,
                   image0=array_hash(clean[i]),image1=array_hash(clean[j]),
                   allowed0=array_hash(allowed[i]),allowed1=array_hash(allowed[j]))
    path=Path(path);resumed=path.exists()
    if resumed:
        with np.load(path,allow_pickle=False) as z:
            if json.loads(str(z['signature']))!=signature:
                raise RuntimeError('Cached correspondence signature changed')
            p,q=z['p'],z['q'];extra=json.loads(str(z['extra']))
    else:
        p,q,extra=matcher(i,j,clean[i],clean[j],allowed[i],allowed[j])
    p=np.asarray(p,np.float32);q=np.asarray(q,np.float32)
    if p.shape!=q.shape or p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('Matcher returned malformed correspondences')
    for points,k in [(p,i),(q,j)]:
        xy=np.rint(points).astype(int);h,w=allowed[k].shape
        if len(xy) and (np.any(xy<0) or np.any(xy[:,0]>=w) or np.any(xy[:,1]>=h) or not allowed[k][xy[:,1],xy[:,0]].all()):
            raise ValueError('Matcher returned an excluded pixel')
    if not resumed:
        path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.npz.tmp')
        with tmp.open('wb') as f:
            np.savez_compressed(f,p=p,q=q,extra=canonical(extra),signature=canonical(signature))
        tmp.replace(path)
    return p,q,extra,resumed


def spatial_edge_audit(p,q,size):
    """Quadrant holdout diagnostic only; does not fit or select production poses."""
    blocks=(p[:,0]>=size/2).astype(int)+2*(p[:,1]>=size/2).astype(int)
    rows=[]
    for block in range(4):
        test=blocks==block;train=~test
        row=dict(block=block,train=int(train.sum()),heldout=int(test.sum()))
        if train.sum()<16 or test.sum()<8:
            rows.append(dict(row,status='insufficient spatial support'));continue
        cv2.setRNGSeed(29)
        H,keep=cv2.findHomography(p[train],q[train],cv2.USAC_MAGSAC,3,maxIters=10000,confidence=.999)
        if H is None or not np.isfinite(H).all():
            rows.append(dict(row,status='no finite transform'));continue
        error=np.linalg.norm(transform(H,p[test])-q[test],axis=1)
        rows.append(dict(row,status='measured',median_px=float(np.median(error)),
                         fraction_below_5px=float((error<5).mean()),train_inliers=int(keep.sum())))
    return dict(folds=rows,interpretation='Spatial extrapolation diagnostic, not a calibrated probability or an acceptance gate')


def register_sequence(clean,allowed,target,fov,region,cache_dir,matcher,model_signature):
    """Receives sanitized RGB only. No scoring truth argument is accepted."""
    for image,mask in zip(clean,allowed):
        if image[~mask].any():
            raise ValueError('Registration requires excluded RGB already zeroed')
    edges=[];records=[];size=clean[target].shape[0]
    for i,j in pair_indices(len(clean),target):
        tag=region if target in (i,j) else 'donor_only'
        path=Path(cache_dir)/(f'{tag}_{i}_{j}.npz')
        p,q,extra,resumed=cached_pair(path,i,j,clean,allowed,matcher,model_signature)
        edge,info=propose_edge(i,j,p,q)
        if edge is not None:edges.append(edge)
        records.append(dict(i=i,j=j,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            matches=len(p),proposal=info,spatial_holdout=spatial_edge_audit(p,q,size),matcher=extra))
    methods={};graphs={}
    for method in ['loftr_homography','loftr_metric_free']:
        if method=='loftr_homography':P,diag,active=fit_graph(len(clean),target,edges,size)
        else:P,diag,active=fit_metric_graph(len(clean),target,edges,size,fov)
        diag['poses']={str(k):v.tolist() for k,v in P.items()}
        methods[method]=audit_poses(clean,allowed,P,diag,active,target);graphs[method]=diag
    return methods,graphs,records


def score_transfer(clean,allowed,hidden,target,regs,policies,truth=None):
    donors=[i for i in range(len(clean)) if i!=target]
    result=fuse(clean[target],hidden,[clean[i] for i in donors],[allowed[i] for i in donors],
                [regs[i] for i in donors],agreement_threshold=255)
    stats={}
    for name,p in policies.items():
        accepted=policy_mask(result,hidden,p)
        stats[name]=dict(accepted_pixels=int(accepted.sum()),coverage=float(accepted.sum()/max(1,hidden.sum())),
                         abstention=float(1-accepted.sum()/max(1,hidden.sum())),
                         single_view_pixels=int((accepted&(result['count']==1)).sum()),
                         multi_view_pixels=int((accepted&(result['count']>=2)).sum()))
        if truth is not None:
            stats[name].update(error=error_metrics(result['image'],truth,accepted),
                               gradient=gradient_error(result['image'],truth,accepted))
    evidence={k:result[k] for k in ['count','disagreement','provenance']}
    evidence.update(mask=hidden,prediction_on_mask=result['image'][hidden])
    if truth is not None:
        evidence['absolute_rgb_error_sum_on_mask']=np.abs(result['image'][hidden].astype(np.int16)-truth[hidden].astype(np.int16)).sum(1).astype(np.uint16)
    return stats,evidence


def same_pixel_error(new,control,policies):
    """Compare saved error vectors on identical accepted pixels, never donor agreement."""
    assert np.array_equal(new['mask'],control['mask'])
    mask=new['mask'];out={}
    for name,p in policies.items():
        common=policy_mask(new,mask,p)&policy_mask(control,mask,p);sel=common[mask]
        out[name]=dict(pixels=int(common.sum()))
        for label,evidence in [('new',new),('control',control)]:
            values=evidence['absolute_rgb_error_sum_on_mask'][sel].astype(float)/3
            out[name][label]=dict(mae_255=float(values.mean()) if len(values) else None,
                                  over_20_fraction=float((values>20).mean()) if len(values) else None)
    return out
