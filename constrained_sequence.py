"""Reduced-degree sequence geometry and spatial deletion sensitivity.

Affine/similarity transforms are approximations, not calibrated 3D poses.
Deletion sensitivity is a stability diagnostic, not a confidence probability.
All fits use training correspondences only. Hidden RGB is never an input.
"""
import cv2
import numpy as np
from sequence_geometry import initialize,transform
from repair import Registration


def fit_constrained_graph(n,target,edges,size,model='affine',max_iterations=80):
    if model not in ['affine','similarity']:raise ValueError('Unknown model')
    poses=initialize(n,target,edges)
    active=[e for e in edges if e['i'] in poses and e['j'] in poses]
    info=dict(model=model,connected_frames=len(poses),edges=len(active),optimizer_success=False)
    if len(poses)<2:return poses,dict(info,reason='target disconnected'),active
    nodes=[i for i in sorted(poses) if i!=target];width=6 if model=='affine' else 4;lookup={i:k*width for k,i in enumerate(nodes)}
    scale=size/2;center=(size-1)/2
    N=np.array([[1/scale,0,-center/scale],[0,1/scale,-center/scale],[0,0,1.]])
    inv=np.linalg.inv(N);rows=sum(len(e['p'])*2 for e in active);A=np.zeros((rows,len(nodes)*width));b=np.zeros(rows);offset=0
    def design(p):
        x,y=((p-center)/scale).T;m=len(p);D=np.zeros((2*m,width))
        if model=='affine':
            D[::2,0]=x;D[::2,1]=y;D[::2,2]=1;D[1::2,3]=x;D[1::2,4]=y;D[1::2,5]=1
        else:
            D[::2,0]=x;D[::2,1]=-y;D[::2,2]=1;D[1::2,0]=y;D[1::2,1]=x;D[1::2,3]=1
        return D*scale
    for e in active:
        end=offset+2*len(e['p'])
        for i,p,sign in [(e['i'],e['p'],1),(e['j'],e['q'],-1)]:
            if i==target:b[offset:end]-=sign*(p-center).ravel()
            else:A[offset:end,lookup[i]:lookup[i]+width]+=sign*design(p)
        offset=end
    x,_,rank,singular=np.linalg.lstsq(A,b,rcond=None)
    info.update(rank=int(rank),parameters=A.shape[1],condition_number=float(singular[0]/max(1e-15,singular[-1])))
    if rank<A.shape[1] or info['condition_number']>1e8:return poses,dict(info,reason='rank-deficient or ill-conditioned graph'),active
    success=False;step=None
    for it in range(max_iterations):
        residual=(A@x-b).reshape(-1,2);norm=np.linalg.norm(residual,axis=1)
        weights=np.repeat(np.sqrt(np.minimum(1,2/np.maximum(norm,1e-9))),2)
        updated,_,rank,_=np.linalg.lstsq(A*weights[:,None],b*weights,rcond=None)
        if rank<A.shape[1] or not np.isfinite(updated).all():break
        step=float(np.max(np.linalg.norm((A@(updated-x)).reshape(-1,2),axis=1)));x=updated
        if step<.001:success=True;break
    result={target:np.eye(3)}
    for i,k in lookup.items():
        p=x[k:k+width]
        H=np.r_[p,[0,0,1]].reshape(3,3) if model=='affine' else np.array([[p[0],-p[1],p[2]],[p[1],p[0],p[3]],[0,0,1]])
        result[i]=inv@H@N
    info.update(optimizer_success=success,iterations=it+1 if max_iterations else 0,maximum_prediction_step_px=step,
                median_training_residual_px=float(np.median(np.linalg.norm((A@x-b).reshape(-1,2),axis=1))),
                reason='converged robust linear graph' if success else 'IRLS did not converge')
    return result,info,active


def audit_poses(images,allowed,poses,summary,active,target):
    """Same v1 audit/support limits, plus mandatory optimizer convergence."""
    size=images[target].shape[0];empty=np.zeros((size,size),bool);regs={}
    for i,source in enumerate(images):
        if i==target:continue
        info=dict(method=summary.get('model','homography'),accepted=False,inliers=0,median_reprojection_px=0.)
        def reject(reason):return Registration(None,empty,dict(info,reason=reason))
        if not summary.get('optimizer_success',False):regs[i]=reject('optimizer did not converge or target disconnected');continue
        if i not in poses:regs[i]=reject('frame disconnected from target');continue
        incident=[e for e in active if i in [e['i'],e['j']]];info['graph_degree']=len(incident)
        if len(incident)<2:regs[i]=reject('fewer than two graph edges');continue
        local=[];train=[];validation=[];H=poses[i]
        for e in incident:
            local.append(e['p'] if e['i']==i else e['q'])
            train.extend(np.linalg.norm(transform(poses[e['i']],e['p'])-transform(poses[e['j']],e['q']),axis=1))
            validation.extend(np.linalg.norm(transform(poses[e['i']],e['vp'])-transform(poses[e['j']],e['vq']),axis=1))
        validation=np.asarray(validation);good=validation<5
        info.update(inliers=sum(len(p) for p in local),median_reprojection_px=float(np.median(train)),heldout_matches=len(validation),heldout_inlier_fraction=float(good.mean()),heldout_inliers=int(good.sum()),homography=H.tolist())
        if good.sum()<6 or good.mean()<.3 or np.median(train)>3:regs[i]=reject('joint/held-out geometry failed');continue
        corners=np.float32([[0,0],[size-1,0],[size-1,size-1],[0,size-1]])
        denom=np.c_[corners,np.ones(4)]@H[2];quad=transform(H,corners).astype(np.float32)
        if np.any(denom<=.15) or not cv2.isContourConvex(quad) or not .25<cv2.contourArea(quad,oriented=True)/size**2<4:
            regs[i]=reject('implausible transform');continue
        pts=transform(H,np.concatenate(local)).astype(np.float32);hull=cv2.convexHull(pts)
        support=np.zeros((size,size),np.uint8);cv2.fillConvexPoly(support,np.rint(hull).astype(np.int32),1)
        safe=cv2.erode(allowed[i].astype(np.uint8),np.ones((5,5),np.uint8));visible=(cv2.warpPerspective(safe,H,(size,size),flags=cv2.INTER_NEAREST)>0)&allowed[target]&(support>0)
        warped=cv2.warpPerspective(source,H,(size,size));error=np.abs(warped.astype(np.float32)-images[target]).mean(2)
        info.update(audit_pixels=int(visible.sum()),audit_median_rgb=float(np.median(error[visible])) if visible.any() else None,audit_p90_rgb=float(np.percentile(error[visible],90)) if visible.any() else None)
        if visible.sum()<500 or info['audit_median_rgb']>12 or info['audit_p90_rgb']>35:regs[i]=reject('visible-ground photometric audit failed');continue
        info.update(accepted=True,reason='converged geometry and visible-ground audit passed');regs[i]=Registration(H,support>0,info)
    return regs


def spatial_deletions(n,target,edges,size,model,poses):
    """Leave a world-space block out of ALL edges, without resampling pixels."""
    assignments=[];counts=np.zeros(8,int)
    for e in edges:
        mid=(transform(poses[e['i']],e['p'])+transform(poses[e['j']],e['q']))/2
        blocks=(mid[:,0]>=size/2).astype(int)+2*np.floor(mid[:,1]/size*4).astype(int).clip(0,3)
        assignments.append(blocks);counts+=np.bincount(blocks,minlength=8)
    total=int(counts.sum());attempts=[];fits=[]
    for block in range(8):
        if counts[block]<max(8,total*.05):continue
        subset=[]
        for e,assignment in zip(edges,assignments):
            keep=assignment!=block
            if keep.sum()>=4:subset.append(dict(e,p=e['p'][keep],q=e['q'][keep]))
        P,info,active=fit_constrained_graph(n,target,subset,size,model)
        attempts.append(dict(block=block,removed=int(counts[block]),**info));fits.append(P if info['optimizer_success'] else {})
    return fits,dict(attempts=attempts,required_minimum=3,all_attempts_required_per_donor=True)


def stability_support(regs,poses,replicates,size,max_px=2):
    axis=np.linspace(0,size-1,96);xx,yy=np.meshgrid(axis,axis);grid=np.c_[xx.ravel(),yy.ravel()]
    out={};maps={}
    for i,reg in regs.items():
        if reg.homography is None:out[i]=reg;continue
        info=dict(reg.diagnostics,stability_replicates=len(replicates),stability_threshold_px=max_px)
        if len(replicates)<3 or any(i not in P for P in replicates):
            out[i]=Registration(None,np.zeros_like(reg.support),dict(info,accepted=False,reason='spatial deletion loses constrained geometry'));continue
        base_source=transform(np.linalg.inv(poses[i]),grid);delta=np.zeros(len(grid))
        for P in replicates:delta=np.maximum(delta,np.linalg.norm(transform(P[i],base_source)-grid,axis=1))
        # Interpolation only evaluates a sensitivity estimate; no probability claim.
        y,x=np.mgrid[:size,:size];mx=(x*(95/(size-1))).astype(np.float32);my=(y*(95/(size-1))).astype(np.float32)
        sensitivity=cv2.remap(delta.reshape(96,96).astype(np.float32),mx,my,cv2.INTER_LINEAR)
        support=reg.support&(sensitivity<=max_px);maps[i]=sensitivity
        info.update(stable_support_pixels=int(support.sum()),original_support_pixels=int(reg.support.sum()),stability_max_on_original_support=float(sensitivity[reg.support].max()) if reg.support.any() else None)
        out[i]=Registration(reg.homography,support,info)
    return out,maps
