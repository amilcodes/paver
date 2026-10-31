"""Sequence-wide planar alignment with masked, native-resolution tiled features.

Correspondence validation is held out before edge RANSAC and joint fitting.
This is a planar pose graph, not calibrated 3D reconstruction.
"""
import hashlib,heapq
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from matching_v2 import LearnedMatcher
from repair import Registration

class TiledMatcher(LearnedMatcher):
    name='disk_tiled_lightglue'
    def __init__(self,device='auto'):
        super().__init__(device,max_keypoints=1024)
    def describe(self,image,allowed):
        torch=self.torch;clean=image.copy();clean[~allowed]=0
        key=hashlib.sha256(clean.tobytes()+allowed.tobytes()).hexdigest()
        if key in self.cache:return self.cache[key]
        h,w=allowed.shape;ys,xs=np.nonzero(allowed)
        if not len(xs):return super().describe(image,allowed)
        # Tile the annotated ground bounding box at the original pixel scale.
        def starts(lo,hi,limit):
            if hi-lo<1024:return [max(0,min(limit-1024,(lo+hi-1024)//2))]
            a=list(range(max(0,lo),max(0,hi-1024)+1,768));a.append(min(limit-1024,hi-1024))
            return sorted(set(a))
        chunks=[];distance=cv2.distanceTransform(allowed.astype(np.uint8),cv2.DIST_L2,5)
        for y in starts(int(ys.min()),int(ys.max()+1),h):
            for x in starts(int(xs.min()),int(xs.max()+1),w):
                crop=clean[y:y+1024,x:x+1024]
                if allowed[y:y+1024,x:x+1024].sum()<2000:continue
                tensor=torch.from_numpy(crop.transpose(2,0,1).copy()).float().to(self.device)/255
                with torch.inference_mode():features=self.extractor.extract(tensor,resize=1024)
                xy=features['keypoints'][0].cpu().numpy();points=xy+[x,y]
                ix=np.rint(points[:,0]).astype(int).clip(0,w-1);iy=np.rint(points[:,1]).astype(int).clip(0,h-1)
                keep=(distance[iy,ix]>32)&(xy[:,0]>32)&(xy[:,1]>32)&(xy[:,0]<crop.shape[1]-32)&(xy[:,1]<crop.shape[0]-32)
                if not keep.any():continue
                k=torch.tensor(keep,device=self.device)
                chunks.append((torch.tensor(points[keep],device=self.device,dtype=torch.float32),features['descriptors'][0][k],features['keypoint_scores'][0][k]))
        if not chunks:return super().describe(image,allowed)
        pts,desc,scores=[torch.cat([c[i] for c in chunks]) for i in range(3)]
        xy=pts.cpu().numpy();strength=scores.cpu().numpy();selected=[];occupied=set();bins={}
        # Dedupe overlap, retain features from the entire ground strip.
        for i in np.argsort(-strength):
            cell=tuple(np.rint(xy[i]/6).astype(int));band=int(xy[i,1]/h*8)
            if cell in occupied or bins.get(band,0)>=256:continue
            occupied.add(cell);bins[band]=bins.get(band,0)+1;selected.append(int(i))
            if len(selected)==2048:break
        idx=torch.tensor(selected,device=self.device)
        result={'keypoints':pts[idx][None],'descriptors':desc[idx][None],'keypoint_scores':scores[idx][None],
                'image_size':torch.tensor([[w,h]],device=self.device,dtype=torch.float32)}
        self.cache[key]=result
        return result

def transform(H,p):
    q=np.c_[p,np.ones(len(p))]@H.T
    return q[:,:2]/np.where(abs(q[:,2:])<1e-8,1e-8,q[:,2:])

def propose_edge(i,j,p,q,seed=29):
    info={'i':i,'j':j,'matches':len(p),'accepted':False}
    if len(p)<16:return None,dict(info,reason='insufficient matches')
    order=np.random.default_rng(seed+i*17+j*31).permutation(len(p));cut=int(len(p)*.75)
    tr,va=order[:cut],order[cut:];cv2.setRNGSeed(seed)
    H,k=cv2.findHomography(p[tr],q[tr],cv2.USAC_MAGSAC,3,maxIters=10000,confidence=.999)
    if H is None or k is None or not np.isfinite(H).all():return None,dict(info,reason='no finite edge transform')
    fit=tr[k.ravel()>0];err=np.linalg.norm(transform(H,p[va])-q[va],axis=1)
    inliers=int(len(fit));fraction=float(len(fit)/len(tr));valid=err<5
    info.update(train_inliers=inliers,inlier_ratio=fraction,validation_matches=len(va),validation_inliers=int(valid.sum()),validation_fraction=float(valid.mean()),validation_median_px=float(np.median(err)))
    if inliers<8 or fraction<.2 or valid.sum()<3 or valid.mean()<.3:return None,dict(info,reason='edge train/held-out support failed')
    # Collinear road marks do not determine a stable projective plane.
    spread=np.sqrt(np.maximum(0,np.linalg.eigvalsh(np.cov(p[fit].T))))
    if spread.min()<12:return None,dict(info,reason='edge support nearly collinear')
    if len(fit)>160:fit=np.random.default_rng(seed).choice(fit,160,replace=False)
    edge={'i':i,'j':j,'p':p[fit],'q':q[fit],'vp':p[va],'vq':q[va],'H':H,'info':info}
    return edge,dict(info,accepted=True,reason='edge proposal passes held-out correspondence check')

def initialize(n,target,edges):
    graph={i:[] for i in range(n)}
    for e in edges:
        i,j=e['i'],e['j'];H=e['H']
        try:inv=np.linalg.inv(H)
        except np.linalg.LinAlgError:continue
        cost=1/max(1,len(e['p']))
        graph[j].append((i,H,cost));graph[i].append((j,inv,cost))
    poses={target:np.eye(3)};distance={target:0};heap=[(0,target)]
    while heap:
        d,i=heapq.heappop(heap)
        if d>distance[i]:continue
        for j,H,cost in graph[i]:
            if d+cost<distance.get(j,float('inf')):
                P=poses[i]@H
                if abs(P[2,2])<1e-8:continue
                distance[j]=d+cost;poses[j]=P/P[2,2];heapq.heappush(heap,(d+cost,j))
    return poses

def fit_graph(n,target,edges,size):
    poses=initialize(n,target,edges)
    active=[e for e in edges if e['i'] in poses and e['j'] in poses]
    if len(poses)<2:return poses,{'connected_frames':len(poses),'reason':'target disconnected'},active
    nodes=[i for i in sorted(poses) if i!=target];lookup={i:k for k,i in enumerate(nodes)}
    scale=size/2;N=np.array([[1/scale,0,-1],[0,1/scale,-1],[0,0,1.]])
    inv=np.linalg.inv(N)
    def pack(H):
        z=N@H@inv;z=z/z[2,2];return z.ravel()[:8]
    x=np.concatenate([pack(poses[i]) for i in nodes])
    def unpack(x):return {target:np.eye(3),**{i:inv@np.r_[x[8*k:8*k+8],1].reshape(3,3)@N for i,k in lookup.items()}}
    def residual(x):
        H=unpack(x)
        return np.concatenate([(transform(H[e['i']],e['p'])-transform(H[e['j']],e['q'])).ravel() for e in active])
    rows=sum(2*len(e['p']) for e in active);sparsity=lil_matrix((rows,len(x)),dtype=int);start=0
    for e in active:
        end=start+2*len(e['p'])
        for i in [e['i'],e['j']]:
            if i in lookup:sparsity[start:end,lookup[i]*8:lookup[i]*8+8]=1
        start=end
    result=least_squares(residual,x,loss='soft_l1',f_scale=2,jac_sparsity=sparsity.tocsr(),max_nfev=120,ftol=1e-6,xtol=1e-6,gtol=1e-6)
    H=unpack(result.x)
    for i in H:H[i]/=H[i][2,2]
    return H,{'connected_frames':len(H),'edges':len(active),'optimizer_success':bool(result.success),'optimizer_message':result.message,
              'nfev':result.nfev,'initial_median_fit_px':float(np.median(abs(residual(x)))),'final_median_fit_px':float(np.median(abs(residual(result.x))))},active

def sequence_registrations(images,allowed,edges,target=2):
    n=len(images);size=images[target].shape[0];poses,summary,active=fit_graph(n,target,edges,size)
    regs={};empty=np.zeros((size,size),bool);anchor=images[target]
    for i,source in enumerate(images):
        if i==target:continue
        info={'method':'joint_planar_sequence','accepted':False,'inliers':0,'median_reprojection_px':0.0}
        def reject(reason):return Registration(None,empty,dict(info,reason=reason))
        if i not in poses:regs[i]=reject('frame disconnected from target');continue
        incident=[e for e in active if i in [e['i'],e['j']]]
        info['graph_degree']=len(incident)
        if len(incident)<2:regs[i]=reject('fewer than two graph edges');continue
        H=poses[i];local=[];validation=[];train=[]
        for e in incident:
            local.append(e['p'] if e['i']==i else e['q'])
            train.extend(np.linalg.norm(transform(poses[e['i']],e['p'])-transform(poses[e['j']],e['q']),axis=1).tolist())
            validation.extend(np.linalg.norm(transform(poses[e['i']],e['vp'])-transform(poses[e['j']],e['vq']),axis=1).tolist())
        validation=np.asarray(validation);good=validation<5
        info.update(inliers=sum(len(p) for p in local),median_reprojection_px=float(np.median(train)),heldout_matches=len(validation),heldout_inlier_fraction=float(good.mean()),heldout_inliers=int(good.sum()),heldout_inlier_median_px=float(np.median(validation[good])) if good.any() else None,homography=H.tolist())
        if good.sum()<6 or good.mean()<.3 or np.median(train)>3:regs[i]=reject('joint/held-out geometry failed');continue
        corners=np.float32([[0,0],[size-1,0],[size-1,size-1],[0,size-1]])
        denom=np.c_[corners,np.ones(4)]@H[2];quad=transform(H,corners).astype(np.float32)
        if np.any(denom<=.15) or not cv2.isContourConvex(quad) or not .25<cv2.contourArea(quad,oriented=True)/size**2<4:
            regs[i]=reject('implausible joint transform');continue
        pts=transform(H,np.concatenate(local)).astype(np.float32);hull=cv2.convexHull(pts)
        support=np.zeros((size,size),np.uint8);cv2.fillConvexPoly(support,np.rint(hull).astype(np.int32),1)
        safe=cv2.erode(allowed[i].astype(np.uint8),np.ones((5,5),np.uint8))
        visible=(cv2.warpPerspective(safe,H,(size,size),flags=cv2.INTER_NEAREST)>0)&allowed[target]&(support>0)
        warped=cv2.warpPerspective(source,H,(size,size));error=np.abs(warped.astype(np.float32)-anchor).mean(2)
        info.update(audit_pixels=int(visible.sum()),audit_median_rgb=float(np.median(error[visible])) if visible.any() else None,audit_p90_rgb=float(np.percentile(error[visible],90)) if visible.any() else None)
        if visible.sum()<500 or info['audit_median_rgb']>12 or info['audit_p90_rgb']>35:
            regs[i]=reject('visible-ground photometric audit failed');continue
        info.update(accepted=True,reason='joint sequence geometry and visible-ground audit passed')
        regs[i]=Registration(H,support>0,info)
    summary['poses']={str(k):v.tolist() for k,v in poses.items()}
    return regs,summary
