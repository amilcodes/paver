"""Masked DISK + LightGlue matching and audited planar registration.

No hidden target pixels enter extraction, matching, model selection or fusion.
Neural scores and geometric gates are diagnostics, not probability estimates.
"""
import hashlib
import cv2
import numpy as np
from repair import Registration


class LearnedMatcher:
    name='disk_lightglue'
    def __init__(self, device='auto', max_keypoints=2048):
        import torch
        from lightglue import DISK, LightGlue
        self.torch = torch
        torch.set_num_threads(4)
        torch.manual_seed(20261031)
        self.device = ('mps' if torch.backends.mps.is_available() else 'cpu') if device=='auto' else device
        self.extractor = DISK(max_num_keypoints=max_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features='disk', flash=False, depth_confidence=-1,
                                width_confidence=-1, filter_threshold=.2).eval().to(self.device)
        self.cache = {}

    def describe(self, image, allowed):
        torch=self.torch
        clean=image.copy();clean[~allowed]=0
        key=hashlib.sha256(clean.tobytes()+allowed.tobytes()).hexdigest()
        if key in self.cache:return self.cache[key]
        tensor=torch.from_numpy(clean.transpose(2,0,1).copy()).float().to(self.device)/255.
        with torch.inference_mode():features=self.extractor.extract(tensor,resize=1024)
        points=features['keypoints'][0].cpu().numpy()
        distance=cv2.distanceTransform(allowed.astype(np.uint8),cv2.DIST_L2,5)
        xy=np.rint(points).astype(int)
        xy[:,0]=xy[:,0].clip(0,image.shape[1]-1);xy[:,1]=xy[:,1].clip(0,image.shape[0]-1)
        keep=torch.tensor(distance[xy[:,1],xy[:,0]]>24,device=self.device)
        for name in ['keypoints','descriptors','keypoint_scores']:
            features[name]=features[name][:,keep]
        self.cache[key]=features
        return features

    def correspondences(self, source, target, source_allowed, target_allowed):
        torch=self.torch
        a=self.describe(source,source_allowed);b=self.describe(target,target_allowed)
        na,nb=a['keypoints'].shape[1],b['keypoints'].shape[1]
        if min(na,nb)<4:return np.empty((0,2),np.float32),np.empty((0,2),np.float32),dict(source_features=na,target_features=nb)
        with torch.inference_mode():out=self.matcher({'image0':a,'image1':b})
        pairs=out['matches'][0].cpu().numpy()
        sp=a['keypoints'][0].cpu().numpy()[pairs[:,0]]
        tp=b['keypoints'][0].cpu().numpy()[pairs[:,1]]
        return sp.astype(np.float32),tp.astype(np.float32),dict(source_features=na,target_features=nb,
                    matching_scores=out['scores'][0].cpu().numpy().tolist())


class SIFTMatcher:
    name='sift'
    def __init__(self):self.cache={}
    def describe(self,image,allowed):
        clean=image.copy();clean[~allowed]=0
        key=hashlib.sha256(clean.tobytes()+allowed.tobytes()).hexdigest()
        if key in self.cache:return self.cache[key]
        kp,desc=cv2.SIFT_create(nfeatures=10000,contrastThreshold=.015).detectAndCompute(
            cv2.cvtColor(clean,cv2.COLOR_RGB2GRAY),allowed.astype(np.uint8)*255)
        if desc is None:return [],None
        distance=cv2.distanceTransform(allowed.astype(np.uint8),cv2.DIST_L2,5)
        keep=[i for i,k in enumerate(kp) if distance[min(int(k.pt[1]),image.shape[0]-1),
                          min(int(k.pt[0]),image.shape[1]-1)]>6*k.size+2]
        result=([kp[i] for i in keep],desc[keep] if keep else None)
        self.cache[key]=result
        return result
    def correspondences(self,source,target,source_allowed,target_allowed):
        sk,sd=self.describe(source,source_allowed);tk,td=self.describe(target,target_allowed)
        info=dict(source_features=len(sk),target_features=len(tk))
        if sd is None or td is None or min(len(sd),len(td))<2:
            return np.empty((0,2),np.float32),np.empty((0,2),np.float32),info
        matcher=cv2.BFMatcher(cv2.NORM_L2)
        def ratio(a,b):return {m.queryIdx:m.trainIdx for pair in matcher.knnMatch(a,b,k=2)
               if len(pair)==2 for m,n in [pair] if m.distance<.75*n.distance}
        a,b=ratio(sd,td),ratio(td,sd)
        pairs=[(i,j) for i,j in a.items() if b.get(j)==i]
        return np.float32([sk[i].pt for i,_ in pairs]).reshape(-1,2),np.float32([tk[j].pt for _,j in pairs]).reshape(-1,2),info


def fit_registration(sp,tp,source,target,source_allowed,target_allowed,extra=None,seed=7):
    """MAGSAC + shape gates, five-fold correspondence audit, visible-pixel audit.

    Cross-validation uses the geometric inlier set selected using all matches,
    so it is a stability diagnostic, not independent scene-ground-truth evidence.
    Target photometric audit explicitly excludes requested repair pixels.
    """
    cv2.setRNGSeed(seed)
    info=dict(extra or {},matches=len(sp),inliers=0,accepted=False)
    empty=np.zeros(target.shape[:2],bool)
    def reject(reason):
        info['reason']=reason
        return Registration(None,empty,info)
    if len(sp)<12:return reject('fewer than 12 learned matches')
    H,mask=cv2.findHomography(sp,tp,cv2.USAC_MAGSAC,3.,maxIters=10000,confidence=.999)
    if H is None or not np.isfinite(H).all():return reject('invalid homography')
    keep=mask.ravel().astype(bool)
    in_sp,in_tp=sp[keep],tp[keep]
    pred=cv2.perspectiveTransform(in_sp[:,None],H)[:,0]
    errors=np.linalg.norm(pred-in_tp,axis=1)
    hull=cv2.convexHull(in_tp)
    fraction=cv2.contourArea(hull)/target.shape[0]/target.shape[1]
    info.update(inliers=int(keep.sum()),inlier_ratio=float(keep.mean()),
                median_reprojection_px=float(np.median(errors)),inlier_hull_fraction=float(fraction),
                homography=H.tolist(),source_inliers=in_sp.tolist(),target_inliers=in_tp.tolist())
    if keep.sum()<12 or keep.mean()<.25 or fraction<.015:
        return reject('weak or concentrated correspondence support')
    # Reject a projective horizon through the source and reflection/collapse.
    h,w=source.shape[:2]
    corners=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    denominator=np.c_[corners,np.ones(4)]@H[2]
    if np.any(denominator<=.15):return reject('projective horizon through source')
    quad=cv2.perspectiveTransform(corners[:,None],H)[:,0]
    area=cv2.contourArea(quad,oriented=True)/(h*w)
    if not cv2.isContourConvex(quad) or not .25<area<4:
        return reject('implausible planar scale or orientation')
    # Deterministic randomized folds, including refit variation at repair region.
    order=np.random.default_rng(seed).permutation(len(in_sp));fold_errors=[]
    for fold in range(5):
        val=order[fold::5];train=np.setdiff1d(order,val)
        if len(train)<8 or len(val)<2:continue
        fold_H,_=cv2.findHomography(in_sp[train],in_tp[train],0)
        if fold_H is None:continue
        fold_pred=cv2.perspectiveTransform(in_sp[val,None],fold_H)[:,0]
        fold_errors.extend(np.linalg.norm(fold_pred-in_tp[val],axis=1).tolist())
    info['cv_p90_px']=float(np.percentile(fold_errors,90)) if fold_errors else None
    if not fold_errors or info['cv_p90_px']>5:return reject('unstable withheld-correspondence geometry')
    support=np.zeros(target.shape[:2],np.uint8)
    cv2.fillConvexPoly(support,np.rint(hull).astype(np.int32),255)
    warped=cv2.warpPerspective(source,H,(target.shape[1],target.shape[0]))
    source_safe=cv2.erode(source_allowed.astype(np.uint8),np.ones((5,5),np.uint8))
    valid=(cv2.warpPerspective(source_safe,H,(target.shape[1],target.shape[0]),flags=cv2.INTER_NEAREST)>0)
    valid &= target_allowed & (support>0)
    # Keep raw color; audit must not learn from the hidden target patch.
    difference=np.abs(warped.astype(np.float32)-target.astype(np.float32)).mean(2)
    info['audit_pixels']=int(valid.sum())
    info['audit_median_rgb']=float(np.median(difference[valid])) if valid.any() else None
    info['audit_p90_rgb']=float(np.percentile(difference[valid],90)) if valid.any() else None
    if valid.sum()<500 or info['audit_median_rgb']>12 or info['audit_p90_rgb']>35:
        return reject('visible-ground photometric audit failed')
    info.update(accepted=True,reason='accepted audited planar registration')
    return Registration(H,support>0,info)


def register_learned(matcher,source,target,source_allowed,target_allowed):
    sp,tp,extra=matcher.correspondences(source,target,source_allowed,target_allowed)
    extra['method']=matcher.name
    return fit_registration(sp,tp,source,target,source_allowed,target_allowed,extra)
