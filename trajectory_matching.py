"""Sequence-derived GPS scale initializes masked optical-flow correspondences.

GPS is only a search prior. Pixel transfers still require independent held-out
correspondence, sequence geometry and photometric acceptance checks.
"""
import math
import cv2
import numpy as np
from sequence_geometry import transform


def motion_prior(metadata,edges,target=2,size=2200):
    az=math.radians(metadata[target]['properties']['view:azimuth'])
    B=np.array([[math.cos(az),-math.sin(az)],[-math.sin(az),-math.cos(az)]])
    lon,lat=metadata[target]['geometry']['coordinates'][:2];rot=[];pos=[];c=(size-1)/2
    for m in metadata:
        x,y=m['geometry']['coordinates'][:2];pos.append(B.T@np.array([(x-lon)*111320*math.cos(math.radians(lat)),(y-lat)*111320]))
        a=math.radians(m['properties']['view:azimuth'])-az
        rot.append(np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]]))
    scales=[];detail=[]
    for e in edges:
        i,j=e['i'],e['j'];d=pos[i]-pos[j]
        if np.linalg.norm(d)<1:continue
        a=(e['p']-c)@rot[i].T-(e['q']-c)@rot[j].T
        k=float(np.median(-(a@d)/(d@d)))
        if 20<k<150:scales.append(k);detail.append(dict(i=i,j=j,pixels_per_meter=k))
    if not scales:return None,dict(reason='no observed sequence scale')
    # Equal edge weighting prevents one textured location dominating scale.
    scale=float(np.median(scales));poses=[]
    for R,t in zip(rot,pos):
        H=np.eye(3);H[:2,:2]=R;H[:2,2]=c-R@np.array([c,c])+scale*t;poses.append(H)
    return poses,dict(pixels_per_meter=scale,scale_edge_count=len(scales),edge_scales=detail,
                      scale_mad=float(np.median(abs(np.asarray(scales)-scale))))


def track_prior(source,target,sa,ta,H):
    # Mask before all filtering, including pyramids and corner detection.
    sg=cv2.cvtColor(source,cv2.COLOR_RGB2GRAY);tg=cv2.cvtColor(target,cv2.COLOR_RGB2GRAY)
    sg[~sa]=0;tg[~ta]=0
    sd=cv2.distanceTransform(sa.astype(np.uint8),cv2.DIST_L2,5)
    td=cv2.distanceTransform(ta.astype(np.uint8),cv2.DIST_L2,5)
    points=cv2.goodFeaturesToTrack(sg,maxCorners=3500,qualityLevel=.001,minDistance=8,
                                  mask=(sd>48).astype(np.uint8)*255,blockSize=7)
    empty=np.empty((0,2),np.float32)
    if points is None:return empty,empty,dict(corners=0,tracked=0)
    p=points.reshape(-1,2);guess=transform(H,p).astype(np.float32);h,w=sg.shape
    valid=(guess[:,0]>50)&(guess[:,0]<w-50)&(guess[:,1]>50)&(guess[:,1]<h-50)
    p=p[valid];guess=guess[valid]
    if not len(p):return empty,empty,dict(corners=len(points),tracked=0)
    kw=dict(winSize=(21,21),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01),flags=cv2.OPTFLOW_USE_INITIAL_FLOW,minEigThreshold=.00001)
    q,ok,err=cv2.calcOpticalFlowPyrLK(sg,tg,p[:,None,:],guess[:,None,:].copy(),**kw)
    back,bok,berr=cv2.calcOpticalFlowPyrLK(tg,sg,q,p[:,None,:].copy(),**kw)
    q=q.reshape(-1,2);back=back.reshape(-1,2)
    ix=np.rint(q[:,0]).astype(int).clip(0,w-1);iy=np.rint(q[:,1]).astype(int).clip(0,h-1)
    good=(ok.ravel()>0)&(bok.ravel()>0)&(np.linalg.norm(p-back,axis=1)<1)&(td[iy,ix]>48)&(np.linalg.norm(q-guess,axis=1)<100)
    pp=[];qq=[]
    for x,y in zip(p[good],q[good]):
        a=cv2.getRectSubPix(sg,(21,21),tuple(x)).astype(float);b=cv2.getRectSubPix(tg,(21,21),tuple(y)).astype(float)
        a-=a.mean();b-=b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
        if den>1 and float(np.sum(a*b)/den)>.85:pp.append(x);qq.append(y)
    return np.asarray(pp,np.float32).reshape(-1,2),np.asarray(qq,np.float32).reshape(-1,2),dict(corners=len(points),prior_visible=len(p),cycle_pass=int(good.sum()),tracked=len(pp))
