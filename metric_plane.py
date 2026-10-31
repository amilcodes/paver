"""Sequence cameras constrained by known nadir intrinsics and one shared plane.

Ground points are (X,Y,1) in units of target-camera plane distance.
M_i = R_i [[1,0,-C_x],[0,1,-C_y],[0,0,h_i]], M_target=R_target.
Source-to-target H_i = K R_target inv(M_i) inv(K).
This is planar reconstruction, not a general 3D scene or proven metric recovery.
"""
import math
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from sequence_geometry import initialize,transform
from trajectory_matching import motion_prior


def intrinsics(size,fov):
    f=size/(2*np.tan(np.deg2rad(fov/2)));c=(size-1)/2
    return np.array([[f,0,c],[0,f,c],[0,0,1.]])

def rotation(v):return cv2.Rodrigues(np.asarray(v,dtype=float))[0]
def plane_camera(v):
    R=rotation(v[:3]);x,y=v[3:5];h=np.exp(v[5])
    return R@np.array([[1.,0,-x],[0,1.,-y],[0,0,h]])

def camera_parameters(M):
    s=(np.linalg.norm(M[:,0])+np.linalg.norm(M[:,1]))/2
    u,v=M[:,0]/s,M[:,1]/s;Q=np.c_[u,v,np.cross(u,v)];U,_,V=np.linalg.svd(Q)
    R=U@np.diag([1,1,np.linalg.det(U@V)])@V;t=R.T@(M[:,2]/s)
    h=float(np.clip(t[2],.40001,2.49999))
    return np.r_[cv2.Rodrigues(R)[0].ravel(),-t[:2],np.log(h)]

def fit_metric_graph(n,target,edges,size,fov,metadata=None,use_gps=False,max_nfev=160):
    initial=initialize(n,target,edges);active=[e for e in edges if e['i'] in initial and e['j'] in initial]
    info=dict(model='metric_gps' if use_gps else 'metric_free',connected_frames=len(initial),edges=len(active),optimizer_success=False)
    if len(initial)<2:return initial,dict(info,reason='target disconnected'),active
    nodes=[i for i in sorted(initial) if i!=target];lookup={i:2+6*k for k,i in enumerate(nodes)};K=intrinsics(size,fov);invK=np.linalg.inv(K)
    inverse_maps={i:np.linalg.inv(invK@initial[i]@K) for i in nodes}
    def plane_constraints(angles):
        Rt=rotation([angles[0],angles[1],0]);result=[]
        for G in inverse_maps.values():
            M=G@Rt;a,b=M[:,0],M[:,1];aa=np.dot(a,a);bb=np.dot(b,b);normal=max(1e-12,(aa+bb)/2)
            result.extend([np.dot(a,b)/normal,(aa-bb)/normal])
        return np.array(result)
    upgrade=least_squares(plane_constraints,[0.,0.],bounds=(-.35,.35),loss='soft_l1',f_scale=.02,max_nfev=100)
    Rt=rotation([*upgrade.x,0]);x=np.r_[upgrade.x,np.concatenate([camera_parameters(inverse_maps[i]@Rt) for i in nodes])]
    lower=np.r_[[-.35,-.35],np.tile([-np.pi,-np.pi,-np.pi,-100,-100,np.log(.4)],len(nodes))]
    upper=np.r_[[.35,.35],np.tile([np.pi,np.pi,np.pi,100,100,np.log(2.5)],len(nodes))]
    gps={};yaw={};height0=2.2
    if use_gps:
        if metadata is None:raise ValueError('GPS model needs metadata')
        a0=math.radians(metadata[target]['properties']['view:azimuth']);lon,lat=metadata[target]['geometry']['coordinates'][:2]
        B=np.array([[math.cos(a0),-math.sin(a0)],[-math.sin(a0),-math.cos(a0)]])
        for i in nodes:
            lon1,lat1=metadata[i]['geometry']['coordinates'][:2]
            gps[i]=B.T@np.array([(lon1-lon)*111320*math.cos(math.radians(lat)),(lat1-lat)*111320])
            a=math.radians(metadata[i]['properties']['view:azimuth'])-a0;yaw[i]=(a+np.pi)%(2*np.pi)-np.pi
        _,scale=motion_prior(metadata,active,target,size)
        if scale.get('pixels_per_meter'):height0=float(np.clip(K[0,0]/scale['pixels_per_meter'],1.0001,4.4999))
        x=np.r_[x,np.log(height0)];lower=np.r_[lower,np.log(1.)];upper=np.r_[upper,np.log(4.5)]
    x=np.clip(x,lower+1e-7,upper-1e-7)
    def unpack(v):
        R=rotation([v[0],v[1],0]);poses={target:np.eye(3)}
        for i,k in lookup.items():
            H=K@R@np.linalg.inv(plane_camera(v[k:k+6]))@invK
            poses[i]=H/(H[2,2] if abs(H[2,2])>1e-9 else 1e-9)
        return poses
    def point_residual(v):
        H=unpack(v)
        return np.concatenate([(transform(H[e['i']],e['p'])-transform(H[e['j']],e['q'])).ravel() for e in active])
    def residual(v):
        result=point_residual(v)
        if use_gps:
            prior=[v[0]/.15,v[1]/.15];height=np.exp(v[-1])
            for i,k in lookup.items():
                prior.extend((v[k+3:k+5]*height-gps[i])/2.)
                prior.extend([v[k+5]/.15,math.atan2(math.sin(v[k+2]+yaw[i]),math.cos(v[k+2]+yaw[i]))/.14])
            result=np.r_[result,prior]
        return result
    count=sum(2*len(e['p']) for e in active);prior_rows=2+4*len(nodes) if use_gps else 0
    J=lil_matrix((count+prior_rows,len(x)),dtype=int);start=0
    for e in active:
        end=start+2*len(e['p']);J[start:end,:2]=1
        for i in [e['i'],e['j']]:
            if i!=target:J[start:end,lookup[i]:lookup[i]+6]=1
        start=end
    if use_gps:
        J[count:count+2,:2]=1
        for j,i in enumerate(nodes):
            row=count+2+4*j;k=lookup[i];J[row:row+4,k:k+6]=1;J[row:row+2,-1]=1
    if max_nfev<1:return initial,dict(info,reason='optimization budget disabled'),active
    fitted=least_squares(residual,x,bounds=(lower,upper),jac_sparsity=J.tocsr(),loss='soft_l1',f_scale=2,max_nfev=max_nfev,ftol=1e-6,xtol=1e-6,gtol=1e-6)
    H=unpack(fitted.x);boundary=(fitted.x-lower<1e-4)|(upper-fitted.x<1e-4)
    info.update(solver_converged=bool(fitted.success),optimizer_success=bool(fitted.success and not boundary.any()),boundary_hits=np.flatnonzero(boundary).tolist(),solver_message=fitted.message,nfev=fitted.nfev,
                parameters=len(x),initial_plane_tilt=upgrade.x.tolist(),plane_tilt_rad=fitted.x[:2].tolist(),initial_median_fit_px=float(np.median(np.linalg.norm(point_residual(x).reshape(-1,2),axis=1))),
                median_training_residual_px=float(np.median(np.linalg.norm(point_residual(fitted.x).reshape(-1,2),axis=1))),camera_parameters={str(i):fitted.x[k:k+6].tolist() for i,k in lookup.items()},
                target_height_m=float(np.exp(fitted.x[-1])) if use_gps else None,
                reason='converged physical plane model' if fitted.success and not boundary.any() else 'optimizer nonconvergence or physical bound reached')
    return H,info,active


def gradient_error(pred,truth,accepted):
    # Three-pixel erosion keeps derivative stencils inside accepted transfers.
    safe=cv2.erode(accepted.astype(np.uint8),np.ones((7,7),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    if not safe.any():return dict(pixels=0,gradient_mae=None,edge_pixels=0,orientation_error=None)
    def gradients(im):
        gray=cv2.cvtColor(im.astype(np.float32),cv2.COLOR_RGB2GRAY)
        return np.stack([cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3,scale=1/8),cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3,scale=1/8)],axis=2)
    p,t=gradients(pred)[safe],gradients(truth)[safe];pn=np.linalg.norm(p,axis=1);tn=np.linalg.norm(t,axis=1);edges=tn>=4
    orientation=1-np.sum(p*t,axis=1)/np.maximum(pn*tn,1e-6);orientation=np.clip(orientation,0,2)
    return dict(pixels=int(safe.sum()),gradient_mae=float(np.abs(p-t).mean()),edge_pixels=int(edges.sum()),orientation_error=float(orientation[edges].mean()) if edges.any() else None)
