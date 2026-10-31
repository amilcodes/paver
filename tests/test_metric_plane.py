import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from metric_plane import intrinsics,rotation,plane_camera,fit_metric_graph,gradient_error
from sequence_geometry import transform


def fixture():
 n=5;target=2;size=512;fov=145;K=intrinsics(size,fov);Rt=rotation([.07,-.05,0]);rng=np.random.default_rng(21);world=rng.uniform(-1.5,1.5,(120,2))
 M={2:Rt};params={0:[.03,-.09,.05,-1.4,.4,np.log(1.06)],1:[.09,-.03,.02,-.7,.1,np.log(.96)],3:[.10,-.07,-.04,.6,-.25,np.log(1.03)],4:[.02,-.02,-.07,1.3,-.4,np.log(.98)]}
 for i,p in params.items():M[i]=plane_camera(p)
 truth={i:K@Rt@np.linalg.inv(m)@np.linalg.inv(K) for i,m in M.items()};edges=[]
 for i,j in [(0,1),(1,2),(0,3),(3,2),(3,4),(4,2),(1,4)]:
  p=transform(K@M[i],world);q=transform(K@M[j],world);H=K@M[j]@np.linalg.inv(M[i])@np.linalg.inv(K);H/=H[2,2]
  edges.append(dict(i=i,j=j,p=p[:90],q=q[:90],vp=p[90:],vq=q[90:],H=H))
 return edges,truth,size,fov

def test_known_perspective_plane_projection_and_indirect_fit():
 edges,truth,size,fov=fixture();poses,info,_=fit_metric_graph(5,2,edges,size,fov)
 assert info['optimizer_success'],info
 for e in edges:
  assert np.max(np.linalg.norm(transform(poses[e['i']],e['vp'])-transform(poses[e['j']],e['vq']),axis=1))<.02
 for i in truth:
  xy=np.array([[190,180],[250,250],[320,330.]])
  assert np.max(abs(transform(poses[i],xy)-transform(truth[i],xy)))<.1

def test_disconnected_and_exhausted_fit_never_succeed():
 edges,_,size,fov=fixture()
 _,info,_=fit_metric_graph(5,2,[e for e in edges if 2 not in [e['i'],e['j']]],size,fov)
 assert not info['optimizer_success'];assert info['connected_frames']==1
 _,info,_=fit_metric_graph(5,2,edges,size,fov,max_nfev=0)
 assert not info['optimizer_success']

def test_gradient_diagnostic_separates_offset_from_shift():
 rng=np.random.default_rng(25);g=cv2.GaussianBlur(rng.integers(40,150,(128,128,3),dtype=np.uint8),(5,5),0);mask=np.zeros((128,128),bool);mask[20:108,20:108]=True
 offset=gradient_error(g+20,g,mask);shift=gradient_error(np.roll(g,3,axis=1),g,mask)
 assert offset['gradient_mae']<1e-4
 assert shift['gradient_mae']>1

def test_gradient_stencil_does_not_read_outside_accepted_area():
 rng=np.random.default_rng(32);g=rng.integers(30,170,(64,64,3),dtype=np.uint8);mask=np.zeros((64,64),bool);mask[15:49,15:49]=True
 altered=g.copy();altered[~mask]=0
 assert gradient_error(altered,g,mask)['gradient_mae']==0

def test_gps_regularization_retains_known_heldout_geometry():
 import math
 edges,truth,size,fov=fixture();xy=[[-1.4,.4],[-.7,.1],[0,0],[.6,-.25],[1.3,-.4]];angles=[.05,.02,0,-.04,-.07];height=2.2
 metadata=[]
 for p,a in zip(xy,angles):
  metadata.append({'properties':{'view:azimuth':(-math.degrees(a))%360},'geometry':{'coordinates':[p[0]*height/(111320*math.cos(math.radians(45))),45-p[1]*height/111320]}})
 poses,info,_=fit_metric_graph(5,2,edges,size,fov,metadata=metadata,use_gps=True)
 assert info['optimizer_success'],info
 error=np.concatenate([np.linalg.norm(transform(poses[e['i']],e['vp'])-transform(poses[e['j']],e['vq']),axis=1) for e in edges])
 assert np.percentile(error,90)<.1
 assert 1<info['target_height_m']<4.5
