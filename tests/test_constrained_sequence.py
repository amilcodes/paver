import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from constrained_sequence import fit_constrained_graph,audit_poses,stability_support
from sequence_geometry import transform
from repair import Registration

def fixture(model):
 rng=np.random.default_rng(42);world=rng.uniform(20,230,(120,2));poses={2:np.eye(3)}
 for i in [0,1,3,4]:
  a=.02*(i-2);c,s=np.cos(a),np.sin(a);H=np.array([[c,-s,5*(i-2)],[s,c,12*(i-2)],[0,0,1.]])
  if model=='affine':H[0,1]+=.03*(i-2)
  poses[i]=H
 edges=[]
 for i,j in [(0,1),(1,2),(2,3),(3,4),(0,3),(1,4)]:
  p=transform(np.linalg.inv(poses[i]),world);q=transform(np.linalg.inv(poses[j]),world);H=np.linalg.inv(poses[j])@poses[i]
  edges.append(dict(i=i,j=j,p=p[:90],q=q[:90],vp=p[90:],vq=q[90:],H=H))
 return edges,poses,world

def test_known_indirect_sequence_geometry_for_both_models():
 for model in ['affine','similarity']:
  edges,truth,points=fixture(model);poses,diag,_=fit_constrained_graph(5,2,edges,256,model)
  assert diag['optimizer_success']
  for i in truth:assert np.max(abs(transform(poses[i],points)-transform(truth[i],points)))<1e-6

def test_nonconverged_geometry_cannot_authorize_transfer():
 edges,poses,_=fixture('affine');P,d,a=fit_constrained_graph(5,2,edges,256,'affine',max_iterations=0)
 assert not d['optimizer_success']
 images=[np.full((256,256,3),100,np.uint8)]*5;allowed=[np.ones((256,256),bool)]*5
 regs=audit_poses(images,allowed,P,d,a,2)
 assert all(r.homography is None and not r.support.any() for r in regs.values())

def test_collinear_correspondences_cannot_solve_affine_graph():
 edges,_,_=fixture('affine')
 for e in edges:e['p'][:,0]=100;e['q'][:,0]=100
 _,d,_=fit_constrained_graph(5,2,edges,256,'affine')
 assert not d['optimizer_success'];assert d['rank']<d['parameters']

def test_spatial_instability_and_disconnection_force_abstention():
 support=np.ones((64,64),bool);I=np.eye(3);reg=Registration(I,support,{'accepted':True})
 moved=I.copy();moved[0,2]=5
 out,_=stability_support({0:reg},{0:I},[{0:I},{0:I},{0:moved}],64)
 assert not out[0].support.any()
 out,_=stability_support({0:reg},{0:I},[{0:I},{},{0:I}],64)
 assert out[0].homography is None
