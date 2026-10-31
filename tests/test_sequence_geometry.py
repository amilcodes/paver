import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from sequence_geometry import propose_edge,fit_graph,initialize,transform,sequence_registrations

def points():
 x,y=np.meshgrid(np.linspace(40,210,11),np.linspace(40,210,11))
 return np.c_[x.ravel(),y.ravel()].astype(np.float32)
def translation(dx,dy):return np.array([[1.,0,dx],[0,1.,dy],[0,0,1.]])
def make_edges():
 # Camera zero has no direct edge to the anchored target (2).
 shifts=[(-20,-14),(-10,-7),(0,0),(10,7),(20,14)];world=points();edges=[]
 for i,j in [(0,1),(1,2),(0,3),(3,2),(3,4),(2,4),(1,3)]:
  p=world-np.array(shifts[i]);q=world-np.array(shifts[j]);e,info=propose_edge(i,j,p.astype(np.float32),q.astype(np.float32))
  assert e is not None,info
  edges.append(e)
 return edges,shifts

def test_sequence_recovers_indirect_camera_geometry():
 edges,shifts=make_edges();poses,info,active=fit_graph(5,2,edges,256)
 assert info['optimizer_success'];assert len(poses)==5
 for i,(dx,dy) in enumerate(shifts):
  assert np.max(np.abs(transform(poses[i],points())-transform(translation(dx,dy),points())))<.01

def test_disconnected_frames_are_not_invented():
 e,info=propose_edge(0,1,points(),points()+[5,0])
 poses,diag,active=fit_graph(3,2,[e],256)
 assert set(poses)=={2};assert diag['connected_frames']==1;assert not active

def test_bad_heldout_correspondences_reject_edge_despite_perfect_training_fit():
 p=points();q=p+[12.,5.];order=np.random.default_rng(29+31).permutation(len(p));cut=int(.75*len(p));q[order[cut:]]+=[65,40]
 edge,info=propose_edge(0,1,p,q.astype(np.float32))
 assert edge is None;assert info['train_inliers']>=80;assert info['validation_inliers']==0

def test_joint_fit_resists_a_small_fraction_of_bad_training_correspondences():
 edges,shifts=make_edges();e=edges[0];e['q']=e['q'].copy();e['q'][:3]+=40
 poses,info,_=fit_graph(5,2,edges,256)
 residual=np.linalg.norm(transform(poses[0],points())-transform(translation(*shifts[0]),points()),axis=1)
 assert np.median(residual)<1

def test_tiled_model_cannot_see_hidden_target_pixels():
 import os,pytest
 if os.environ.get('RUN_LEARNED_TESTS')!='1':pytest.skip('Local model integration check')
 from sequence_geometry import TiledMatcher
 from PIL import Image
 root=Path(__file__).resolve().parents[1]
 im=np.array(Image.open(root/'benchmark_v2/results/dev_marked/sift/original.jpg').resize((1200,1200)))
 allowed=np.ones((1200,1200),bool);allowed[400:800,400:800]=False
 altered=im.copy();altered[~allowed]=255-im[~allowed]
 matcher=TiledMatcher('auto');a=matcher.describe(im,allowed);matcher.cache.clear();b=matcher.describe(altered,allowed)
 assert a['keypoints'].shape[1]>10
 assert matcher.torch.equal(a['keypoints'],b['keypoints'])
 assert matcher.torch.equal(a['descriptors'],b['descriptors'])
