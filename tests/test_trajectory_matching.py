import json,sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from trajectory_matching import motion_prior,track_prior


def test_trajectory_scale_uses_observed_ground_motion():
    metadata=[{'properties':{'view:azimuth':0},'geometry':{'coordinates':[0,i*5/111320]}} for i in range(5)]
    p=np.array([[100,100],[200,200],[300,400],[200,600]],float)
    edge={'i':0,'j':1,'p':p,'q':p+[0,400]}
    poses,info=motion_prior(metadata,[edge],target=2)
    assert abs(info['pixels_per_meter']-80)<1e-6
    assert np.allclose(poses[0][:2,2],[0,800])
    assert np.allclose(poses[4][:2,2],[0,-800])


def test_flow_cannot_observe_hidden_target_content():
    rng=np.random.default_rng(12)
    source=cv2.GaussianBlur(rng.integers(0,255,(320,320,3),dtype=np.uint8),(3,3),0)
    H=np.array([[1,0,8],[0,1,7],[0,0,1]],float)
    target=cv2.warpPerspective(source,H,(320,320));allowed=np.ones((320,320),bool);allowed[130:190,130:190]=False
    altered=target.copy();altered[~allowed]=rng.integers(0,255,(int((~allowed).sum()),3),dtype=np.uint8)
    a,b,_=track_prior(source,target,np.ones_like(allowed),allowed,H)
    c,d,_=track_prior(source,altered,np.ones_like(allowed),allowed,H)
    assert len(a)>10
    assert np.array_equal(a,c) and np.array_equal(b,d)
    assert np.median(np.linalg.norm(b-a-[8,7],axis=1))<.1
