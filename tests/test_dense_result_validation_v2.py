import sys
from pathlib import Path
import cv2
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from metric_plane import gradient_error
from validate_dense_run_v2 import gradient_metrics,verify_evidence,verify_selected_donors
from test_dense_result_validation import fixture


@pytest.mark.parametrize('seed',[0,1,42])
def test_float32_orientation_reproduces_frozen_scorer_after_stencil_check(seed):
    rng=np.random.default_rng(seed)
    truth=rng.integers(0,256,(96,96,3),dtype=np.uint8)
    pred=rng.integers(80,84,(96,96,3),dtype=np.uint8)
    pred[24:60,20:65]=[80,81,83]
    mask=np.zeros((96,96),bool);mask[4:-4,4:-4]=True;mask[40:46]=False
    assert gradient_metrics(pred,truth,mask)==gradient_error(pred,truth,mask)


def test_independent_stencil_detects_wrong_gradient_primitive(monkeypatch):
    image=np.zeros((32,32,3),np.uint8);mask=np.ones((32,32),bool)
    original=cv2.Sobel
    def corrupt(*args,**kwargs):return original(*args,**kwargs)+.25
    monkeypatch.setattr(cv2,'Sobel',corrupt)
    with pytest.raises(ValueError,match='independent stencil'):gradient_metrics(image,image,mask)


@pytest.mark.parametrize('footprint,empty',[(False,False),(True,False),(False,True)])
def test_saved_evidence_and_provenance_checks_remain_intact(footprint,empty):
    args,images,allowed=fixture(footprint=footprint,all_abstain=empty)
    verify_evidence(**args);verify_selected_donors(args['evidence'],images,allowed,args['registrations'])
    args['reported']['candidate']['coverage']+=.1
    with pytest.raises(ValueError):verify_evidence(**args)
