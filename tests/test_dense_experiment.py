import sys,json
from pathlib import Path
import cv2
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dense_experiment import masked_inputs,register_sequence,score_transfer,cached_pair,same_pixel_error
from run_dense import verify_certificate,certificate_signature,save_evidence

POLICIES={'candidate':{'min_views':1,'max_disagreement':18},'corroborated':{'min_views':2,'max_disagreement':12}}


def fixture():
    size=256;rng=np.random.default_rng(728);world=rng.integers(30,220,(size,320,3),dtype=np.uint8)
    offsets=[0,8,16];images=[world[:,o:o+size].copy() for o in offsets]
    allowed=[]
    for _ in images:
        a=np.zeros((size,size),bool);a[12:-12,12:-12]=True;allowed.append(a)
    hidden=np.zeros((size,size),bool);hidden[105:145,105:145]=True
    return images,allowed,hidden,offsets


def oracle(offsets,bias=0,calls=None):
    def match(i,j,a,b,ma,mb):
        if calls is not None:calls.append((i,j))
        assert not a[~ma].any() and not b[~mb].any()
        x,y=np.meshgrid(np.arange(28,228,8),np.arange(28,228,8));p=np.c_[x.ravel(),y.ravel()].astype(np.float32)
        q=p+[offsets[i]-offsets[j]+bias*(i-j),0];xy=q.astype(int)
        valid=(xy[:,0]>=0)&(xy[:,0]<256)
        p,q=p[valid],q[valid];valid=ma[p[:,1].astype(int),p[:,0].astype(int)]&mb[q[:,1].astype(int),q[:,0].astype(int)]
        return p[valid],q[valid].astype(np.float32),{'fixture':'known translation oracle, not a learned model'}
    return match


def test_full_geometry_recovers_known_texture_and_hidden_changes_only_scoring(tmp_path):
    ims,allowed,hole,offsets=fixture();clean,als=masked_inputs(ims,allowed,hole,1);calls=[]
    regs,graphs,refs=register_sequence(clean,als,1,140,'hidden',tmp_path,oracle(offsets,calls=calls),{'oracle':1})
    changed=[im.copy() for im in ims];changed[1][hole]=255
    clean2,als2=masked_inputs(changed,allowed,hole,1)
    for a,b in zip(clean,clean2):assert np.array_equal(a,b)
    again,graphs2,_=register_sequence(clean2,als2,1,140,'hidden',tmp_path,oracle(offsets,calls=calls),{'oracle':1})
    assert len(calls)==3 and graphs==graphs2
    for name,r in regs.items():
        assert graphs[name]['optimizer_success']
        good,ev=score_transfer(clean,als,hole,1,r,POLICIES,ims[1])
        altered,ev2=score_transfer(clean2,als2,hole,1,again[name],POLICIES,changed[1])
        assert good['candidate']['coverage']>.9,good
        assert good['candidate']['error']['mae_255']<1,good
        assert altered['candidate']['error']['mae_255']>80
        assert np.array_equal(ev['prediction_on_mask'],ev2['prediction_on_mask'])
        actual,_=score_transfer(clean,als,hole,1,r,POLICIES,None)
        assert 'error' not in actual['candidate'] and 'gradient' not in actual['candidate']
        same=same_pixel_error(ev,ev,POLICIES)
        assert same['candidate']['pixels']==good['candidate']['accepted_pixels']


def test_coherent_but_wrong_matches_are_rejected_by_visible_photometry(tmp_path):
    ims,allowed,hole,offsets=fixture();clean,als=masked_inputs(ims,allowed,hole,1)
    regs,graphs,_=register_sequence(clean,als,1,140,'hidden',tmp_path,oracle(offsets,bias=24),{'wrong_oracle':1})
    for name,r in regs.items():
        assert graphs[name]['optimizer_success']
        assert all(x.diagnostics['reason']=='visible-ground photometric audit failed' for x in r.values())
        stats,_=score_transfer(clean,als,hole,1,r,POLICIES,ims[1])
        assert stats['candidate']['coverage']==0 and stats['candidate']['abstention']==1
        assert stats['candidate']['error']['pixels']==0


def test_cache_refuses_changed_input_model_mask_and_excluded_matches(tmp_path):
    ims,allowed,hole,offsets=fixture();clean,als=masked_inputs(ims,allowed,hole,1)
    path=tmp_path/'pair.npz';match=oracle(offsets)
    cached_pair(path,0,1,clean,als,match,{'oracle':1})
    changed=[x.copy() for x in clean];changed[0][20,20]=0
    for rgb,masks,model in [(changed,als,{'oracle':1}),(clean,als,{'oracle':2})]:
        with pytest.raises(RuntimeError,match='signature changed'):cached_pair(path,0,1,rgb,masks,match,model)
    ma=[x.copy() for x in als];ma[0][20,20]=False
    with pytest.raises(RuntimeError,match='signature changed'):cached_pair(path,0,1,clean,ma,match,{'oracle':1})
    def excluded(*args):return np.array([[0.,0.]]),np.array([[0.,0.]]),{}
    with pytest.raises(ValueError,match='excluded pixel'):cached_pair(tmp_path/'invalid.npz',0,1,clean,als,excluded,{})
    assert not (tmp_path/'invalid.npz').exists()


def test_registration_rejects_unsanitized_input(tmp_path):
    ims,allowed,hole,offsets=fixture()
    with pytest.raises(ValueError,match='already zeroed'):
        register_sequence(ims,allowed,1,140,'hidden',tmp_path,oracle(offsets),{})


def test_model_admission_rejects_preprocessing_only_and_stale_validation(tmp_path):
    path=tmp_path/'certificate.json'
    path.write_text(json.dumps({'passed':True,'signature':certificate_signature(),'actual_model_executed':False}))
    with pytest.raises(RuntimeError,match='Preprocessing-only'):verify_certificate(path)
    path.write_text(json.dumps({'passed':True,'signature':{}}))
    with pytest.raises(RuntimeError,match='stale'):verify_certificate(path)


def test_evidence_is_immutable_on_resume(tmp_path):
    path=tmp_path/'evidence.npz';evidence={'mask':np.zeros((8,8),bool)}
    save_evidence(path,evidence);save_evidence(path,evidence)
    evidence['mask'][1,1]=True
    with pytest.raises(RuntimeError,match='differs'):save_evidence(path,evidence)
