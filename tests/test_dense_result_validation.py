import copy,json,sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from repair import Registration
from dense_experiment import masked_inputs,score_transfer,same_pixel_error
from validate_dense_run import verify_evidence,verify_selected_donors,verify_common,checked_ref,sha,verify_certificate

POLICIES={'candidate':{'min_views':1,'max_disagreement':18},'corroborated':{'min_views':2,'max_disagreement':12},'strict':{'min_views':2,'max_disagreement':8}}


def fixture(footprint=False,all_abstain=False,offset=6):
    y,x=np.mgrid[:32,:32];g=20+5*x+2*y
    truth=np.stack([g,g*.8+10,g*.6+30],axis=2).astype(np.uint8)
    images=[(truth.astype(int)+offset).clip(0,255).astype(np.uint8),truth,
            (truth.astype(int)+2*offset).clip(0,255).astype(np.uint8)]
    hole=np.zeros((32,32),bool);hole[4:28,4:28]=True
    allowed=[np.ones((32,32),bool) for _ in images]
    clean,als=masked_inputs(images,allowed,hole,1)
    s0=hole.copy();s0[24:]=False;s1=s0.copy();s1[:,:16]=False
    regs={}
    for i,support,quality in [(0,s0,10),(2,s1,20)]:
        diagnostics=dict(accepted=not all_abstain,inliers=quality,median_reprojection_px=0.,homography=np.eye(3).tolist())
        regs[i]=Registration(None if all_abstain else np.eye(3),support,diagnostics)
    stats,evidence=score_transfer(clean,als,hole,1,regs,POLICIES,None if footprint else truth)
    return dict(evidence=evidence,reported=stats,mask=hole,truth=None if footprint else truth,
                footprint=footprint,policies=POLICIES,donor_count=2,graph={'optimizer_success':not all_abstain},
                registrations=[regs[i].diagnostics for i in [0,2]]),[images[i] for i in [0,2]],[allowed[i] for i in [0,2]]


def test_independent_metrics_and_selected_source_pixels_agree_with_production():
    args,images,allowed=fixture();audit=verify_evidence(**args)
    assert audit['policies']['candidate']['accepted_pixels']==480
    assert audit['policies']['candidate']['coverage']==pytest.approx(5/6)
    assert audit['policies']['candidate']['error']['mae_255']==9
    assert audit['coverage_error_abstention']['candidate']['accepted_rgb_mae_le_20_pixels']==480
    assert audit['policies']['candidate']['gradient']['gradient_mae']>0
    assert verify_selected_donors(args['evidence'],images,allowed,args['registrations'])['selected_donor_pixels']==480
    common=same_pixel_error(args['evidence'],args['evidence'],POLICIES)
    verify_common(args['evidence'],args['evidence'],POLICIES,common)


@pytest.mark.parametrize('corruption', ['mask','outside_count','donor_count','invalid_provenance','nan_disagreement',
                                       'error_vector','coverage','rgb','p90','gradient','nonconvergence','rejected_donor'])
def test_corruption_and_inflated_claims_fail_closed(corruption):
    args,_,_=fixture();e=args['evidence'];m=args['reported']['candidate']
    if corruption=='mask':e['mask'][5,5]=False
    elif corruption=='outside_count':e['count'][0,0]=1
    elif corruption=='donor_count':e['count'][5,5]=3
    elif corruption=='invalid_provenance':e['provenance'][5,5]=2
    elif corruption=='nan_disagreement':e['disagreement'][5,17]=np.nan
    elif corruption=='error_vector':e['absolute_rgb_error_sum_on_mask'][0]=0
    elif corruption=='coverage':m['coverage']=1.
    elif corruption=='rgb':m['error']['mae_255']=0.
    elif corruption=='p90':m['error']['p90_rgb']+=1
    elif corruption=='gradient':m['gradient']['gradient_mae']+=.1
    elif corruption=='nonconvergence':args['graph']['optimizer_success']=False
    elif corruption=='rejected_donor':args['registrations'][0]['accepted']=False
    with pytest.raises(ValueError):verify_evidence(**args)


def test_all_abstentions_have_null_error_not_zero_accuracy():
    args,images,allowed=fixture(all_abstain=True);result=verify_evidence(**args)
    assert result['policies']['candidate']['abstention']==1
    assert result['policies']['candidate']['error']['mae_255'] is None
    assert result['policies']['candidate']['gradient']['gradient_mae'] is None
    verify_selected_donors(args['evidence'],images,allowed,args['registrations'])
    args['reported']['candidate']['error']['mae_255']=0
    with pytest.raises(ValueError,match='must be null'):verify_evidence(**args)


def test_footprint_rejects_accuracy_and_error_vector_even_when_transfers_exist():
    args,_,_=fixture(footprint=True);result=verify_evidence(**args)
    assert 'error' not in result['policies']['candidate']
    assert 'accepted_rgb_mae_le_20_pixels' not in result['coverage_error_abstention']['candidate']
    args['evidence']['absolute_rgb_error_sum_on_mask']=np.zeros(args['mask'].sum(),np.uint16)
    with pytest.raises(ValueError,match='footprint truth'):verify_evidence(**args)
    del args['evidence']['absolute_rgb_error_sum_on_mask']
    args['reported']['candidate']['error']={'mae_255':0}
    with pytest.raises(ValueError,match='fields differ'):verify_evidence(**args)


def test_stored_prediction_must_come_from_selected_source_and_allowed_ground():
    args,images,allowed=fixture();e=args['evidence']
    e['prediction_on_mask'][0]+=1
    with pytest.raises(ValueError,match='selected donor'):verify_selected_donors(e,images,allowed,args['registrations'])
    e['prediction_on_mask'][0]-=1;allowed[0][4,4]=False
    with pytest.raises(ValueError,match='allowed source'):verify_selected_donors(e,images,allowed,args['registrations'])


def test_error_is_measured_even_if_donors_agree_perfectly():
    args,images,allowed=fixture(offset=30)
    # Replace both observations with the same wrong road brightness. Perfect
    # agreement remains distinct from measured hidden-ground error.
    images[1]=images[0].copy();e=args['evidence'];e['disagreement'][:]=0
    take=args['mask']&(e['count']>0);prediction=np.zeros_like(args['truth']);prediction[take]=images[0][take]
    e['prediction_on_mask']=prediction[args['mask']]
    e['absolute_rgb_error_sum_on_mask']=np.abs(prediction[args['mask']].astype(int)-args['truth'][args['mask']].astype(int)).sum(1).astype(np.uint16)
    # Generate the report using the old scoring functions, not this validator.
    from benchmark import error_metrics
    from metric_plane import gradient_error
    for name,policy in POLICIES.items():
        mask=args['mask']&(e['count']>=policy['min_views'])
        n=int(mask.sum());args['reported'][name]=dict(accepted_pixels=n,coverage=n/576,abstention=1-n/576,
              single_view_pixels=int((mask&(e['count']==1)).sum()),multi_view_pixels=int((mask&(e['count']>=2)).sum()),
              error=error_metrics(prediction,args['truth'],mask),gradient=gradient_error(prediction,args['truth'],mask))
    result=verify_evidence(**args);verify_selected_donors(e,images,allowed,args['registrations'])
    assert result['policies']['candidate']['coverage']>0.8
    assert result['coverage_error_abstention']['candidate']['accepted_rgb_mae_le_20_fraction_of_mask']==0
    assert result['coverage_error_abstention']['candidate']['accepted_rgb_mae_gt_20_pixels']==480


def test_same_pixel_comparison_rejects_falsified_score():
    args,_,_=fixture();e=args['evidence'];report=same_pixel_error(e,e,POLICIES)
    report['candidate']['control']['mae_255']=0
    with pytest.raises(ValueError,match='numeric value'):verify_common(e,e,POLICIES,report)


def test_hash_and_root_boundary_are_enforced(tmp_path):
    path=tmp_path/'artifact';path.write_bytes(b'original');ref=dict(path='artifact',sha256=sha(path))
    assert checked_ref(tmp_path,ref)==path
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash mismatch'):checked_ref(tmp_path,ref)
    with pytest.raises(ValueError,match='escapes'):checked_ref(tmp_path,dict(path='../escape',sha256='unused'))


def test_certificate_nan_cannot_pass_by_comparison_bypass(tmp_path):
    plan=dict(model={'test_fixture':True},signature={'validate_dense_model.py':'test-only'})
    certificate=dict(passed=True,actual_model_executed=True,excluded_rgb_correspondence_invariance=True,
                     signature=dict(model=plan['model'],validator_sha256='test-only',thresholds=dict(minimum_matches=32,median_error_px=1.5,p90_error_px=3.),device='mps'),
                     translation_matches=100,translation_median_error_px=float('nan'),translation_p90_error_px=0.)
    path=tmp_path/'synthetic-invalid-certificate.json';path.write_text(json.dumps(certificate))
    with pytest.raises(ValueError,match='invalid certificate error'):verify_certificate(tmp_path,dict(path=path.name,sha256=sha(path)),plan)
