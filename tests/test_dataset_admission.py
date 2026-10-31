import json, sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_admission import plan, freeze_ledger, effective_splits, digest


def row(pid, collection, x, y=45):
    return dict(id=pid, collection=collection, longitude=x, latitude=y)


def frozen(**labels):
    return dict(collections={c:dict(group=c,split=s) for c,s in labels.items()}, targets=[])


def test_two_distant_points_in_new_route_bridge_train_and_test():
    f = frozen(a='train', b='test')
    result = plan([row('a','a',0), row('b','b',1), row('n0','new',.01), row('n1','new',1.01)], f)
    assert result['quarantined_collections'] == ['a','b','new']
    assert not result['additions'] and not result['targets']
    assert f['collections']['b']['split'] == 'test'


def test_transitive_new_routes_inherit_frozen_split():
    result = plan([row('a','a',0),row('x','x',.05),row('y','y',.1)],frozen(a='train'))
    assert result['additions']['x']['split'] == result['additions']['y']['split'] == 'train'
    assert result['effective_groups']['a'] == result['effective_groups']['y']
    assert not result['quarantined_collections']


def test_preexisting_group_membership_propagates_conflicts():
    f=frozen(a='train',b='train',c='test'); f['collections']['b']['group']='a'
    result=plan([row('a','a',0),row('b','b',1),row('c','c',2),row('x0','x',1.01),row('x1','x',2.01)],f)
    assert result['quarantined_collections']==['a','b','c','x']


def test_new_observations_of_old_collections_can_reveal_conflict():
    result=plan([row('a','a',0),row('b','b',.01)],frozen(a='train',b='test'))
    assert result['metrics']['newly_observed']==0
    assert result['quarantined_collections']==['a','b']


def test_isolated_assignment_and_target_choice_are_order_independent():
    rows=[row('x0','x',0),row('x1','x',.01),row('y','y',2)]
    a,b=plan(rows,frozen()),plan(list(reversed(rows)),frozen())
    assert a==b and len(a['targets'])==2


def test_baseline_development_neighborhood_cannot_enter_holdout():
    result=plan([row('new','new',0)],frozen(),[dict(collection='old',coordinates=[.01,45])])
    assert result['additions']['new']['split']=='development'


def test_ledger_preserves_base_and_excludes_quarantined_targets(tmp_path):
    base=frozen(a='train',b='test');base['targets']=[dict(id='p',collection='a',group='a',split='train')]
    path=tmp_path/'splits_v1.json';path.write_text(json.dumps(base));before=path.read_bytes()
    addition=plan([row('a','a',0),row('b','b',1),row('x0','x',.01),row('x1','x',1.01)],base)
    ledger=dict(base_sha256=digest(path),prior_ledgers=[],plan=addition)
    dest=tmp_path/'admissions/0001.json';freeze_ledger(dest,ledger);stamp=dest.stat().st_mtime_ns
    freeze_ledger(dest,ledger);assert dest.stat().st_mtime_ns==stamp
    merged=effective_splits(tmp_path)
    assert path.read_bytes()==before and merged['collections']==base['collections'] and not merged['targets']
    with pytest.raises(RuntimeError,match='Immutable'):
        freeze_ledger(dest,dict(ledger,changed=True))
    changed=dict(base);changed['targets']=[];path.write_text(json.dumps(changed))
    with pytest.raises(AssertionError,match='Frozen base changed'):
        effective_splits(tmp_path)


def test_duplicate_ids_and_reassignment_are_rejected():
    with pytest.raises(ValueError,match='Duplicate'):
        plan([row('same','a',0),row('same','b',1)],frozen())
