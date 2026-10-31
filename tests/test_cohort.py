import json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cohort import choose_targets
from acquire_admitted import pending_targets
from dataset_admission import digest


def test_selection_retains_cached_and_diversifies_without_scores():
    targets=[dict(id=c,collection=c,group=g,split='train') for c,g in [('a','g1'),('b','g1'),('c','g2'),('d','g3')]]
    a=choose_targets(dict(targets=targets),{'a'},dict(train=3))
    b=choose_targets(dict(targets=list(reversed(targets))),{'a'},dict(train=3))
    assert a==b and {t['collection'] for t in a}=={'a','c','d'}


def test_cohort_refuses_unfillable_quota():
    with pytest.raises(ValueError,match='Cannot fill'):
        choose_targets(dict(targets=[]),set(),dict(test=1))


def test_pending_skips_cached_incomplete_and_exhausted_failures(tmp_path):
    targets=[dict(id=c,collection=c,group=c,split='train',metadata_path=c+'.json') for c in ['a','b','c']]
    for t in targets:
        p=tmp_path/t['metadata_path'];p.write_text('{}');t['metadata_sha256']=digest(p)
    frozen=dict(collections={c:dict(group=c,split='train') for c in ['a','b','c']},targets=targets)
    p=tmp_path/'splits_v1.json';p.write_text(json.dumps(frozen))
    cohort=dict(base_sha256=digest(p),admission_ledgers=[],targets=targets)
    (tmp_path/'windows').mkdir();(tmp_path/'windows/a.json').write_text(json.dumps(dict(collection='a',target_id='a',complete_window=False)))
    (tmp_path/'failures').mkdir();(tmp_path/'failures/b.json').write_text(json.dumps(dict(attempts=3)))
    selected=pending_targets(tmp_path,cohort,'train',5)
    assert [t['collection'] for t in selected]==['c']
    (tmp_path/'c.json').write_text('changed')
    with pytest.raises(AssertionError):pending_targets(tmp_path,cohort,'train',5)
