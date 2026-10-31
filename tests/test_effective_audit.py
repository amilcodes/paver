import json,sqlite3,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit_effective_dataset import audit_current
from dataset_admission import digest


def fixture(tmp_path):
    data=tmp_path/'dataset_v1';data.mkdir();(data/'metadata').mkdir();(data/'images').mkdir();(data/'windows').mkdir()
    (data/'splits_v1.json').write_text(json.dumps(dict(collections={'a':dict(group='a',split='train'),'b':dict(group='b',split='test')},targets=[])))
    db=sqlite3.connect(data/'catalog.sqlite')
    db.execute('CREATE TABLE pictures(id,collection,longitude,latitude,metadata_path,metadata_sha256,eligible)')
    def add(pid,c,x):
        p=data/'metadata'/(pid+'.json');p.write_text(json.dumps(dict(id=pid,collection=c,properties={'license':'etalab-2.0'},geometry={'coordinates':[x,45]})))
        db.execute('INSERT INTO pictures VALUES(?,?,?,?,?,?,1)',(pid,c,x,45,str(p.relative_to(data)),digest(p)));db.commit()
    add('pa','a',0);add('pb','b',1)
    im=data/'images/pa.jpg';im.write_bytes(b'checksum-only-fixture')
    # Legacy reused records have no inline metadata hash; catalog hash is pinned.
    (data/'windows/a.json').write_text(json.dumps(dict(collection='a',complete_window=False,images=[dict(id='pa',collection='a',sample_sha256=digest(im),license='etalab-2.0')])))
    return data,db,add


def test_new_observation_quarantines_entire_route_and_does_not_silently_clear(tmp_path):
    data,db,add=fixture(tmp_path)
    assert audit_current(data)['integrity_allows_acquisition']
    add('new','a',1.01)
    with pytest.raises(RuntimeError,match='split audit failed'):audit_current(data)
    assert json.loads((data/'active_quarantine.json').read_text())['collections']==['a','b']
    db.execute("DELETE FROM pictures WHERE id='new'");db.commit()
    with pytest.raises(RuntimeError,match='split audit failed'):audit_current(data)


def test_legacy_metadata_is_verified_and_corrupt_images_block(tmp_path):
    data,db,_=fixture(tmp_path)
    assert audit_current(data)['legacy_window_metadata_verified_against_catalog']==1
    (data/'images/pa.jpg').write_bytes(b'corrupted')
    with pytest.raises(RuntimeError,match='integrity/split audit failed'):audit_current(data)
    assert not json.loads((data/'effective_audit.json').read_text())['integrity_pass']
