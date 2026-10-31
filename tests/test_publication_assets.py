import hashlib
import io
import tarfile
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from fetch_assets import safe_target, extract_verified

def archive(name, data, kind=tarfile.REGTYPE):
    b=io.BytesIO()
    with tarfile.open(fileobj=b,mode='w:gz') as t:
        i=tarfile.TarInfo(name);i.type=kind;i.size=len(data) if kind==tarfile.REGTYPE else 0
        t.addfile(i,io.BytesIO(data) if kind==tarfile.REGTYPE else None)
    return b.getvalue()

def test_traversal_and_symlink_escape_rejected(tmp_path):
    for name in ['../outside','/absolute','a/../../escape','bad\\path']:
        with pytest.raises(ValueError): safe_target(tmp_path,name)
    (tmp_path/'link').symlink_to(tmp_path.parent)
    with pytest.raises(ValueError): safe_target(tmp_path,'link/escape')

def test_corruption_and_links_rejected_before_writing(tmp_path):
    record=dict(path='payload',size=3,sha256=hashlib.sha256(b'abc').hexdigest())
    with pytest.raises(ValueError): extract_verified(archive('payload',b'bad'),[record],tmp_path)
    with pytest.raises(ValueError): extract_verified(archive('payload',b'',tarfile.SYMTYPE),[record],tmp_path)
    assert not (tmp_path/'payload').exists()

def test_resume_and_existing_data_preserved(tmp_path,monkeypatch):
    import fetch_assets
    monkeypatch.setattr(fetch_assets.shutil,'disk_usage',lambda _:type('Usage',(),{'free':20*2**30})())
    record=dict(path='nested/payload',size=3,sha256=hashlib.sha256(b'abc').hexdigest());payload=archive(record['path'],b'abc')
    extract_verified(payload,[record],tmp_path);extract_verified(payload,[record],tmp_path)
    p=tmp_path/record['path'];p.write_bytes(b'own')
    with pytest.raises(ValueError):extract_verified(payload,[record],tmp_path)
    assert p.read_bytes()==b'own'
