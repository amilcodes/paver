import argparse,json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import dataset


def setup_catalog(monkeypatch,tmp_path):
 monkeypatch.setattr(dataset,'ROOT',tmp_path);monkeypatch.setattr(dataset,'DATA',tmp_path/'dataset_v1')
 return dataset.connection()

def add(db,pid,collection,lon,lat):
 item={'id':pid,'collection':collection,'geometry':{'coordinates':[lon,lat]},'properties':{'datetime':'2024-01-01','license':'etalab-2.0','pers:interior_orientation':{'field_of_view':360,'sensor_array_dimensions':[4096,2048]}}}
 dataset.ingest(db,item,'fixture');db.commit()

def test_split_groups_entire_routes_and_nearby_collections(monkeypatch,tmp_path):
 db=setup_catalog(monkeypatch,tmp_path)
 add(db,'p1','route-a',1,45);add(db,'p2','route-a',3,46)
 add(db,'p3','route-b',3.01,46);add(db,'p4','route-c',5,48)
 old=tmp_path/'benchmark_v2/data/old';old.mkdir(parents=True)
 (old/'target_record.json').write_text(json.dumps({'id':'baseline','collection':'old','coordinates':[1.01,45]}))
 dataset.freeze(argparse.Namespace())
 frozen=json.loads((dataset.DATA/'splits_v1.json').read_text());c=frozen['collections']
 assert c['route-a']['group']==c['route-b']['group']
 assert c['route-a']['split']==c['route-b']['split']=='development'
 assert c['route-c']['group']!=c['route-a']['group']
 before=(dataset.DATA/'splits_v1.json').read_bytes()
 with pytest.raises(RuntimeError,match='Frozen snapshot exists'):dataset.freeze(argparse.Namespace())
 assert (dataset.DATA/'splits_v1.json').read_bytes()==before

def test_bad_license_and_non_spheres_are_not_eligible(monkeypatch,tmp_path):
 db=setup_catalog(monkeypatch,tmp_path)
 for pid,lic,dims,fov in [('x','unknown',[4096,2048],360),('y','etalab-2.0',[4096,2160],90)]:
  dataset.ingest(db,{'id':pid,'collection':pid,'geometry':{'coordinates':[0,0]},'properties':{'license':lic,'pers:interior_orientation':{'field_of_view':fov,'sensor_array_dimensions':dims}}},'fixture')
 assert db.execute('SELECT sum(eligible) FROM pictures').fetchone()[0]==0

def test_cache_checksum_is_verified_before_resuming(monkeypatch,tmp_path):
 db=setup_catalog(monkeypatch,tmp_path);folder=dataset.DATA/'images';folder.mkdir()
 (folder/'x.jpg').write_bytes(b'changed');(folder/'x.json').write_text(json.dumps({'sample_sha256':'expected'}))
 item={'id':'x','properties':{'license':'etalab-2.0','pers:interior_orientation':{'field_of_view':360,'sensor_array_dimensions':[4096,2048]}}}
 with pytest.raises(RuntimeError,match='checksum mismatch'):dataset.download_image(db,item)

def test_download_resume_advances_to_next_collection_in_same_group(monkeypatch,tmp_path):
 setup_catalog(monkeypatch,tmp_path)
 dataset.save(dataset.DATA/'splits_v1.json',{'collections':{'a':{'group':'g','split':'train'},'b':{'group':'g','split':'train'}},'targets':[{'id':'p1','collection':'a','group':'g','split':'train'},{'id':'p2','collection':'b','group':'g','split':'train'}]})
 called=[]
 def fake_window(db,pid,radius,label):
  called.append(pid);dataset.save(dataset.DATA/'windows'/(label+'.json'),{'target_id':pid})
 monkeypatch.setattr(dataset,'window',fake_window)
 for _ in range(2):dataset.download(argparse.Namespace(split='train',limit=1,radius=2))
 assert called==['p1','p2']


def test_catalog_refuses_low_disk_before_database_or_network(monkeypatch):
 def blocked(extra=0):raise RuntimeError('Storage guard: preserve 8 GiB free')
 def forbidden(*args,**kwargs):raise AssertionError('Must not acquire or mutate the catalog')
 monkeypatch.setattr(dataset,'disk_guard',blocked)
 monkeypatch.setattr(dataset,'connection',forbidden)
 monkeypatch.setattr(dataset,'get',forbidden)
 with pytest.raises(RuntimeError,match='preserve 8 GiB'):
  dataset.catalog(argparse.Namespace(queries=20,ring=1))
