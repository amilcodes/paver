"""Audit immutable split assignments against new metadata and cached windows."""
import json,math,sqlite3
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from dataset import DATA,sha,save,now

def audit():
 split=json.loads((DATA/'splits_v1.json').read_text());assign=split['collections'];db=sqlite3.connect(DATA/'catalog.sqlite')
 rows=db.execute('SELECT id,collection,longitude,latitude,metadata_path,metadata_sha256 FROM pictures WHERE eligible=1').fetchall()
 assigned=[r for r in rows if r[1] in assign];unassigned=len(rows)-len(assigned)
 def xyz(row):
  lon,lat=math.radians(row[2]),math.radians(row[3]);return [6371000*math.cos(lat)*math.cos(lon),6371000*math.cos(lat)*math.sin(lon),6371000*math.sin(lat)]
 tree=cKDTree(np.array([xyz(r) for r in assigned]));conflicts=[];quarantine=set();count=0
 for i,j in tree.query_pairs(5000,output_type='ndarray'):
  a,b=assigned[i],assigned[j]
  if assign[a[1]]['split']!=assign[b[1]]['split']:
   count+=1;quarantine.update([a[1],b[1]])
   if len(conflicts)<100:conflicts.append([a[0],b[0]])
 failures=[];images=set();windows=[]
 for path in (DATA/'windows').glob('*.json'):
  w=json.loads(path.read_text());seen=set()
  for r in w['images']:
   im=DATA/'images'/(r['id']+'.jpg')
   if r['id'] in seen or r['collection']!=w['collection']:failures.append([path.name,'duplicate or mixed collection'])
   seen.add(r['id']);images.add(r['id'])
   if not im.exists() or sha(im)!=r['sample_sha256']:failures.append([path.name,r['id'],'image checksum mismatch'])
   if r.get('license')!='etalab-2.0':failures.append([path.name,r['id'],'unapproved license'])
  windows.append(dict(name=path.stem,images=len(w['images']),collection=w['collection'],split=assign.get(w['collection'],{}).get('split','unassigned'),annotation_status=w['annotation_status']))
 for r in rows:
  if sha(DATA/r[4])!=r[5]:failures.append([r[0],'metadata checksum mismatch'])
 out=dict(created=now(),assigned_eligible_images=len(assigned),unassigned_eligible_images=unassigned,split_counts={s:sum(x['split']==s for x in assign.values()) for s in ['development','train','validation','test']},spatial_groups=len(set(x['group'] for x in assign.values())),cross_split_pairs_within_5km=count,quarantined_collections=sorted(quarantine),conflict_examples=conflicts,cached_images=len(images),windows=windows,integrity_failures=failures,integrity_pass=not failures,spatial_split_audit_pass=count==0,evaluation_ready=False,note='Masks and verified road/rig labels are not annotated for the new cohort. Spatial audit uses known samples, not complete route geometry. Re-run after every acquisition batch; quarantine split conflicts before evaluation.')
 save(DATA/'audit.json',out);print(json.dumps({k:v for k,v in out.items() if k not in ['windows','conflict_examples']}))
 if failures or count:raise SystemExit(1)
if __name__=='__main__':audit()
