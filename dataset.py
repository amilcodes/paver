"""Resumable, licensed Panoramax catalog and sequence-window acquisition.

A query stratum is a geographic proposal, never a verified road label.
Frozen splits group entire capture collections and neighboring sampled locations.
"""
import argparse,hashlib,io,json,math,shutil,sqlite3,time,urllib.parse,urllib.request
from datetime import datetime,timezone
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'dataset_v1'
LICENSES={'etalab-2.0'}
REGIONS=[
 ('strasbourg',7.75,48.58),('alsace_north',7.7,48.85),('vosges',7.0,48.1),
 ('bordeaux',-.58,44.84),('gironde',-.65,44.95),('medoc',-.9,45.12),
 ('nantes',-1.55,47.22),('loire',-.75,47.32),('briere',-2.15,47.4),
 ('rennes',-1.68,48.11),('brest',-4.49,48.39),('brittany',-3.1,48.0),
 ('lyon',4.84,45.76),('annecy',6.13,45.9),('alps',6.5,45.5),
 ('paris',2.35,48.86),('ile_de_france',2.0,48.65),('normandy',-.36,49.18),
 ('lille',3.06,50.63),('amiens',2.29,49.89),('ardennes',4.72,49.76),
 ('toulouse',1.44,43.6),('gers',.58,43.65),('pyrenees',.1,43.0),
 ('montpellier',3.88,43.61),('cevennes',3.58,44.0),('camargue',4.5,43.55),
 ('marseille',5.37,43.3),('nice',7.26,43.71),('provence',5.55,43.83),
 ('clermont',3.09,45.78),('auvergne',2.85,45.55),('limoges',1.26,45.83),
 ('dijon',5.04,47.32),('bourgogne',4.5,47.2),('jura',5.91,46.9),
 ('tours',.69,47.39),('berry',2.4,47.08),('orleans',1.91,47.9),
 ('la_rochelle',-1.15,46.16),('vendee',-1.45,46.67),('dordogne',.72,45.18),
 ('ajaccio',8.74,41.92),('bastia',9.45,42.7),('corsica_inland',9.15,42.3),
 ('metz',6.18,49.12),('nancy',6.18,48.69),('champagne',4.03,49.26)]

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def connection():
 DATA.mkdir(exist_ok=True);db=sqlite3.connect(DATA/'catalog.sqlite')
 db.executescript('''CREATE TABLE IF NOT EXISTS pictures(id TEXT PRIMARY KEY,collection TEXT,longitude REAL,latitude REAL,datetime TEXT,license TEXT,producer TEXT,width INTEGER,height INTEGER,spherical INTEGER,eligible INTEGER,reason TEXT,metadata_path TEXT,metadata_sha256 TEXT,query_name TEXT);
 CREATE TABLE IF NOT EXISTS queries(name TEXT PRIMARY KEY,url TEXT,status TEXT,count INTEGER,sha256 TEXT,error TEXT,updated TEXT);
 CREATE TABLE IF NOT EXISTS downloads(id TEXT PRIMARY KEY,status TEXT,path TEXT,error TEXT,updated TEXT);''');return db

def get(url):
 # One worker; delay also applies to metadata. Retry transient failures only.
 for attempt in range(3):
  time.sleep(.7)
  try:
   req=urllib.request.Request(url,headers={'User-Agent':'PanoramaRepairResearch/0.1'})
   with urllib.request.urlopen(req,timeout=45) as response:return response.read()
  except Exception:
   if attempt==2:raise
   time.sleep(2**attempt)

def ingest(db,item,query):
 props=item['properties'];interior=props.get('pers:interior_orientation',{});dims=interior.get('sensor_array_dimensions',[0,0]);w,h=dims[:2]
 spherical=interior.get('field_of_view')==360 and w==2*h and w>0
 reason='eligible' if spherical and props.get('license') in LICENSES and w>=4096 else 'license, full-sphere, or resolution not eligible'
 path=DATA/'metadata'/(item['id']+'.json')
 if not path.exists():save(path,item)
 xy=item['geometry']['coordinates']
 db.execute('INSERT OR IGNORE INTO pictures VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(item['id'],item['collection'],xy[0],xy[1],props.get('datetime'),props.get('license'),props.get('geovisio:producer'),w,h,int(spherical),int(reason=='eligible'),reason,str(path.relative_to(DATA)),sha(path),query))

def catalog(args):
 # Metadata consumes disk too; refuse before opening a writable catalog or
 # contacting the provider, and recheck between bounded query batches.
 disk_guard(16*1024**2)
 db=connection();done=0
 for region,x,y in REGIONS:
  ring=getattr(args,'ring',0);radius=.18+.12*max(0,ring-1)
  offsets=[(0,0),(.06,.04),(-.06,-.04),(.12,0),(0,.12)] if ring==0 else [(radius,0),(-radius,0),(0,radius),(0,-radius),(.7*radius,.7*radius)]
  for dx,dy in offsets:
   name=f'{region}_{dx}_{dy}' if ring==0 else f'{region}_ring{ring}_{dx}_{dy}';row=db.execute('SELECT status FROM queries WHERE name=?',(name,)).fetchone()
   if row and row[0]=='complete':continue
   if done>=args.queries:break
   disk_guard(16*1024**2)
   bbox=[x+dx-.018,y+dy-.018,x+dx+.018,y+dy+.018]
   url='https://panoramax.ign.fr/api/search?'+urllib.parse.urlencode({'bbox':','.join(map(str,bbox)),'limit':100})
   try:
    raw=get(url);payload=json.loads(raw);items=payload.get('features',[])
    path=DATA/'queries'/(name+'.json');path.parent.mkdir(exist_ok=True);path.write_bytes(raw)
    for item in items:ingest(db,item,name)
    db.execute('INSERT OR REPLACE INTO queries VALUES(?,?,?,?,?,?,?)',(name,url,'complete',len(items),sha(path),None,now()));db.commit()
    print('CATALOG',name,len(items),flush=True)
   except Exception as e:
    db.execute('INSERT OR REPLACE INTO queries VALUES(?,?,?,?,?,?,?)',(name,url,'failed',0,None,str(e),now()));db.commit();print('FAILED',name,str(e),flush=True)
   done+=1
  if done>=args.queries:break
 summary(db)

def summary(db):
 counts={k:db.execute(q).fetchone()[0] for k,q in {
 'pictures':'SELECT count(*) FROM pictures','eligible_pictures':'SELECT count(*) FROM pictures WHERE eligible=1',
 'eligible_collections':'SELECT count(DISTINCT collection) FROM pictures WHERE eligible=1',
 'completed_queries':"SELECT count(*) FROM queries WHERE status='complete'",'failed_queries':"SELECT count(*) FROM queries WHERE status='failed'",
 'downloaded_images':"SELECT count(*) FROM downloads WHERE status='complete'"}.items()}
 counts.update(updated=now(),queries_per_ring=len(REGIONS)*5,max_supported_rings=6,geographic_proposals='France, diverse settlement and terrain regions; actual road/rig classes await annotation',not_a_population_sample=True)
 save(DATA/'summary.json',counts);print(json.dumps(counts),flush=True)

def freeze(args):
 import numpy as np
 from scipy.spatial import cKDTree
 db=connection();path=DATA/'splits_v1.json'
 if path.exists():raise RuntimeError('Frozen snapshot exists. New rows remain unassigned; create a reviewed new version.')
 rows=db.execute('SELECT id,collection,longitude,latitude FROM pictures WHERE eligible=1 ORDER BY id').fetchall()
 if not rows:raise RuntimeError('No eligible data')
 parent={r[1]:r[1] for r in rows}
 def find(x):
  while x!=parent[x]:parent[x]=parent[parent[x]];x=parent[x]
  return x
 def union(a,b):
  a,b=find(a),find(b)
  if a!=b:parent[max(a,b)]=min(a,b)
 # Earth-centered coordinates give a conservative 5-km neighborhood boundary.
 def xyz(lon,lat):
  lon,lat=math.radians(lon),math.radians(lat);return [6371000*math.cos(lat)*math.cos(lon),6371000*math.cos(lat)*math.sin(lon),6371000*math.sin(lat)]
 positions=np.array([xyz(r[2],r[3]) for r in rows]);tree=cKDTree(positions)
 for a,b in tree.query_pairs(5000,output_type="ndarray"):union(rows[a][1],rows[b][1])
 baseline=[]
 for f in (ROOT/'benchmark_v2/data').glob('*/target_record.json'):
  d=json.loads(f.read_text());baseline.append(d)
 contaminated=set()
 for d in baseline:
  if d['collection'] in parent:contaminated.add(find(d['collection']))
  for i in tree.query_ball_point(xyz(*d['coordinates'][:2]),5000):contaminated.add(find(rows[i][1]))
 collections={}
 for c in parent:
  group=find(c);v=int(hashlib.sha256(('split-v1:'+group).encode()).hexdigest()[:8],16)%10
  split='development' if group in contaminated else 'test' if v<2 else 'validation' if v<4 else 'train'
  collections[c]={'group':group,'split':split}
 # One pinned target per collection, selected by deterministic hash, not result quality.
 targets=[]
 for c in sorted(collections):
  items=[r for r in rows if r[1]==c];chosen=min(items,key=lambda r:hashlib.sha256(('target-v1:'+r[0]).encode()).hexdigest())
  targets.append(dict(id=chosen[0],collection=c,**collections[c]))
 save(path,dict(created=now(),catalog_rows=len(rows),spatial_exclusion_m=5000,collections=collections,targets=targets,
  limitations=['Spatial grouping uses sampled locations, not complete route geometries. Recheck full downloaded windows before evaluation.', 'No image scoring or tuning on test groups. New catalog rows are unassigned until reviewed.', 'Road categories and footprint/ground masks require annotation.'],baseline_development_ids=[d['id'] for d in baseline]))
 print('FROZEN',len(targets),'collections',len(set(c['group'] for c in collections.values())),'spatial groups',flush=True)

def disk_guard(extra=0):
 free=shutil.disk_usage(DATA).free
 if free-extra<8*1024**3:raise RuntimeError('Storage guard: preserve 8 GiB free')
 used=sum(p.stat().st_size for p in (DATA/'images').glob('*')) if (DATA/'images').exists() else 0
 if used+extra>8*1024**3:raise RuntimeError('Storage guard: 8 GiB image-cache cap')

def metadata(db,pid,query):
 path=DATA/'metadata'/(pid+'.json')
 if not path.exists():
  d=json.loads(get(f'https://panoramax.ign.fr/api/pictures/{pid}'));ingest(db,d,query);db.commit()
 return json.loads(path.read_text())

def download_image(db,d):
 p=d['properties'];inter=p.get('pers:interior_orientation',{});dims=inter.get('sensor_array_dimensions',[0,0])
 if p.get('license') not in LICENSES or inter.get('field_of_view')!=360 or dims[0]!=dims[1]*2:raise ValueError('Image license or sphere metadata not eligible')
 folder=DATA/'images';folder.mkdir(exist_ok=True);dest=folder/(d['id']+'.jpg');record=folder/(d['id']+'.json')
 if dest.exists() and record.exists():
  r=json.loads(record.read_text())
  if sha(dest)!=r['sample_sha256']:raise RuntimeError('Cached sample checksum mismatch')
  return r
 # Reuse verified baseline samples; do not download the same public pixels twice.
 for old in (ROOT/'benchmark_v2/data').glob('*/*_record.json'):
  r=json.loads(old.read_text())
  if r['id']==d['id']:
   sample=old.parent/(r['name']+'.jpg')
   if sha(sample)!=r['sample_sha256']:raise RuntimeError('Baseline sample checksum mismatch')
   disk_guard(sample.stat().st_size);shutil.copyfile(sample,dest);r=dict(r,reused_from=str(sample.relative_to(ROOT)))
   save(record,r);db.execute('INSERT OR REPLACE INTO downloads VALUES(?,?,?,?,?)',(d['id'],'complete',str(dest.relative_to(DATA)),None,now()));db.commit();return r
 disk_guard(200*1024**2);time.sleep(.7);tmp=folder/(d['id']+'.download');h=hashlib.sha256();total=0
 try:
  with urllib.request.urlopen(d['assets']['hd']['href'],timeout=60) as response, tmp.open('wb') as stream:
   while chunk:=response.read(1024**2):
    total+=len(chunk)
    if total>150*1024**2:raise RuntimeError('Image exceeds 150 MiB acquisition limit')
    h.update(chunk);stream.write(chunk)
  im=Image.open(tmp);original=im.size
  if im.width!=im.height*2:raise ValueError('Decoded image is not full 2:1 sphere')
  im.draft('RGB',(6144,3072));im=im.convert('RGB')
  if im.width>6144:im=im.resize((6144,3072),Image.Resampling.LANCZOS)
  part=dest.with_suffix('.part');im.save(part,format='JPEG',quality=93,subsampling=0);part.replace(dest)
  r=dict(id=d['id'],collection=d['collection'],sample_sha256=sha(dest),original_sha256=h.hexdigest(),original_bytes=total,
   original_size=original,sample_size=im.size,license=p['license'],producer=p.get('geovisio:producer'),capture_time=p.get('datetime'),
   asset_url=d['assets']['hd']['href'],metadata_sha256=sha(DATA/'metadata'/(d['id']+'.json')),coordinates=d['geometry']['coordinates'],
   modifications='Resized to maximum 6144 width and JPEG quality93/subsampling0; upstream public blur preserved')
  save(record,r);db.execute('INSERT OR REPLACE INTO downloads VALUES(?,?,?,?,?)',(d['id'],'complete',str(dest.relative_to(DATA)),None,now()));db.commit();return r
 except Exception as e:
  db.execute('INSERT OR REPLACE INTO downloads VALUES(?,?,?,?,?)',(d['id'],'failed',None,str(e),now()));db.commit();raise
 finally:tmp.unlink(missing_ok=True)

def window(db,pid,radius,label):
 target=metadata(db,pid,label);items={'target':target};seen={pid};stops=[]
 for direction in ['prev','next']:
  current=target
  for i in range(1,radius+1):
   link=next((l for l in current['links'] if l['rel']==direction),None)
   if link is None:stops.append(direction+': end of sequence');break
   current=metadata(db,link['id'],label)
   if current['collection']!=target['collection'] or current['id'] in seen:raise RuntimeError('Cross-collection or duplicate neighbor')
   items[direction+str(i)]=current;seen.add(current['id'])
 records=[]
 for name,item in items.items():records.append(dict(download_image(db,item),name=name));print('WINDOW',label,name,flush=True)
 manifest=dict(target_id=pid,collection=target['collection'],radius=radius,complete_window=len(records)==2*radius+1,images=records,stops=stops,created=now(),annotation_status='unannotated',evaluation_status='not evaluated')
 save(DATA/'windows'/(label+'.json'),manifest);return manifest

def download(args):
 db=connection();frozen=json.loads((DATA/'splits_v1.json').read_text());count=0
 targets=[t for t in frozen['targets'] if t['split']==args.split]
 existing_groups={frozen['collections'][x.stem]['group'] for x in (DATA/'windows').glob('*.json') if x.stem in frozen['collections']}
 # Give geographically different groups their first window before adding repeats.
 targets.sort(key=lambda t:(t['group'] in existing_groups,hashlib.sha256(t['group'].encode()).hexdigest(),t['collection']))
 chosen_groups=set()
 for t in targets:
  if t['group'] in chosen_groups:continue
  label=t['collection'];path=DATA/'windows'/(label+'.json')
  if path.exists():continue
  failure=DATA/'failures'/(label+'.json');attempts=json.loads(failure.read_text()).get('attempts',0) if failure.exists() else 0
  if attempts>=3:continue
  chosen_groups.add(t['group'])
  try:
   window(db,t['id'],args.radius,label);failure.unlink(missing_ok=True)
  except Exception as e:save(failure,dict(error=str(e),target=t,at=now(),attempts=attempts+1));print('FAILED',label,str(e),flush=True)
  count+=1
  if count>=args.limit:break
 summary(db)

def extend(args):
 db=connection()
 for case in args.case:
  d=json.loads((ROOT/'benchmark_v2/data'/case/'target_record.json').read_text());window(db,d['id'],args.radius,case)
 summary(db)

if __name__=='__main__':
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
 q=sub.add_parser('catalog');q.add_argument('--queries',type=int,default=20);q.add_argument('--ring',type=int,choices=range(6),default=0)
 sub.add_parser('freeze')
 q=sub.add_parser('download');q.add_argument('--limit',type=int,default=5);q.add_argument('--radius',type=int,default=2);q.add_argument('--split',choices=['train','validation','test','development'],default='train')
 q=sub.add_parser('extend');q.add_argument('--case',action='append',required=True);q.add_argument('--radius',type=int,default=5)
 a=p.parse_args();globals()[a.command](a)
