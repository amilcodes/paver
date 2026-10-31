import json,urllib.request,pathlib,hashlib,io,concurrent.futures,time
from datetime import datetime,timezone
from PIL import Image
ROOT=pathlib.Path(__file__).resolve().parent/'benchmark_v2';ROOT.mkdir(exist_ok=True)
selection=json.loads((ROOT/'selection.json').read_text())['target_ids']
def get(url):
 for trial in range(3):
  try:
   with urllib.request.urlopen(url,timeout=60) as r:return r.read()
  except Exception:
   if trial==2:raise
   time.sleep(1)
def fetch(case,pid):
 folder=ROOT/'data'/case;folder.mkdir(parents=True,exist_ok=True)
 def meta(pid):
  p=folder/f'{pid}.json'
  if not p.exists():p.write_bytes(get(f'https://panoramax.ign.fr/api/pictures/{pid}'))
  return json.loads(p.read_text())
 target=meta(pid);items={'target':target}
 for direction in ['prev','next']:
  current=target
  for i in [1,2]:
   link=next((l for l in current['links'] if l['rel']==direction),None)
   if link is None:break
   current=meta(link['id'])
   if current['collection']!=target['collection']:raise ValueError('Cross-collection neighbor')
   items[f'{direction}{i}']=current
 records=[]
 for name,d in items.items():
  if d['properties'].get('license')!='etalab-2.0':raise ValueError('Unexpected license')
  file=folder/f'{name}.jpg';recordpath=folder/f'{name}_record.json'
  if file.exists() and recordpath.exists():records.append(json.load(open(recordpath)));continue
  data=get(d['assets']['hd']['href'])
  im=Image.open(io.BytesIO(data));original=im.size;im.draft('RGB',(6144,3072));im=im.convert('RGB')
  if im.width!=2*im.height:raise ValueError('Not a full sphere')
  if im.width>6144:im=im.resize((6144,3072),Image.Resampling.LANCZOS)
  im.save(file,quality=93,subsampling=0)
  record={'name':name,'id':d['id'],'collection':d['collection'],'sample_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'original_sha256':hashlib.sha256(data).hexdigest(),'original_size':original,'sample_size':im.size,'metadata_file':f'{d["id"]}.json','asset_url':d['assets']['hd']['href'],'producer':d['properties'].get('geovisio:producer'),'capture_time':d['properties']['datetime'],'license':'etalab-2.0','coordinates':d['geometry']['coordinates']}
  recordpath.write_text(json.dumps(record,indent=2));records.append(record)
  print(case,name,'downloaded',len(data),flush=True)
 manifest={'fetched_at_utc':datetime.now(timezone.utc).isoformat(),'license_url':'https://ia.numerique.gouv.fr/licence-ouverte-open-licence/','images':records,'modifications':'Public blurred imagery resized to maximum 6144px width and JPEG re-encoded at quality93, subsampling0.'}
 (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 return case,len(records)
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
 futs=[pool.submit(fetch,k,v) for k,v in selection.items()]
 for f in concurrent.futures.as_completed(futs):
  try:print('DONE',f.result(),flush=True)
  except Exception as e:print('ERROR',repr(e),flush=True)
