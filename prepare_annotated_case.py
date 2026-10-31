"""Compile a reviewed per-frame annotation into immutable benchmark masks."""
import argparse,json,hashlib
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from repair import polygon_mask,nadir
from benchmark import ROOT,digest,save_json

def prepare(path,preview=None,freeze=False):
 annotation=json.loads(path.read_text());size=annotation['projection']['size'];shape=(size,size,3);case=annotation['scene_id'];folder=ROOT/'benchmark_v3/prepared'/case;folder.mkdir(parents=True,exist_ok=True)
 split=json.loads((ROOT/'dataset_v1/splits_v1.json').read_text())['collections'][annotation['collection']]
 assert split['split']==annotation['split']=='train','This annotation path only admits frozen training collections'
 existing=folder/'manifest.json'
 if existing.exists() and json.loads(existing.read_text())['annotation_sha256']!=digest(path):raise RuntimeError('Frozen annotation changed; create a new version')
 compiled={};counts={};out=Image.new('RGB',(1500,1000));draw=ImageDraw.Draw(out)
 for k,name in enumerate(annotation['frame_order']):
  frame=annotation['frames'][name];source=ROOT/frame['image'];assert digest(source)==frame['sample_sha256']
  for category in ['ego','ground']:
   for poly in frame[category]:
    xy=np.asarray(poly);assert len(xy)>=3 and np.isfinite(xy).all() and (xy>=0).all() and (xy<=1).all()
  ego=polygon_mask(shape,frame['ego']);ground=polygon_mask(shape,frame['ground']);allowed=ground&~ego
  assert allowed.sum()>10000;compiled[name+'_ego']=ego;compiled[name+'_ground']=ground;compiled[name+'_allowed']=allowed;counts[name]={'ego':int(ego.sum()),'allowed':int(allowed.sum())}
  if preview:
   im=nadir(np.array(Image.open(source)),size,annotation['projection']['fov']);colored=im.copy();colored[allowed]=(colored[allowed]*.65+[0,90,0]).clip(0,255);colored[ego]=(colored[ego]*.55+[100,0,0]).clip(0,255)
   if name=='target':
    for region,polys in annotation['regions'].items():
     m=polygon_mask(shape,polys)&allowed;colored[m]=(im[m]*.4+[30,70,130]).clip(0,255)
   im=Image.fromarray(colored).resize((500,500));out.paste(im,(k%3*500,k//3*500));draw.text((k%3*500+5,k//3*500+5),name,fill='white')
 footprint=compiled['target_ego'];compiled['region_footprint']=footprint
 region_counts={'footprint':{'pixels':int(footprint.sum()),'ratio_to_footprint':1.}}
 for name,polys in annotation['regions'].items():
  raw=polygon_mask(shape,polys);mask=raw&compiled['target_allowed'];assert mask.sum()>1000
  compiled['region_'+name]=mask;region_counts[name]={'pixels':int(mask.sum()),'ratio_to_footprint':float(mask.sum()/footprint.sum()),'fraction_clipped':float(1-mask.sum()/raw.sum())}
 if preview:out.save(preview)
 if freeze:
  assert annotation['status']=='reviewed_frozen_before_scoring'
  if existing.exists():
   prior=json.loads(existing.read_text());assert digest(folder/'masks.npz')==prior['mask_sha256']
   stored=np.load(folder/'masks.npz');assert set(stored.files)==set(compiled)
   for key,value in compiled.items():assert np.array_equal(stored[key],value),key
   print(json.dumps({'scene':case,'frozen':True,'unchanged':True}));return
  np.savez_compressed(folder/'masks.npz',**compiled)
  save_json(existing,{'scene_id':case,'collection':annotation['collection'],'annotation_path':str(path.relative_to(ROOT)),'annotation_sha256':digest(path),'mask_sha256':digest(folder/'masks.npz'),'frames':counts,'regions':region_counts,'evaluation_status':'not_run','split':'train','no_under_car_ground_truth':True})
 print(json.dumps({'scene':case,'frames':counts,'regions':region_counts,'frozen':freeze}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('annotation');p.add_argument('--preview');p.add_argument('--freeze',action='store_true');a=p.parse_args();prepare(Path(a.annotation).resolve(),a.preview,a.freeze)
