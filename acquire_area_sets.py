"""Three contiguous, geolocated real-occlusion area sets; no reconstruction tuning.

Uses the licensed shared cache without rewriting original five-frame windows,
split assignments, annotation masks or experiments. All targets were already
development/train. New metadata is audited for split conflicts before downloads.
"""
import argparse,fcntl,json,math
from pathlib import Path
from datetime import datetime,timezone
import dataset
from benchmark import ROOT,digest
from dense_experiment import immutable_json
from dataset_admission import effective_splits
from audit_effective_dataset import audit_current
from resource_guard import require_storage
from run_dense import heavy_job

OUT=ROOT/'area_sets_v1'
AREAS=[
 dict(name='gironde',collection='5240ad3e-e15d-42ed-88f6-f58502ffa5be',target='fda158d7-bc27-4f5f-940c-7a55c3b66bee',radius=150,split='development',
      anchor_observations=['camera vehicle/nadir obstruction','low-texture rural asphalt'],selection='Entire short collection, bounded by150 links each direction'),
 dict(name='nantes',collection='816fcf1f-d4ee-4ec1-9535-f66c2d608404',target='f1340f34-7b52-4313-97f6-654c2c11eb0e',radius=25,split='train',
      anchor_observations=['large off-centre van and roof equipment','moving adjacent traffic','vehicle shadows'],selection='51 consecutive frames centered on previously annotated training anchor'),
 dict(name='bourges',collection='0bf007dd-97f9-4572-9614-8b0683a0ec5e',target='1a9e9b37-1d3e-45ee-ae9f-f4035a792212',radius=25,split='train',
      anchor_observations=['opaque nadir logo over vehicle','junction/turn','walls and hedges outside flat-road model'],selection='51 consecutive frames centered on previously annotated training anchor')]


def traverse(anchor,fetch,radius):
    if radius<1:raise ValueError('Positive radius required')
    collection=anchor['collection'];seen={anchor['id']};sides={};stops={}
    for direction in ['prev','next']:
        current=anchor;chain=[]
        for step in range(1,radius+1):
            links=[l for l in current['links'] if l['rel']==direction]
            if not links:stops[direction]='collection endpoint';break
            if len(links)!=1:raise ValueError('Ambiguous temporal link')
            item=fetch(links[0]['id'])
            if item['id']!=links[0]['id'] or item['collection']!=collection or item['id'] in seen:
                raise ValueError('Duplicate, wrong-ID or cross-collection temporal link')
            seen.add(item['id']);chain.append(item);current=item
        else:stops[direction]='bounded window radius reached'
        sides[direction]=chain
    return list(reversed(sides['prev']))+[anchor]+sides['next'],stops


def distance_m(a,b):
    x1,y1=map(math.radians,a[:2]);x2,y2=map(math.radians,b[:2]);v=math.sin((y2-y1)/2)**2+math.cos(y1)*math.cos(y2)*math.sin((x2-x1)/2)**2
    return 2*6371000*math.asin(min(1,math.sqrt(v)))


def plan_area(spec,db,effective):
    dest=OUT/(spec['name']+'_plan.json')
    if dest.exists():
        p=json.loads(dest.read_text())
        if p['selection']!=spec:raise ValueError('Frozen area selection changed')
        return p
    if effective['collections'][spec['collection']]['split']!=spec['split']:raise ValueError('Frozen split mismatch')
    def fetch(pid):
        require_storage(ROOT);return dataset.metadata(db,pid,'area_sets_v1_'+spec['name'])
    anchor=fetch(spec['target']);items,stops=traverse(anchor,fetch,spec['radius'])
    frames=[]
    for i,item in enumerate(items):
        p=item['properties'];sensor=p.get('pers:interior_orientation',{});dims=sensor.get('sensor_array_dimensions',[0,0])
        if p.get('license')!='etalab-2.0' or sensor.get('field_of_view')!=360 or dims[0]!=2*dims[1]:raise ValueError('Non-eligible sphere/license')
        mp=dataset.DATA/'metadata'/(item['id']+'.json')
        frames.append(dict(index=i,id=item['id'],collection=item['collection'],capture_time=p['datetime'],
            coordinates=item['geometry']['coordinates'],azimuth_degrees=p.get('view:azimuth'),
            horizontal_accuracy_m=p.get('quality:horizontal_accuracy'),license=p['license'],producer=p.get('geovisio:producer'),
            metadata_path=str(mp.relative_to(ROOT)),metadata_sha256=digest(mp)))
    coords=[r['coordinates'] for r in frames];steps=[distance_m(a,b) for a,b in zip(coords,coords[1:])]
    record=dict(version=1,selection=spec,frames=frames,stops=stops,full_collection=all(s=='collection endpoint' for s in stops.values()),
        path_length_m=sum(steps),minimum_step_m=min(steps,default=0),maximum_step_m=max(steps,default=0),
        target_index=next(i for i,r in enumerate(frames) if r['id']==spec['target']),
        source_sha256={n:digest(ROOT/n) for n in ['acquire_area_sets.py','dataset.py','dataset_admission.py','resource_guard.py']},
        frozen_base_split_sha256=digest(dataset.DATA/'splits_v1.json'),ledger_chain=effective['ledger_chain'],
        collection_metadata_sha256=digest(OUT/(spec['name']+'_collection.json')),
        pose_status='GPS positions and metadata azimuth, not calibrated ground-truth 6DoF poses',
        pixel_ground_truth_status='Real occlusions have no clean target counterpart. Controlled corruptions must retain a separate hidden original.',
        review_status='Anchor reviewed previously; extended frames not yet individually annotated or scored')
    immutable_json(dest,record);return record


def acquire(spec,plan,db):
    dest=OUT/(spec['name']+'_manifest.json')
    if dest.exists():
        m=json.loads(dest.read_text())
        if m['plan_sha256']!=digest(OUT/(spec['name']+'_plan.json')):raise ValueError('Area plan changed')
        for r in m['frames']:
            if digest(ROOT/r['image_path'])!=r['sample_sha256']:raise ValueError('Cached area image changed')
        print('RESUME',spec['name'],len(m['frames']),flush=True);return
    rows=[]
    for r in plan['frames']:
        require_storage(ROOT);path=ROOT/r['metadata_path']
        if digest(path)!=r['metadata_sha256']:raise ValueError('Frozen source metadata differs')
        item=json.loads(path.read_text());record=dataset.download_image(db,item)
        rows.append(dict(r,**{k:v for k,v in record.items() if k not in r},image_path=str((dataset.DATA/'images'/(r['id']+'.jpg')).relative_to(ROOT))))
        print('IMAGE',spec['name'],r['index']+1,'/',len(plan['frames']),r['id'],flush=True)
    record=dict(area=spec['name'],split=spec['split'],plan_sha256=digest(OUT/(spec['name']+'_plan.json')),
        frames=rows,complete_planned_run=True,full_collection=plan['full_collection'],path_length_m=plan['path_length_m'],
        anchor_id=spec['target'],anchor_observations=spec['anchor_observations'],
        annotation_status='anchor-only annotations; extended frames await review',evaluation_status='not evaluated as an area set',
        created_at_utc=datetime.now(timezone.utc).isoformat())
    immutable_json(dest,record)
    features=[dict(type='Feature',geometry=dict(type='Point',coordinates=r['coordinates']),properties={k:r[k] for k in ['index','id','collection','capture_time','license','producer','image_path']}) for r in rows]
    features.append(dict(type='Feature',geometry=dict(type='LineString',coordinates=[r['coordinates'] for r in rows]),properties=dict(area=spec['name'],split=spec['split'],path_length_m=plan['path_length_m'])))
    immutable_json(OUT/(spec['name']+'_route.geojson'),dict(type='FeatureCollection',features=features))


def main(args):
    OUT.mkdir(exist_ok=True)
    with heavy_job(),(dataset.DATA/'.acquisition.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);effective=effective_splits(dataset.DATA);db=dataset.connection()
        selected=[s for s in AREAS if not args.area or s['name'] in args.area]
        plans=[plan_area(s,db,effective) for s in selected]
        audit=audit_current(dataset.DATA)
        print('PLAN',[(s['name'],len(p['frames']),p['full_collection'],round(p['path_length_m'],1)) for s,p in zip(selected,plans)],flush=True)
        if args.download:
            for spec,plan in zip(selected,plans):acquire(spec,plan,db)
            audit_current(dataset.DATA)
        dataset.summary(db)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--area',action='append',choices=[a['name'] for a in AREAS]);p.add_argument('--download',action='store_true')
    main(p.parse_args())
