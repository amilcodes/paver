"""Integrity and spatial/temporal continuity audit of the contiguous area sets.

Does not estimate reconstruction accuracy, promise precise GPS, or label every
new frame with the anchor's obstruction class. Prior benchmarks remain frozen.
"""
import argparse,hashlib,json,math,sqlite3
from pathlib import Path
from datetime import datetime,timezone
from PIL import Image
from dataset_admission import effective_splits
from dense_experiment import immutable_json

ROOT=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read_checked(relative,checksum):
    p=(ROOT/relative).resolve()
    if not p.is_relative_to(ROOT) or sha(p)!=checksum:raise ValueError('Artifact changed: '+str(p))
    return p


def arc_length(a,b):
    # Independent atan2 cross/dot computation on the unit sphere.
    def xyz(p):
        lon,lat=map(math.radians,p[:2]);return (math.cos(lat)*math.cos(lon),math.cos(lat)*math.sin(lon),math.sin(lat))
    u,v=xyz(a),xyz(b)
    cross=(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0])
    return 6371000*math.atan2(math.sqrt(sum(c*c for c in cross)),sum(a*b for a,b in zip(u,v)))


def audit():
    out=ROOT/'area_sets_v1';effective=effective_splits(ROOT/'dataset_v1');areas={};seen=set()
    for area in ['gironde','nantes','bourges']:
        pp=out/(area+'_plan.json');mp=out/(area+'_manifest.json')
        plan=json.loads(pp.read_text());manifest=json.loads(mp.read_text());spec=plan['selection'];frames=manifest['frames']
        assert manifest['complete_planned_run'] and manifest['plan_sha256']==sha(pp)
        assert manifest['split']==effective['collections'][spec['collection']]['split']==spec['split']
        assert spec['split'] in ['development','train'] and spec['collection'] not in effective['quarantined_collections']
        assert sha(ROOT/'dataset_v1/splits_v1.json')==plan['frozen_base_split_sha256']
        for entry in plan['ledger_chain']:assert sha(ROOT/'dataset_v1/admissions'/entry['name'])==entry['sha256']
        for name,h in plan['source_sha256'].items():assert sha(ROOT/name)==h
        assert len(frames)==len(plan['frames']) and [r['id'] for r in frames]==[r['id'] for r in plan['frames']]
        assert len({r['id'] for r in frames})==len(frames)
        metadata=[];resolutions=set();producers=set();timestamps=[]
        for index,(r,frozen) in enumerate(zip(frames,plan['frames'])):
            assert r['index']==index and r['collection']==spec['collection']
            assert all(r[k]==v for k,v in frozen.items())
            image=read_checked(r['image_path'],r['sample_sha256']);meta=read_checked(r['metadata_path'],r['metadata_sha256'])
            item=json.loads(meta.read_text());props=item['properties'];orientation=props['pers:interior_orientation']
            assert item['id']==r['id'] and item['collection']==spec['collection'] and props['license']==r['license']=='etalab-2.0'
            assert orientation['field_of_view']==360 and item['geometry']['coordinates']==r['coordinates']
            assert props.get('geovisio:producer')==r['producer'] and props['datetime']==r['capture_time']
            with Image.open(image) as im:
                assert im.width==2*im.height and im.width>=4096;resolutions.add(im.size);im.verify()
            metadata.append(item);timestamps.append(datetime.fromisoformat(r['capture_time'].replace('Z','+00:00')))
            producers.add(r['producer']);seen.add(r['id'])
        for a,b,ta,tb in zip(metadata,metadata[1:],timestamps,timestamps[1:]):
            assert next(l['id'] for l in a['links'] if l['rel']=='next')==b['id']
            assert next(l['id'] for l in b['links'] if l['rel']=='prev')==a['id']
            assert tb>=ta
        if plan['full_collection']:
            assert not any(l['rel']=='prev' for l in metadata[0]['links'])
            assert not any(l['rel']=='next' for l in metadata[-1]['links'])
            collection=json.loads((out/(area+'_collection.json')).read_text())
            assert sha(out/(area+'_collection.json'))==plan['collection_metadata_sha256']
            assert len(frames)==collection['stats:items']['count']
        coords=[r['coordinates'] for r in frames];distances=[arc_length(a,b) for a,b in zip(coords,coords[1:])]
        assert math.isclose(sum(distances),plan['path_length_m'],abs_tol=1e-5)
        geo=json.loads((out/(area+'_route.geojson')).read_text());assert len(geo['features'])==len(frames)+1
        for r,f in zip(frames,geo['features'][:-1]):
            assert f['geometry']['coordinates']==r['coordinates'] and f['properties']['id']==r['id']
        assert geo['features'][-1]['geometry']['coordinates']==coords
        areas[area]=dict(verified_images=len(frames),full_collection=plan['full_collection'],split=spec['split'],
            path_length_m=sum(distances),mean_step_m=sum(distances)/max(1,len(distances)),maximum_step_m=max(distances,default=0),
            latitude_range=[min(x[1] for x in coords),max(x[1] for x in coords)],longitude_range=[min(x[0] for x in coords),max(x[0] for x in coords)],
            start_time=frames[0]['capture_time'],end_time=frames[-1]['capture_time'],resolutions=sorted(resolutions),producers=sorted(producers),
            plan_sha256=sha(pp),manifest_sha256=sha(mp),route_geojson_sha256=sha(out/(area+'_route.geojson')),
            extended_frame_annotations=False,actual_hidden_ground_truth=False)
    for version in ['sequence_v6','sequence_v7']:
        plan=json.loads((ROOT/version/'execution_plan.json').read_text())
        for name,h in plan['signature'].items():assert sha(ROOT/name)==h
    cohortpath=ROOT/'dataset_v1/cohorts/initial_100_v1.json';cohort=json.loads(cohortpath.read_text())
    heldout={t['collection'] for t in cohort['targets'] if t['split'] in ['validation','test']}
    with sqlite3.connect('file:'+str(ROOT/'dataset_v1/catalog.sqlite')+'?mode=ro',uri=True) as db:
        downloaded={x[0] for x in db.execute("SELECT DISTINCT p.collection FROM pictures p JOIN downloads d ON p.id=d.id WHERE d.status='complete'")}
    assert not heldout&downloaded
    return dict(created_at_utc=datetime.now(timezone.utc).isoformat(),passed=True,areas=areas,total_distinct_images=len(seen),
        validator_sha256=sha(Path(__file__)),cohort_sha256=sha(cohortpath),previous_experiment_sources_unchanged=True,
        heldout_images_downloaded=False,limits=['Mapped GPS/heading is not calibrated ground-truth6DoF.',
        'Continuity/image integrity and existing split safety are audited; natural occlusion correctness is not.',
        'Anchor classes are not blanket annotations for every frame; expanded areas remain development/train.'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    report=audit();immutable_json(args.output,report);print(json.dumps(report,indent=2))
