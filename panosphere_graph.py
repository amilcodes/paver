"""Sparse metadata retrieval graph, distinct from query-specific visual evidence.

A sphere has360-degree horizontal coverage: heading difference is recorded, not
a pinhole-FOV rejection. Geometry/visibility verification must exclude the target
test mask and cannot use clean hidden RGB. Candidate scores are not probabilities.
"""
import json,math
from pathlib import Path
from datetime import datetime
from benchmark import ROOT,digest
from dense_experiment import immutable_json

CONFIG=dict(max_distance_m=75.,max_time_seconds=90.,neighbors=12,sequence_backbone_steps=2,time_weight_m_per_second=.05)


def distance(a,b):
    lon1,lat1=map(math.radians,a[:2]);lon2,lat2=map(math.radians,b[:2])
    v=math.sin((lat2-lat1)/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 12742000*math.asin(min(1,math.sqrt(v)))


def build_nodes_edges(areas,config=CONFIG):
    nodes=[];edges=[]
    for area,manifest in sorted(areas.items()):
        if manifest['split'] not in ['development','train']:raise ValueError('Held-out area forbidden for development graph')
        frames=manifest['frames']
        if len({x['id'] for x in frames})!=len(frames):raise ValueError('Duplicate frame')
        nodes.extend(dict(id=r['id'],area=area,index=r['index'],collection=r['collection'],split=manifest['split'],
            coordinates=r['coordinates'],capture_time=r['capture_time'],azimuth_degrees=r.get('azimuth_degrees'),
            image_path=r['image_path'],sample_sha256=r['sample_sha256']) for r in frames)
        for a in frames:
            candidates=[];ta=datetime.fromisoformat(a['capture_time'].replace('Z','+00:00'))
            for b in frames:
                if a['id']==b['id']:continue
                d=distance(a['coordinates'],b['coordinates']);dt=abs((datetime.fromisoformat(b['capture_time'].replace('Z','+00:00'))-ta).total_seconds())
                if a['collection']!=b['collection'] or d>config['max_distance_m'] or dt>config['max_time_seconds']:continue
                step=abs(a['index']-b['index']);yaw=None
                if a.get('azimuth_degrees') is not None and b.get('azimuth_degrees') is not None:
                    yaw=abs((a['azimuth_degrees']-b['azimuth_degrees']+180)%360-180)
                candidates.append(dict(target=a['id'],donor=b['id'],area=area,distance_m=d,time_delta_seconds=dt,
                    sequence_steps=step,heading_difference_degrees=yaw,rank_cost=d+config['time_weight_m_per_second']*dt,
                    backbone=step<=config['sequence_backbone_steps'],status='candidate_not_visually_verified'))
            candidates.sort(key=lambda x:(x['rank_cost'],x['donor']))
            keep={c['donor'] for c in candidates[:config['neighbors']]}|{c['donor'] for c in candidates if c['backbone']}
            edges.extend(c for c in candidates if c['donor'] in keep)
    if len({n['id'] for n in nodes})!=len(nodes):raise ValueError('Image belongs to multiple areas')
    return nodes,edges


def build():
    areas={};references={}
    for area in ['gironde','nantes','bourges']:
        path=ROOT/'area_sets_v1'/(area+'_manifest.json');areas[area]=json.loads(path.read_text());references[area]=dict(path=str(path.relative_to(ROOT)),sha256=digest(path))
        if not areas[area]['complete_planned_run']:raise ValueError('Area incomplete')
    nodes,edges=build_nodes_edges(areas)
    result=dict(version=1,stage='metadata_candidate_graph',configuration=CONFIG,nodes=nodes,edges=edges,
        source_sha256=digest(Path(__file__)),input_manifests=references,
        interpretation='Directed donor candidates, not proof of visibility or recoverability. GPS/heading are approximate.',
        visual_verification='Query-specific masked correspondences, heldout geometry, convergence, visible-ground audit and support overlap. Never unmasked target features.',
        scope='Within-collection links only. Cross-route/date matching needs separate scene-change and split checks.')
    immutable_json(ROOT/'network_v1/candidate_graph.json',result)
    print(json.dumps(dict(nodes=len(nodes),directed_candidates=len(edges),areas={a:sum(e['area']==a for e in edges) for a in areas})))
    return result


if __name__=='__main__':build()
