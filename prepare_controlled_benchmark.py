"""Freeze a three-anchor, multi-corruption regression battery before scoring.

One already-reviewed anchor per area is a bootstrap, not247 annotated targets or
independent generalization. All mask locations inherit reviewed visible ground.
"""
import json
from pathlib import Path
import numpy as np
from benchmark import ROOT,digest,masks
from dense_experiment import immutable_json,array_hash
from controlled_occlusions import FAMILIES,SEVERITIES,mask_shape,make_mask
from run_dense import SOURCES,model_signature,ref,read_ref,save_evidence

OUT=ROOT/'controlled_v1'
AREA_CASES={'gironde':'gironde_0','nantes':'nantes_van_002','bourges':'bourges_turn_001'}


def prepare():
    cfg=json.loads((ROOT/'benchmark_v2/protocol.json').read_text());oldplan=json.loads((ROOT/'sequence_v6/execution_plan.json').read_text())
    graphpath=ROOT/'network_v1/candidate_graph.json';graph=json.loads(graphpath.read_text());neighbors={}
    for edge in graph['edges']:neighbors.setdefault(edge['target'],set()).add(edge['donor'])
    areas=[]
    for area,case in AREA_CASES.items():
        manifestpath=ROOT/'area_sets_v1'/(area+'_manifest.json');manifest=json.loads(manifestpath.read_text());indexed={r['id']:r for r in manifest['frames']}
        if area=='gironde':
            old=next(c for c in oldplan['cases'] if c['case']==case);frames=old['frames'];names=[r['name'] for r in frames]
            vehicle,known,_=masks(cfg,cfg['cases'][case]);allowed=[known[n] for n in names];basis={}
            refs=[ref(ROOT/'benchmark_v2/protocol.json'),ref(ROOT/'benchmark_v2/data'/case/'manifest.json')]
            for loc in ['front','rear']:
                path=ROOT/'benchmark_v2/results'/case/'disk_lightglue'/('hidden_'+loc+'_evidence.npz');refs.append(ref(path))
                with np.load(path,allow_pickle=False) as z:basis[loc]=z['mask']
            footprint=vehicle
        else:
            annpath=ROOT/'benchmark_v3'/('annotations_train_002.json' if area=='nantes' else 'annotations_train_001.json')
            a=json.loads(annpath.read_text());names=a['frame_order'];frames=[dict(name=n,id=a['frames'][n]['id'],**ref(ROOT/a['frames'][n]['image'])) for n in names]
            prepared=ROOT/'benchmark_v3/prepared'/case;mp=prepared/'masks.npz';pm=json.loads((prepared/'manifest.json').read_text())
            assert digest(annpath)==pm['annotation_sha256'] and digest(mp)==pm['mask_sha256']
            with np.load(mp,allow_pickle=False) as z:
                allowed=[z[n+'_allowed'] for n in names];basis={loc:z['region_large_'+loc] for loc in ['front','rear']};footprint=z['region_footprint']
            refs=[ref(annpath),ref(prepared/'manifest.json'),ref(mp)]
        target=names.index('target');target_id=frames[target]['id'];candidate_ids=neighbors.get(target_id,set())
        assert all(r['id'] in indexed and indexed[r['id']]['sample_sha256']==r['sha256'] for r in frames)
        assert all(r['id'] in candidate_ids for i,r in enumerate(frames) if i!=target),'Annotated donor outside frozen candidate graph'
        for r in frames:read_ref(r)
        arrays={f'allowed_{i}':m for i,m in enumerate(allowed)};arrays['footprint']=footprint
        maskspecs=[];scenarios=[]
        for loc in ['front','rear']:
            assert not (basis[loc]&~allowed[target]).any()
            for shape in ['disc','vehicle','strip']:
                for severity in SEVERITIES:
                    key=f'{loc}_{shape}_{severity}';mask=make_mask(basis[loc],shape,severity)
                    if mask.sum()<500:raise ValueError('Degenerate corruption mask')
                    arrays[key]=mask;maskspecs.append(dict(key=key,location=loc,shape=shape,severity=severity,pixels=int(mask.sum()),
                        ratio_to_real_footprint=float(mask.sum()/footprint.sum()),sha256=array_hash(mask)))
            for family in FAMILIES:
                for severity in SEVERITIES:scenarios.append(dict(key=f'{loc}_{family}_{severity}',location=loc,family=family,severity=severity,mask_key=f'{loc}_{mask_shape(family)}_{severity}'))
        folder=OUT/area;folder.mkdir(parents=True,exist_ok=True);maskpath=folder/'masks.npz';save_evidence(maskpath,arrays)
        reuse=[]
        if area in ['gironde','bourges']:
            old=json.loads((ROOT/'sequence_v6/results'/case/'footprint.json').read_text())
            for pair in old['correspondences']:
                if target not in [pair['i'],pair['j']]:
                    path=Path(pair['path']);reuse.append(dict(i=pair['i'],j=pair['j'],**ref(path if path.is_absolute() else ROOT/path)))
        areas.append(dict(area=area,case=case,split=manifest['split'],frames=frames,target=target,projection={k:cfg[k] for k in ['size','fov']},
            inputs=refs+[ref(manifestpath)],masks=ref(maskpath),mask_specs=maskspecs,scenarios=scenarios,reuse_donor_pairs=reuse,
            graph_donor_candidates=len(candidate_ids),annotated_donors_used=len(frames)-1,
            annotation_limit='Only original five source frames have masks; other graph neighbors are not silently admitted.'))
    sources=list(dict.fromkeys(SOURCES+['panosphere_graph.py','controlled_occlusions.py','prepare_controlled_benchmark.py','run_controlled_benchmark.py','audit_controlled_benchmark.py','summarize_controlled_benchmark.py','validate_dense_run_v2.py','test_controlled_occlusions.py']))
    plan=dict(version=1,name='controlled_v1',status='frozen_before_new_scores',signature={p:digest(ROOT/p) for p in sources},
        graph=ref(graphpath),dataset_audit=ref(ROOT/'area_sets_v1/audit_001.json'),model=model_signature(),policies=cfg['policies'],areas=areas,
        families=FAMILIES,severities=SEVERITIES,anchor_count=3,scenario_count=72,distinct_geometry_masks=36,
        benchmark_scope='Three existing development/train anchors, two reviewed visible-ground locations each. Not a held-out population result.',
        geometry_inputs='Target pixels in the entire corruption mask are zeroed before preprocessing regardless of corruption family. Raw hidden RGB is scoring-only.',
        source_only_baseline='Same mask produces identical donor reconstruction across disc/blur/clipping/jpeg tracks by design. These are correlated conditions, not independent evidence.',
        baselines=['as_damaged','telea_r3','loftr_homography','loftr_metric_free'],
        appearance_thresholds=[10,20],reliability_limit='RGB thresholds measure appearance on withheld visible ground, not a calibrated probability of correct real under-car geometry.')
    immutable_json(OUT/'execution_plan.json',plan)
    print(json.dumps(dict(anchors=3,scenarios=72,geometry_masks=36,plan_sha256=digest(OUT/'execution_plan.json'))))
    return plan


if __name__=='__main__':prepare()
