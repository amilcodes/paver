"""Compact summaries; pixels/conditions are correlated, not independent trials."""
import json
from collections import defaultdict
from benchmark import ROOT,digest
from dense_experiment import immutable_json
from run_dense import read_ref

OUT=ROOT/'controlled_v1'


def summarize():
    plan=json.loads((OUT/'execution_plan.json').read_text());audit=json.loads((OUT/'audit_001.json').read_text())
    if audit.get('passed') is not True or audit['plan_sha256']!=digest(OUT/'execution_plan.json'):raise RuntimeError('Complete audited battery required')
    grouped=defaultdict(list);details=[];masks=[]
    for area in plan['areas']:
        specs={s['key']:s for s in area['mask_specs']}
        for spec in area['mask_specs']:masks.append(dict(area=area['area'],**spec))
        for spec in area['scenarios']:
            path=OUT/area['area']/'scenarios'/(spec['key']+'.json');s=json.loads(path.read_text());g=json.loads(read_ref(s['geometry']).read_text())
            rows=[]
            for method,data in g['methods'].items():
                for policy,v in data['policies'].items():rows.append((method,policy,v,data['appearance'][policy]))
            for method,data in s['baselines'].items():v=data['metrics'];rows.append((method,'full_output_control',v,v['appearance']))
            for method,policy,v,appearance in rows:
                row=dict(area=area['area'],family=spec['family'],severity=spec['severity'],location=spec['location'],
                    method=method,policy=policy,mask_pixels=specs[spec['mask_key']]['pixels'],accepted_pixels=v['accepted_pixels'],
                    coverage=v['coverage'],abstention=v['abstention'],rgb_mae_255=v['error']['mae_255'],
                    over_20_fraction=v['error']['over_20_fraction'],gradient=v['gradient'],appearance=appearance)
                details.append(row);grouped[(area['area'],spec['family'],method,policy)].append(row)
    aggregates=[]
    for (area,family,method,policy),rows in grouped.items():
        n=sum(r['mask_pixels'] for r in rows);a=sum(r['accepted_pixels'] for r in rows)
        good={str(t):sum(r['appearance']['accepted_rgb_mae_le_threshold_pixels'][str(t)] for r in rows) for t in [10,20]}
        aggregates.append(dict(area=area,family=family,method=method,policy=policy,conditions=len(rows),mask_pixels=n,
            accepted_pixels=a,coverage=a/n,abstention=1-a/n,rgb_mae_255=sum((r['rgb_mae_255'] or 0)*r['accepted_pixels'] for r in rows)/a if a else None,
            fraction_of_mask_rgb_mae_le_threshold={k:v/n for k,v in good.items()},
            accepted_rgb_mae_gt_20_fraction=1-good['20']/a if a else None,
            accepted_textured_fraction=sum(r['appearance']['accepted_textured_pixels'] for r in rows)/a if a else None))
    result=dict(version=1,plan_sha256=digest(OUT/'execution_plan.json'),audit_sha256=digest(OUT/'audit_001.json'),
        status='audited_baseline_not_promoted',anchors=3,reviewed_patch_locations=6,scenarios=72,distinct_geometry_masks=36,
        aggregates=aggregates,conditions=details,masks=masks,
        cautions=['Only three previously used anchors; no held-out generalization result.',
            'RGB threshold success is an appearance measurement, not proof of geometric accuracy; asphalt can have low error despite misalignment.',
            'Disc, blur, clipping and JPEG share masks and identical source-only predictions. Their outputs are correlated; as-damaged controls differ.',
            'Natural occluded pixels have no clean reference. Synthetic mask sizes are recorded relative to the real footprint.',
            'Only four annotated donors per anchor admitted so far; the full247-node graph is a retrieval foundation.'],
        next_experiment='Use visible-only failures to predeclare one orientation/sampling-aware matcher treatment, with unchanged gates and all three-area controls. Expand reviewed donors/targets before claiming network-scale performance.')
    immutable_json(OUT/'summary.json',result)
    print(json.dumps(dict(anchors=3,scenarios=72,geometry_masks=36,summary='controlled_v1/summary.json')))
    return result


if __name__=='__main__':summarize()
