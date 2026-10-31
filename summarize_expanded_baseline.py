"""Summarize audited outputs without merging correlated corruption families."""
import json
from collections import defaultdict
from benchmark import ROOT,digest
from run_dense import read_ref,ref
from dense_experiment import immutable_json

RUN=ROOT/'controlled_v2/baseline_v1'


def weighted(rows,value,count):
    n=sum(count(r) for r in rows)
    return sum((value(r) or 0)*count(r) for r in rows)/n if n else None


def aggregate(rows):
    n=sum(r['extended']['mask_pixels'] for r in rows);k=sum(r['metrics']['accepted_pixels'] for r in rows)
    omega=sum(r['extended']['mask_solid_angle_sr'] for r in rows);used=sum(r['extended']['accepted_solid_angle_sr'] for r in rows)
    good=sum(r['extended']['accepted_rgb_mae_le_threshold_pixels']['20'] for r in rows)
    by_anchor=defaultdict(list)
    for r in rows:by_anchor[r['scene']].append(r)
    anchor_coverages=[sum(r['metrics']['accepted_pixels'] for r in a)/sum(r['extended']['mask_pixels'] for r in a) for a in by_anchor.values()]
    out=dict(conditions=len(rows),anchors=len(by_anchor),mask_pixels=n,accepted_pixels=k,coverage=k/n,abstention=1-k/n,
        equal_anchor_mean_coverage=sum(anchor_coverages)/len(anchor_coverages),zero_coverage_conditions=sum(r['metrics']['accepted_pixels']==0 for r in rows),
        mask_solid_angle_sr=omega,accepted_solid_angle_sr=used,angular_coverage=used/omega,angular_abstention=1-used/omega,
        rgb_mae_255=weighted(rows,lambda r:r['metrics']['error']['mae_255'],lambda r:r['metrics']['accepted_pixels']),
        accepted_over_20_fraction=1-good/k if k else None,fraction_of_mask_with_rgb_mae_le_20=good/n,
        maximum_condition_p90_rgb=max((r['metrics']['error']['p90_rgb'] for r in rows if r['metrics']['accepted_pixels']),default=None),
        gradient_mae=weighted(rows,lambda r:r['metrics']['gradient']['gradient_mae'],lambda r:r['metrics']['gradient']['pixels']),
        gradient_pixels=sum(r['metrics']['gradient']['pixels'] for r in rows),
        edge_orientation_error=weighted(rows,lambda r:r['metrics']['gradient']['orientation_error'],lambda r:r['metrics']['gradient']['edge_pixels']),
        edge_pixels=sum(r['metrics']['gradient']['edge_pixels'] for r in rows),
        angular_rgb_mae_255=weighted(rows,lambda r:r['extended']['angular_rgb_mae_255'],lambda r:r['extended']['accepted_solid_angle_sr']),
        angular_fraction_with_rgb_mae_le_20=weighted(rows,lambda r:r['extended']['angular_rgb_mae_le_20_fraction_of_mask'],lambda r:r['extended']['mask_solid_angle_sr']),strata={})
    for label in ['textured','low_texture']:
        values=[r['extended']['strata'][label] for r in rows];total=sum(v['mask_pixels'] for v in values);chosen=sum(v['accepted_pixels'] for v in values)
        out['strata'][label]=dict(mask_pixels=total,accepted_pixels=chosen,coverage=chosen/total if total else None,
            rgb_mae_255=weighted(values,lambda v:v['rgb_mae_255'],lambda v:v['accepted_pixels']),
            accepted_over_20_fraction=weighted(values,lambda v:v['over_20_fraction'],lambda v:v['accepted_pixels']))
    if 'same_pixel_controls' in rows[0]:
        out['same_pixel_controls']={}
        for control in ['as_damaged','telea_r3']:
            values=[r['same_pixel_controls'][control] for r in rows]
            out['same_pixel_controls'][control]=dict(pixels=sum(v['pixels'] for v in values),
                source_mae_255=weighted(values,lambda v:v['source_mae_255'],lambda v:v['pixels']),
                control_mae_255=weighted(values,lambda v:v['control_mae_255'],lambda v:v['pixels']),
                **{key:sum(v[key] for v in values) for key in ['source_better_pixels','source_worse_pixels','equal_pixels']})
    return out


def summarize():
    pp=RUN/'execution_plan.json';plan=json.loads(pp.read_text());ap=RUN/'audit_001.json';audit=json.loads(ap.read_text())
    if audit.get('passed') is not True or audit['plan_sha256']!=digest(pp):raise RuntimeError('Complete matching audit required')
    for p,h in plan['signature'].items():read_ref(dict(path=p,sha256=h))
    hashes={(r['scene'],r['condition']):r['record_sha256'] for r in audit['conditions']}
    groups=defaultdict(list);natural=[];condition_refs=[];diagnostics=[];masks=[]
    for scene in plan['scenes']:
        folder=RUN/scene['scene'];manifest=json.loads(read_ref(scene['manifest']).read_text())
        masks.extend(dict(scene=scene['scene'],area=scene['area'],**m) for m in manifest['mask_specs'])
        for job in scene['jobs']:
            gp=folder/'groups'/(job['key']+'.json');g=json.loads(gp.read_text())
            diagnostics.append(dict(scene=scene['scene'],job=job['key'],record=ref(gp),graphs={method:dict(optimizer_success=v.get('optimizer_success'),
                accepted_donors=sum(r.get('accepted') is True for r in g['methods'][method]['registrations'])) for method,v in g['graphs'].items()}))
            if job['kind']=='natural':
                natural.append(dict(scene=scene['scene'],area=scene['area'],record=ref(gp),methods={m:dict(policies=d['policies'],extended=d['extended']) for m,d in g['methods'].items()}))
        for spec in scene['conditions']:
            path=folder/'conditions'/(spec['key']+'.json')
            if digest(path)!=hashes[(scene['scene'],spec['key'])]:raise RuntimeError('Condition changed after audit')
            s=json.loads(path.read_text());g=json.loads(read_ref(s['geometry']).read_text());condition_refs.append(ref(path));rows=[]
            for method,d in g['methods'].items():
                for policy,v in d['policies'].items():rows.append((method,policy,dict(scene=scene['scene'],metrics=v,extended=d['extended'][policy],same_pixel_controls=s['same_pixel_controls'][method][policy])))
            for method,d in s['baselines'].items():rows.append((method,'full_output_control',dict(scene=scene['scene'],metrics=d['metrics'],extended=d['extended'])))
            for method,policy,row in rows:
                for severity in [spec['severity'],'both']:
                    groups[(scene['area'],spec['family'],severity,method,policy)].append(row)
    aggregates=[dict(area=a,family=f,severity=s,method=m,policy=p,**aggregate(rows)) for (a,f,s,m,p),rows in sorted(groups.items())]
    result=dict(status='audited_expanded_baseline_not_promoted',plan=ref(pp),audit=ref(ap),anchors=6,conditions=144,distinct_controlled_masks=72,natural_masks=6,
        method_evaluations=156,aggregates=aggregates,natural=natural,diagnostics=diagnostics,masks=masks,condition_records=condition_refs,
        original_controls=plan['comparison_controls'],
        limitations=['All six additional anchors are reused development/train areas; no independent generalization result.',
            'No change to matcher, geometry, source ranking or acceptance policy. This establishes broader controls, not a model improvement.',
            'Natural footprint measurements quantify source support only; there is no clean under-car target truth.',
            'Four reviewed donors per anchor. Other metadata graph links remain unverified.',
            'Disc/blur/clipping/JPEG share donor predictions. Correlated masks, conditions, neighbors and pixels are not independent trials.',
            'RGB thresholds are appearance diagnostics, not calibrated geometry. Check texture/gradient errors and same-pixel controls.',
            'Angular coverage weights the same controlled patches. They are not angularly equivalent to the central car cone.',
            'Maximum condition p90 is a worst-condition diagnostic, not a pooled p90. Exact per-condition tails are in linked records.'])
    immutable_json(RUN/'summary.json',result)
    print(json.dumps(dict(anchors=6,conditions=144,geometry_groups=78,summary=str((RUN/'summary.json').relative_to(ROOT)))))
    return result


if __name__=='__main__':summarize()
