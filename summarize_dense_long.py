"""V7 length ablation with frozen controls and independently checked results."""
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from benchmark import ROOT,digest
from dense_experiment import immutable_json
from summarize_dense import aggregate
from validate_dense_run_v2 import checked_ref,load_npz,accepted,require


def subset_stats(evidence,subset):
    n=int(subset.sum());out=dict(pixels=n)
    if 'absolute_rgb_error_sum_on_mask' in evidence:
        values=evidence['absolute_rgb_error_sum_on_mask'][subset[evidence['mask']]].astype(float)/3
        out.update(mae_255=float(values.mean()) if n else None,
                   over_20_pixels=int((values>20).sum()),over_20_fraction=float((values>20).mean()) if n else None)
    return out


def main():
    root=ROOT/'sequence_v7';plan=json.loads((root/'execution_plan.json').read_text())
    auditpath=root/'audit_001.json';audit=json.loads(auditpath.read_text())
    require(audit['passed'] and audit['checked_regions']==5 and audit['plan_sha256']==digest(root/'execution_plan.json'),'missing complete audit')
    verified={r['region']:r['record_sha256'] for r in audit['results']};records=[];changes={}
    for job in plan['cases'][0]['jobs']:
        path=root/'results/gironde_0'/(job['region']+'.json');require(digest(path)==verified[job['region']],'record changed after audit')
        r=json.loads(path.read_text());records.append(r);changes[job['region']]={}
        for method,data in r['methods'].items():
            new=load_npz(checked_ref(ROOT,data['evidence']));old=load_npz(checked_ref(ROOT,job['controls'][method]['evidence']))
            require(np.array_equal(new['mask'],old['mask']),'comparison mask differs');mask=new['mask'];policies={}
            for name,p in plan['policies'].items():
                a=accepted(new,mask,p);b=accepted(old,mask,p)
                policies[name]=dict(gained=subset_stats(new,a&~b),lost=subset_stats(old,b&~a),
                    common_new=subset_stats(new,a&b),common_old=subset_stats(old,a&b),
                    mask_pixels=int(mask.sum()),new_abstained_pixels=int((mask&~a).sum()))
            changes[job['region']][method]=policies
    methods={}
    for method in records[0]['methods']:
        methods[method]={}
        for version,key in [('eleven_frames','methods'),('five_frame_control','controls')]:
            groups={}
            for name,regions in [('footprint',['footprint']),('large_hidden',['hidden_front','hidden_rear']),('local_hidden',['local_front','local_rear'])]:
                groups[name]=aggregate([(r,r[key][method]) for r in records if r['region'] in regions])
            methods[method][version]=groups
    footprints={}
    foot=next(r for r in records if r['region']=='footprint')
    for method,g in foot['graphs'].items():
        footprints[method]=dict(connected_frames=g['connected_frames'],accepted_edges=g.get('edges',0),
            optimizer_success=g.get('optimizer_success',False),optimizer_message=g.get('optimizer_message'),
            donors=[dict(name=n,accepted=r['accepted'],reason=r['reason'],graph_degree=r.get('graph_degree')) for n,r in zip(foot['source_order'],foot['methods'][method]['registrations'])])
    result=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),status='completed_development_length_ablation',
        case='gironde_0',region_method_runs=10,plan_sha256=digest(root/'execution_plan.json'),audit_sha256=digest(auditpath),
        summarizer_sha256=digest(Path(__file__)),methods=methods,changes=changes,footprint_graphs=footprints,
        raw_pair_counts=dict(per_region=25,preserved_central_per_region=7,unique_new_learned_pairs=42),promotion=False,
        interpretation_limits=['Actual footprint has no evaluation truth.',
            'Added agreeing donors share the same model and fitted graph; agreement is not independent correctness.',
            'Untextured asphalt admits low RGB errors at incorrect positions.',
            'All regions are development; no held-out test frames used.'])
    immutable_json(root/'summary.json',result)
    print(json.dumps(dict(methods=methods,footprint_graphs=footprints),indent=2))


if __name__=='__main__':main()
