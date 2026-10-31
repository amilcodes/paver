"""Paired coverage/error results, including full failures and signed controls."""
import json
from collections import defaultdict,Counter
from benchmark import ROOT,digest
from run_dense import ref,read_ref
from dense_experiment import immutable_json
from summarize_expanded_baseline import aggregate,weighted

OUT=ROOT/'sampling_v1'


def partition_aggregate(rows):
    out={}
    for label in ['common','gained','lost']:
        values=[r['partition'][label] for r in rows];n=sum(v['pixels'] for v in values);total=sum(r['extended']['mask_pixels'] for r in rows)
        out[label]=dict(pixels=n,fraction_of_mask=n/total)
        for key in ['new','control']:
            if (label=='gained' and key=='control') or (label=='lost' and key=='new'):continue
            out[label][key]=dict(mae_255=weighted(values,lambda v:v[key]['mae_255'],lambda v:v['pixels']),over_20_fraction=weighted(values,lambda v:v[key]['over_20_fraction'],lambda v:v['pixels']))
    return out


def summarize():
    plan=json.loads((OUT/'execution_plan.json').read_text());audit=json.loads((OUT/'audit_001.json').read_text());ph=digest(OUT/'execution_plan.json')
    if audit.get('passed') is not True or audit['plan_sha256']!=ph:raise RuntimeError('Complete audit required')
    for f,h in plan['signature'].items():read_ref(dict(path=f,sha256=h))
    gh={(r['scene'],r['job']):r['record_sha256'] for r in audit['groups']};ch={(r['scene'],r['condition']):r['record_sha256'] for r in audit['conditions']}
    rows=defaultdict(list);natural=[];hard=[];diagnostics=[];groups={}
    for scene in plan['scenes']:
        pairs={};states={m:Counter() for m in plan['methods']}
        for job in scene['jobs']:
            path=OUT/scene['scene']/'groups'/(job['key']+'.json')
            if digest(path)!=gh[(scene['scene'],job['key'])]:raise RuntimeError('Group changed after audit')
            g=json.loads(path.read_text());groups[(scene['scene'],job['key'])]=g
            for p in g['correspondences']:pairs[p['path']]=p
            for method,v in g['graphs'].items():
                state='disconnected' if v.get('connected_frames',0)<=1 else 'nonconverged' if v.get('optimizer_success') is not True else 'converged'
                states[method][state]+=1
            if job['kind']=='natural':natural.append(dict(scene=scene['scene'],area=scene['area'],record=ref(path),methods={m:{k:d[k] for k in ['policies','extended','control_policies','control_extended','partitions']} for m,d in g['methods'].items()}))
            if job['kind']=='hard_control':hard.append(dict(scene=scene['scene'],region=job['key'],record=ref(path),methods={m:{k:d[k] for k in ['policies','extended','control_policies','control_extended','partitions']} for m,d in g['methods'].items()}))
        diagnostics.append(dict(scene=scene['scene'],unique_pair_caches=len(pairs),accepted_pair_proposals=sum(p['proposal'].get('accepted') is True for p in pairs.values()),proposal_reasons=dict(Counter(p['proposal'].get('reason') for p in pairs.values())),graph_states={k:dict(v) for k,v in states.items()}))
        for c in scene['conditions']:
            path=OUT/scene['scene']/'conditions'/(c['spec']['key']+'.json')
            if digest(path)!=ch[(scene['scene'],c['spec']['key'])]:raise RuntimeError('Condition changed after audit')
            record=json.loads(path.read_text());g=groups[(scene['scene'],c['job'])]
            for method,d in g['methods'].items():
                for policy,v in d['policies'].items():
                    row=dict(scene=scene['scene'],metrics=v,extended=d['extended'][policy],partition=d['partitions'][policy],same_pixel_controls=record['same_pixel_controls'][method][policy],
                        control=dict(scene=scene['scene'],metrics=d['control_policies'][policy],extended=d['control_extended'][policy]))
                    for severity in [c['spec']['severity'],'both']:rows[(scene['area'],c['spec']['family'],severity,method,policy)].append(row)
    aggregates=[dict(area=a,family=f,severity=s,method=m,policy=p,treatment=aggregate(values),control=aggregate([v['control'] for v in values]),partitions=partition_aggregate(values)) for (a,f,s,m,p),values in sorted(rows.items())]
    result=dict(status='audited_sampling_ablation_no_automatic_promotion',plan=ref(OUT/'execution_plan.json'),audit=ref(OUT/'audit_001.json'),groups=126,method_evaluations=252,conditions=216,
        aggregates=aggregates,natural=natural,hard_controls=hard,diagnostics=diagnostics,
        limitations=['Same frozen dev/train masks/sources compared directly; no independent route generalization.','Only max input side changed640->960; no tiled-native crop experiment yet.',
            'Shared-mask families and neighboring frames remain correlated. Do not aggregate them as independent trials.','Natural occlusion has no target truth; projected/angular coverage measure support only.',
            'Lower appearance error does not calibrate geometric correctness. Texture strata, accepted-error tails, abstention and same-pixel controls remain required.',
            'Original Gironde footprint in natural track also represents its identical hard-road footprint control; not duplicated.'])
    immutable_json(OUT/'summary.json',result);print(json.dumps(dict(groups=126,conditions=216,summary='sampling_v1/summary.json')))


if __name__=='__main__':summarize()
