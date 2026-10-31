"""V7 artifact audit, reusing the independently tested v2 scoring validator.

Additionally replay the original v6 edge partition for every reused pair.
This does not certify actual under-car correctness or independent donor errors.
"""
import argparse,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from PIL import Image
from validate_dense_run_v2 import (ROOT,require,checked_ref,sha,array_digest,
    load_npz,validate_record)
from dense_experiment import immutable_json


def load_long_scene(case):
    from repair import nadir,polygon_mask
    from benchmark import masks
    for r in case['inputs']:checked_ref(ROOT,r)
    cfg=json.loads((ROOT/'benchmark_v2/protocol.json').read_text());c=cfg['cases'][case['case']]
    vehicle,known,_=masks(cfg,c)
    default=polygon_mask((cfg['size'],cfg['size'],3),c['ground'])&~vehicle
    names=[r['name'] for r in case['frames']]
    require(len(names)==11 and len(set(names))==11 and len({r['id'] for r in case['frames']})==11,'invalid long window')
    images=[nadir(np.asarray(Image.open(checked_ref(ROOT,r)).convert('RGB')),**case['projection']) for r in case['frames']]
    allowed=[known.get(n,default).copy() for n in names]
    require([array_digest(a) for a in allowed]==case['allowed_mask_hashes'],'source mask hashes changed')
    return images,allowed,names.index('target')


def verify_partitions(record,job):
    from sequence_geometry import propose_edge
    reused={(r['i'],r['j']):r for r in job['reuse']};checked=0
    for row in record['correspondences']:
        r=reused.get((row['i'],row['j']));new=load_npz(checked_ref(ROOT,row))
        indices=[r['old_i'],r['old_j']] if r else [row['i'],row['j']]
        require(row['proposal_indices']==indices and row['proposal_seed']==29,'proposal identity/seed changed')
        if r:
            require(row['reused_from']==r['source'],'reuse provenance changed')
            old=load_npz(checked_ref(ROOT,r['source']))
            for key in ['p','q','extra']:require(np.array_equal(new[key],old[key]),'old observations were modified')
            oldsig=json.loads(str(old['signature']));newsig=json.loads(str(new['signature']))
            require(oldsig['i']==indices[0] and oldsig['j']==indices[1],'old pair IDs changed')
            require({k:v for k,v in oldsig.items() if k not in ['i','j']}=={k:v for k,v in newsig.items() if k not in ['i','j']},'old pair sanitized inputs/model differ')
            checked+=1
        else:require(row['reused_from'] is None,'unplanned cache reuse')
        edge,info=propose_edge(*indices,new['p'],new['q']);info.update(i=row['i'],j=row['j'])
        expected=None if edge is None else {k:array_digest(edge[k]) for k in ['p','q','vp','vq','H']}
        require(expected==row['partition_hashes'] and info==row['proposal'],'partition or proposal changed')
    require(checked==len(reused)==7,'central pair population differs')
    return checked


def audit(regions=None):
    planpath=ROOT/'sequence_v7/execution_plan.json';plan=json.loads(planpath.read_text())
    for name,h in plan['signature'].items():checked_ref(ROOT,dict(path=name,sha256=h))
    for r in plan['protocols']:checked_ref(ROOT,r)
    require(plan['policies']==json.loads((ROOT/'benchmark_v2/protocol.json').read_text())['policies'],'policy changed')
    case=plan['cases'][0];require(case['case']=='gironde_0' and case['split']=='development','wrong scene/split')
    scene=load_long_scene(case);results=[];missing=[]
    for job in case['jobs']:
        if regions and job['region'] not in regions:continue
        path=ROOT/'sequence_v7/results'/case['case']/(job['region']+'.json')
        if not path.exists():missing.append(job['region']);continue
        result=validate_record(ROOT,plan,planpath,case,job,path,scene)
        result['old_pair_partitions_unchanged']=verify_partitions(json.loads(path.read_text()),job)
        results.append(result)
    require(results or missing,'empty selection')
    return dict(created_at_utc=datetime.now(timezone.utc).isoformat(),passed=True if results else None,
        status='all_selected_results_checked' if not missing else 'pending_results',checked_regions=len(results),
        missing_results=missing,plan_sha256=sha(planpath),validator_sha256=sha(Path(__file__)),
        scoring_validator_sha256=sha(ROOT/'validate_dense_run_v2.py'),results=results,
        limitations=['Selected donor RGB, visibility, geometry acceptance, errors and abstention checked; support hulls not independently replayed.',
                    'Real footprint lacks truth. Multi-view agreement is not a correctness probability.',
                    'Development only; smooth asphalt RGB agreement is weak evidence of positional accuracy.'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--region',action='append');p.add_argument('--output',required=True);args=p.parse_args()
    result=audit(args.region);immutable_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='results'},indent=2))
