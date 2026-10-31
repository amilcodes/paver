"""Export audited query-specific donor evidence, never a correctness graph."""
import json
from pathlib import Path
from benchmark import ROOT,digest,policy_mask
from run_dense import ref,read_ref,heavy_job
from run_controlled_benchmark import load_evidence
from dense_experiment import immutable_json


def export():
    folder=ROOT/'sampling_v1';plan=json.loads((folder/'execution_plan.json').read_text());audit=json.loads((folder/'audit_001.json').read_text())
    if audit.get('passed') is not True or audit['plan_sha256']!=digest(folder/'execution_plan.json'):raise RuntimeError('Complete audit required')
    baseline=json.loads(read_ref(plan['baseline_plans'][0]).read_text());graph_ref=baseline['candidate_graph']
    graph=json.loads(read_ref(graph_ref).read_text());links={(e['target'],e['donor']) for e in graph['edges']};rows=[];outside=[]
    hashes={(r['scene'],r['job']):r['record_sha256'] for r in audit['groups']}
    for scene in plan['scenes']:
        target=scene['target'];donors=[i for i in range(5) if i!=target];target_id=scene['frames'][target]['id']
        if scene['area']=='alsace':outside.append(scene['scene']);continue
        for i in donors:
            if (target_id,scene['frames'][i]['id']) not in links:raise RuntimeError('Unretrieved area donor')
        for job in scene['jobs']:
            path=folder/scene['scene']/'groups'/(job['key']+'.json')
            if digest(path)!=hashes[(scene['scene'],job['key'])]:raise RuntimeError('Audited group changed')
            r=json.loads(path.read_text())
            for method,d in r['methods'].items():
                e=load_evidence(d['evidence']);accepted={n:policy_mask(e,e['mask'],p) for n,p in plan['policies'].items()}
                for index,(i,reg) in enumerate(zip(donors,d['registrations'])):
                    rows.append(dict(scene=scene['scene'],target=target_id,donor=scene['frames'][i]['id'],job=job['key'],kind=job['kind'],method=method,
                        geometry_accepted=reg.get('accepted') is True,selected_pixels={n:int((mask&(e['provenance']==index)).sum()) for n,mask in accepted.items()},record=ref(path)))
    result=dict(plan=ref(folder/'execution_plan.json'),audit=ref(folder/'audit_001.json'),source=ref(Path(__file__).resolve()),retrieval_graph=graph_ref,model=plan['model'],
        decisions=rows,outside_area_graph_controls=outside,status='experimental_not_promoted',
        scope='Nine reviewed area targets; query/method-specific donor decisions only. Masks sharing sources are correlated. No calibrated correctness or validity for unqueried metadata links. Alsace hard-road controls retained separately.')
    immutable_json(ROOT/'network_v1/visual_evidence_sampling_960.json',result)
    print(json.dumps(dict(decisions=len(rows),distinct_target_donor_pairs=len({(r['target'],r['donor']) for r in rows}),outside_area_graph_controls=outside)))


if __name__=='__main__':
    with heavy_job():export()
