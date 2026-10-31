"""Post-audit failure accounting from registration evidence only, not RGB truth."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from benchmark import ROOT,digest
from run_dense import ref,read_ref
from dense_experiment import immutable_json

RUN=ROOT/'controlled_v2/baseline_v1'


def describe(values):
    return dict(count=len(values),minimum=min(values) if values else None,median=float(np.median(values)) if values else None,maximum=max(values) if values else None)


def diagnose():
    plan=json.loads((RUN/'execution_plan.json').read_text());audit=json.loads((RUN/'audit_001.json').read_text())
    if audit.get('passed') is not True or audit['plan_sha256']!=digest(RUN/'execution_plan.json'):raise RuntimeError('Complete audit required')
    known={(g['scene'],g['job']):g['record_sha256'] for g in audit['groups']};scenes=[]
    for scene in plan['scenes']:
        pairs={};models={m:Counter() for m in plan['methods']};categories=[]
        for job in scene['jobs']:
            path=RUN/scene['scene']/'groups'/(job['key']+'.json')
            if digest(path)!=known[(scene['scene'],job['key'])]:raise RuntimeError('Audited group changed')
            g=json.loads(path.read_text())
            for p in g['correspondences']:
                read_ref(p);key=p['path']
                if key in pairs and pairs[key]!=p:raise RuntimeError('Shared pair diagnostic differs')
                pairs[key]=p
            for method,v in g['graphs'].items():
                accepted=sum(r.get('accepted') is True for r in g['methods'][method]['registrations'])
                state=('target_disconnected' if v.get('connected_frames',0)<=1 else
                       'optimizer_nonconverged' if v.get('optimizer_success') is not True else
                       'no_accepted_donors' if not accepted else 'accepted_donor_geometry')
                models[method][state]+=1
                categories.append(dict(job=job['key'],kind=job['kind'],method=method,state=state,accepted_donors=accepted,
                    connected_frames=v.get('connected_frames',0),accepted_graph_edges=v.get('edges')))
        pairs=list(pairs.values());rejected=[p for p in pairs if p['proposal'].get('accepted') is not True]
        accepted=[p for p in pairs if p['proposal'].get('accepted') is True]
        scenes.append(dict(scene=scene['scene'],area=scene['area'],unique_pair_caches=len(pairs),
            pair_proposal_reasons=dict(Counter(p['proposal'].get('reason') for p in pairs)),
            accepted_pair_count=len(accepted),rejected_pair_count=len(rejected),raw_match_counts=describe([p['matches'] for p in pairs]),
            rejected_validation_median_px=describe([p['proposal']['validation_median_px'] for p in rejected if 'validation_median_px' in p['proposal']]),
            accepted_validation_median_px=describe([p['proposal']['validation_median_px'] for p in accepted if 'validation_median_px' in p['proposal']]),
            model_failure_counts={m:dict(v) for m,v in models.items()},groups=categories))
    result=dict(plan=ref(RUN/'execution_plan.json'),audit=ref(RUN/'audit_001.json'),source=ref(Path(__file__).resolve()),scenes=scenes,
        uses_clean_target_rgb=False,selects_new_matches_or_donors=False,
        interpretation='Registration/heldout-correspondence diagnostics only. Accepted geometry is not pixel support or correctness. Donor-only pairs deduplicated; mask-dependent pair observations remain correlated.')
    immutable_json(RUN/'visible_evidence_diagnostic.json',result)
    print(json.dumps([dict(scene=s['scene'],accepted_pairs=s['accepted_pair_count'],unique_pairs=s['unique_pair_caches'],failures=s['model_failure_counts']) for s in scenes]))


if __name__=='__main__':diagnose()
