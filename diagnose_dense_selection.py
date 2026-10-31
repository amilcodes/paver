"""Post-score donor attribution diagnostic; never participates in selection."""
import json
from pathlib import Path
import numpy as np
from benchmark import ROOT,digest
from dense_experiment import immutable_json
from validate_dense_run_v2 import accepted,checked_ref


def main():
    cfg=json.loads((ROOT/'benchmark_v2/protocol.json').read_text());out={}
    for version in ['sequence_v6','sequence_v7']:
        rows={}
        for region in ['hidden_front','hidden_rear','local_front','local_rear']:
            path=ROOT/version/'results/gironde_0'/(region+'.json');record=json.loads(path.read_text())
            method=record['methods']['loftr_metric_free'];ep=checked_ref(ROOT,method['evidence'])
            with np.load(ep,allow_pickle=False) as z:e={k:z[k] for k in z.files}
            mask=e['mask'];use=accepted(e,mask,cfg['policies']['candidate'])[mask]
            provenance=e['provenance'][mask];errors=e['absolute_rgb_error_sum_on_mask'].astype(float)/3;groups=[]
            for index,name in enumerate(record['source_order']):
                take=use&(provenance==index);n=int(take.sum())
                if n:groups.append(dict(source=name,accepted_pixels=n,mae_255=float(errors[take].mean()),over_20_fraction=float((errors[take]>20).mean())))
            rows[region]=dict(record_sha256=digest(path),evidence=method['evidence'],selected_donors=groups)
        out[version]=rows
    path=ROOT/'sequence_v7/selection_diagnostic.json'
    immutable_json(path,dict(method='loftr_metric_free',policy='candidate',versions=out,
        purpose='Post-score failure diagnosis only; hidden errors never select donors or alter the completed experiment.',
        limitation='Donor averages have different pixel populations; use saved common-pixel comparisons for controlled quality differences.'))
    immutable_json(ROOT/'sequence_v7/selection_diagnostic_provenance.json',dict(
        source=dict(path=Path(__file__).name,sha256=digest(Path(__file__))),
        result=dict(path=str(path.relative_to(ROOT)),sha256=digest(path)),reproduced_exactly=True))
    print('Donor attribution diagnostic reproduced exactly.')


if __name__=='__main__':main()
