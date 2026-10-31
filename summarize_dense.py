"""Compact fixed-road v6 aggregates, requiring completed independent audits."""
import json
from pathlib import Path
from datetime import datetime,timezone
from benchmark import ROOT,digest
from dense_experiment import immutable_json


def aggregate(rows):
    out={}
    for policy in ['candidate','corroborated','strict']:
        stats=[(d['mask_pixels'],m['policies'][policy]) for d,m in rows]
        total=sum(n for n,_ in stats);used=sum(v['accepted_pixels'] for _,v in stats)
        v=dict(mask_pixels=total,accepted_pixels=used,coverage=used/max(1,total),
               abstention=1-used/max(1,total),single_view_pixels=sum(r['single_view_pixels'] for _,r in stats),
               multi_view_pixels=sum(r['multi_view_pixels'] for _,r in stats))
        if all('error' in r for _,r in stats):
            errors=[r for _,r in stats if r['accepted_pixels']]
            bad=sum(round(r['error']['over_20_fraction']*r['accepted_pixels']) for r in errors)
            v.update(mae_255=sum(r['error']['mae_255']*r['accepted_pixels'] for r in errors)/used if used else None,
                     over_20_fraction=bad/used if used else None,
                     accepted_rgb_mae_le_20_pixels=used-bad,accepted_rgb_mae_le_20_fraction_of_mask=(used-bad)/max(1,total))
            gn=sum(r['gradient']['pixels'] for _,r in stats)
            v['gradient_mae']=sum(r['gradient']['gradient_mae']*r['gradient']['pixels'] for _,r in stats if r['gradient']['pixels'])/gn if gn else None
        out[policy]=v
    return out


def main():
    root=ROOT/'sequence_v6';plan=json.loads((root/'execution_plan.json').read_text())
    audit_names={'gironde_0':'gironde_audit_002.json','alsace_fields_0':'alsace_audit_001.json','bourges_turn_001':'bourges_audit_001.json'}
    report=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),scope='Three fixed failed roads; Gironde and Alsace development, Bourges training. Not held-out test performance.',
                plan_sha256=digest(root/'execution_plan.json'),model_certificate_sha256=digest(root/'model_validation_001.json'),
                summarizer_sha256=digest(Path(__file__)),audits={},cases={},region_method_runs=0,promotion='none',
                limitations=['No ground truth beneath the real car.',
                             'Low RGB error on untextured asphalt can conceal positional error; gradient direction is unstable at near-zero predicted magnitude.',
                             'Multiple-donor agreement and spatial match holdouts are not calibrated correctness probabilities.',
                             'The five-frame matcher/geometry settings are unchanged across all three roads.'])
    for case in plan['cases']:
        name=case['case'];audit_path=root/audit_names[name];audit=json.loads(audit_path.read_text())
        assert audit['passed'] and audit['checked_regions']==5 and audit['plan_sha256']==report['plan_sha256']
        report['audits'][name]=dict(path=str(audit_path.relative_to(ROOT)),sha256=digest(audit_path))
        audited={r['region']:r['record_sha256'] for r in audit['results']};records=[]
        for job in case['jobs']:
            path=root/'results'/name/(job['region']+'.json');assert digest(path)==audited[job['region']]
            records.append(json.loads(path.read_text()))
        methods=list(records[0]['methods'])+['cached_'+m for m in records[0]['controls']]
        summary=dict(split=case['split'],methods={},footprint_matching={})
        for method in methods:
            groups={}
            for group,regions in [('footprint',['footprint']),('large_hidden',['hidden_front','hidden_rear','large_front','large_rear']),('local_hidden',['local_front','local_rear'])]:
                rows=[(d,d['controls'][method.removeprefix('cached_')] if method.startswith('cached_') else d['methods'][method]) for d in records if d['region'] in regions]
                groups[group]=aggregate(rows)
            summary['methods'][method]=groups
        foot=next(d for d in records if d['region']=='footprint')
        summary['footprint_matching']['pairs']=[dict(i=p['i'],j=p['j'],matches=p['matches'],proposal=p['proposal']) for p in foot['correspondences']]
        for method,g in foot['graphs'].items():
            summary['footprint_matching'][method]=dict(connected_frames=g['connected_frames'],optimizer_success=g.get('optimizer_success',False),
                    donors=[dict(accepted=r['accepted'],reason=r['reason']) for r in foot['methods'][method]['registrations']])
        report['cases'][name]=summary;report['region_method_runs']+=len(records)*len(foot['methods'])
    immutable_json(root/'summary.json',report)
    print(json.dumps(dict(region_method_runs=report['region_method_runs'],
                         cases={name:{m:dict(footprint=d['footprint']['candidate'],large_hidden=d['large_hidden']['candidate']) for m,d in c['methods'].items() if not m.startswith('cached_')} for name,c in report['cases'].items()}),indent=2))


if __name__=='__main__':main()
