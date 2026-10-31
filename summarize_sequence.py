"""Compact coverage/error accounting; no renderer or visual report."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CASES=['strasbourg_outer_0','alsace_fields_0','bordeaux_0','gironde_0','nantes_1']
POLICIES=['candidate','corroborated','strict']

def gather(experiment,method):
 out={}
 for case in CASES:
  if experiment=='benchmark_v2':
   folder=ROOT/experiment/'results'/case/method;d=json.loads((folder/'metrics.json').read_text());rows=d['regions'];rows.update(json.loads((folder/'local_probes.json').read_text())['regions'])
   for z in rows.values():
    z['mask_pixels']=z['pixels']
    for v in z['policies'].values():v['error']=v.get('accepted_error')
  else:
   rows={p.stem:json.loads(p.read_text()) for p in (ROOT/experiment/'results'/case).glob('*.json')}
   rows={k:dict(mask_pixels=d['mask_pixels'],policies=d['methods'][method]['policies']) for k,d in rows.items()}
  out[case]=rows
 return out

def aggregate(rows):
 out={}
 for cohort,regions in [('footprint',['footprint']),('large_hidden',['hidden_front','hidden_rear']),('local_hidden',['local_front','local_rear'])]:
  out[cohort]={}
  for policy in POLICIES:
   sample=[rows[c][r] for c in CASES for r in regions if r in rows.get(c,{})]
   total=sum(s['mask_pixels'] for s in sample);accepted=sum(s['policies'][policy]['accepted_pixels'] for s in sample)
   v=dict(total_mask_pixels=total,accepted_pixels=accepted,pooled_coverage=accepted/max(total,1),macro_coverage=sum(s['policies'][policy]['coverage'] for s in sample)/max(len(sample),1),evaluated_regions=len(sample))
   if cohort!='footprint':
    scored=[s['policies'][policy] for s in sample if s['policies'][policy]['error'] and s['policies'][policy]['error']['pixels']]
    n=sum(s['error']['pixels'] for s in scored)
    for field in ['mae_255','over_20_fraction']:
     v[field]=sum(s['error'][field]*s['error']['pixels'] for s in scored)/n if n else None
    v['telea_mae_same_pixels']=sum(s['telea_same_pixels']['mae_255']*s['error']['pixels'] for s in scored)/n if n else None
    v['error_le_20_coverage']=sum(s['error']['pixels']*(1-s['error']['over_20_fraction']) for s in scored)/max(total,1)
   out[cohort][policy]=v
 return out

if __name__=='__main__':
 out={'scope':'The same five car scenes; now development data, not a fresh test set. Hidden errors are not actual under-car ground truth. Pixels are correlated, not independent trials.','experiments':{}}
 for exp,methods in [('benchmark_v2',['disk_lightglue']),('sequence_v1',['tiled_pairwise','sequence_graph']),('sequence_v2',['trajectory_pairwise','sequence_graph'])]:
  for method in methods:
   rows=gather(exp,method);key=exp+'/'+method;out['experiments'][key]=aggregate(rows)
   out['experiments'][key]['per_scene_footprint']={c:{p:rows[c]['footprint']['policies'][p]['coverage'] for p in POLICIES} for c in CASES}
 (ROOT/'autoresearch'/'comparison.json').write_text(json.dumps(out,indent=2)+'\n')
 for name,d in out['experiments'].items():
  print(name,'footprint',round(d['footprint']['candidate']['macro_coverage'],4),'corroborated',round(d['footprint']['corroborated']['macro_coverage'],4),'hidden',d['large_hidden']['candidate'])
