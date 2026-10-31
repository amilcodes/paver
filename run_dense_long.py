"""V7: eleven-frame Gironde test with the exact v6 central-pair observations.

The only treatment is additional frames/edges. Existing pair seeds and heldout
partitions follow their old indices; shifting the target from 2 to 5 must not
silently change those partitions. Hidden RGB enters only the final scorer.
"""
import argparse,gc,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from PIL import Image
from benchmark import ROOT,digest,masks
from repair import nadir,polygon_mask
from dense_experiment import (array_hash,immutable_json,masked_inputs,pair_indices,
    cached_pair,spatial_edge_audit,score_transfer,same_pixel_error)
from sequence_geometry import propose_edge,fit_graph
from metric_plane import fit_metric_graph
from constrained_sequence import audit_poses
from resource_guard import require_storage
from run_dense import (SOURCES,ref,read_ref,model_signature,verify_installed,
    verify_certificate,heavy_job,save_evidence)

OUT=ROOT/'sequence_v7'
METHODS=['loftr_homography','loftr_metric_free']


def remapped_proposal(i,j,p,q,proposal_indices):
    """Fit unchanged observations using stable v6 IDs, then relabel graph nodes."""
    a,b=proposal_indices
    if min(a,b)<0:raise ValueError('Proposal indices must be nonnegative')
    edge,info=propose_edge(a,b,p,q)
    if edge is not None:edge=dict(edge,i=i,j=j,info=dict(edge['info'],i=i,j=j))
    return edge,dict(info,i=i,j=j)


def partition_hashes(edge):
    return None if edge is None else {k:array_hash(edge[k]) for k in ['p','q','vp','vq','H']}


def load_scene(case):
    for r in case['inputs']:read_ref(r)
    config=json.loads((ROOT/'benchmark_v2/protocol.json').read_text())
    vehicle,known,_=masks(config,config['cases'][case['case']])
    default=polygon_mask((config['size'],config['size'],3),config['cases'][case['case']]['ground'])&~vehicle
    allowed=[known.get(r['name'],default).copy() for r in case['frames']]
    if [array_hash(a) for a in allowed]!=case['allowed_mask_hashes']:raise ValueError('Allowed masks changed')
    images=[nadir(np.asarray(Image.open(read_ref(r)).convert('RGB')),**case['projection']) for r in case['frames']]
    return images,allowed,[r['name'] for r in case['frames']].index('target')


def build_plan():
    oldpath=ROOT/'sequence_v6/execution_plan.json';old=json.loads(oldpath.read_text())
    for name,h in old['signature'].items():read_ref(dict(path=name,sha256=h))
    original=next(c for c in old['cases'] if c['case']=='gironde_0')
    config=json.loads((ROOT/'benchmark_v2/protocol.json').read_text())
    window=ROOT/'dataset_v1/windows/gironde_0.json';manifest=json.loads(window.read_text())
    longprotocol=ROOT/'sequence_v3/protocol.json';names=json.loads(longprotocol.read_text())['frames']
    records={r['name']:r for r in manifest['images']};frames=[]
    for name in names:
        r=records[name];image=ref(ROOT/'dataset_v1/images'/(r['id']+'.jpg'))
        if image['sha256']!=r['sample_sha256']:raise ValueError('Frozen long-window image differs')
        frames.append(dict(name=name,id=r['id'],**image,metadata=ref(ROOT/'dataset_v1/metadata'/(r['id']+'.json')),
                           license=r['license'],producer=r['producer'],capture_time=r['capture_time']))
    if len(frames)!=11 or len({r['id'] for r in frames})!=11:raise ValueError('Expected eleven distinct frames')
    mapping={}
    for old_i,r in enumerate(original['frames']):
        new_i=names.index(r['name']);read_ref(r)
        if (r['id'],r['sha256'])!=(frames[new_i]['id'],frames[new_i]['sha256']):raise ValueError('Central source identity changed')
        mapping[new_i]=old_i
    vehicle,known,_=masks(config,config['cases']['gironde_0'])
    default=polygon_mask((config['size'],config['size'],3),config['cases']['gironde_0']['ground'])&~vehicle
    mask_hashes=[array_hash(known.get(n,default)) for n in names]
    jobs=[]
    for original_job in original['jobs']:
        region=original_job['region'];controlpath=ROOT/'sequence_v6/results/gironde_0'/(region+'.json')
        control=json.loads(controlpath.read_text())
        if control['signature']['plan_sha256']!=digest(oldpath):raise ValueError('Stale v6 control')
        oldpairs={(p['i'],p['j']):p for p in control['correspondences']};reuse=[]
        for i,j in pair_indices(11,5):
            if i in mapping and j in mapping:
                a,b=mapping[i],mapping[j];p=oldpairs[a,b];path=Path(p['path'])
                if not path.is_absolute():path=ROOT/path
                oldref=ref(path)
                if oldref['sha256']!=p['sha256']:raise ValueError('Old pair cache differs')
                reuse.append(dict(i=i,j=j,old_i=a,old_j=b,source=oldref))
        controls={m:dict(record=ref(controlpath),evidence=control['methods'][m]['evidence']) for m in METHODS}
        for refs in controls.values():read_ref(refs['evidence'])
        jobs.append(dict(region=region,mask=original_job['mask'],mask_key=original_job['mask_key'],controls=controls,reuse=reuse))
    inputs=[ref(window),ref(longprotocol),ref(oldpath),ref(ROOT/'benchmark_v2/protocol.json')]
    case=dict(case='gironde_0',split='development',frames=frames,projection=original['projection'],inputs=inputs,
              jobs=jobs,allowed_mask_hashes=mask_hashes)
    sources=list(dict.fromkeys(SOURCES+['run_dense_long.py','validate_dense_long.py','test_dense_long.py','validate_dense_run_v2.py','summarize_dense.py']))
    return dict(experiment='sequence_v7_long_dense',signature={n:digest(ROOT/n) for n in sources},
        protocols=inputs,model=model_signature(),policies=old['policies'],cases=[case],
        treatment='Add six frozen neighboring frames; unchanged matching, geometry, masks and gates.',
        central_pair_control='Reuse exact v6 correspondences and proposal index seeds; no re-splitting existing matches.',
        additional_pair_control='Same distance<=2 or target-in-pair topology; seed29 with new nonnegative indices.',
        mask_review='Added-frame masks reuse sequence_v3 reviewed conservative main-road/vehicle masks.',
        limits=['Eleven-frame neighborhood, not full capture-collection reconstruction.',
                'Development only; no actual footprint ground truth; donor agreement is not accuracy.'])


def register_long(clean,allowed,target,fov,job,cache_dir,matcher,model):
    if any(image[~ok].any() for image,ok in zip(clean,allowed)):raise ValueError('Unsanitized registration input')
    reused={(r['i'],r['j']):r for r in job['reuse']};edges=[];records=[]
    # Old indices can be reconstructed without access to unsanitized images.
    oldmap={r['old_i']:r['i'] for r in job['reuse']}
    oldmap.update({r['old_j']:r['j'] for r in job['reuse']})
    oldclean=[clean[oldmap[i]] for i in range(5)];oldallowed=[allowed[oldmap[i]] for i in range(5)]
    for i,j in pair_indices(len(clean),target):
        r=reused.get((i,j));tag=job['region'] if target in (i,j) else 'donor_only'
        path=Path(cache_dir)/(f'{tag}_{i}_{j}.npz')
        def get_match(a,b,ia,ib,ma,mb):
            if r is None:return matcher(a,b,ia,ib,ma,mb)
            def forbidden(*args):raise RuntimeError('Missing immutable central-pair cache')
            p,q,extra,_=cached_pair(read_ref(r['source']),r['old_i'],r['old_j'],oldclean,oldallowed,forbidden,model)
            return p,q,extra
        p,q,extra,_=cached_pair(path,i,j,clean,allowed,get_match,model)
        indices=[r['old_i'],r['old_j']] if r else [i,j]
        # Always verify reused bytes on resume as well.
        if r:
            def forbidden(*args):raise RuntimeError('Missing old observations')
            op,oq,oe,_=cached_pair(read_ref(r['source']),*indices,oldclean,oldallowed,forbidden,model)
            if not np.array_equal(p,op) or not np.array_equal(q,oq) or extra!=oe:raise ValueError('Reused observations changed')
        edge,info=remapped_proposal(i,j,p,q,indices)
        if edge is not None:edges.append(edge)
        records.append(dict(i=i,j=j,**ref(path),matches=len(p),proposal=info,proposal_indices=indices,
            proposal_seed=29,partition_hashes=partition_hashes(edge),reused_from=r['source'] if r else None,
            spatial_holdout=spatial_edge_audit(p,q,clean[target].shape[0]),matcher=extra))
        print('PAIR',job['region'],i,j,len(p),'reused' if r else 'new','accepted',edge is not None,flush=True)
    methods={};graphs={};size=clean[target].shape[0]
    for method in METHODS:
        if method=='loftr_homography':poses,diag,active=fit_graph(len(clean),target,edges,size)
        else:poses,diag,active=fit_metric_graph(len(clean),target,edges,size,fov)
        diag['poses']={str(k):v.tolist() for k,v in poses.items()}
        methods[method]=audit_poses(clean,allowed,poses,diag,active,target);graphs[method]=diag
    return methods,graphs,records


def run(args,plan):
    with heavy_job():
        verify_installed();verify_certificate(args.certificate)
        signature=dict(plan_sha256=digest(OUT/'execution_plan.json'),certificate=ref(Path(args.certificate).resolve()))
        from dense_matching import MaskedLoFTR
        matcher=None;case=plan['cases'][0];images,allowed,target=load_scene(case);completed=0
        for job in case['jobs']:
            if args.region and job['region'] not in args.region:continue
            out=OUT/'results'/case['case'];out.mkdir(parents=True,exist_ok=True);dest=out/(job['region']+'.json')
            if dest.exists():
                old=json.loads(dest.read_text())
                if old['signature']!=signature:raise ValueError('Changed run signature')
                for data in old['methods'].values():read_ref(data['evidence'])
                print('RESUME',job['region'],flush=True);continue
            if completed>=args.limit:break
            start=time.monotonic();require_storage(ROOT)
            with np.load(read_ref(job['mask']),allow_pickle=False) as z:mask=z[job['mask_key']]
            footprint=job['region']=='footprint'
            if (mask&allowed[target]).any() if footprint else (mask&~allowed[target]).any():raise ValueError('Ineligible target mask')
            clean,als=masked_inputs(images,allowed,mask,target)
            if matcher is None:matcher=MaskedLoFTR(args.weights,plan['model']['checkpoint_sha256'],root=ROOT)
            def match(i,j,a,b,ma,mb):return matcher.correspondences(a,b,ma,mb)
            regs,graphs,pairs=register_long(clean,als,target,case['projection']['fov'],job,out/'pairs',match,plan['model'])
            result=dict(case=case['case'],split=case['split'],region=job['region'],signature=signature,
                created_at_utc=datetime.now(timezone.utc).isoformat(),mask_pixels=int(mask.sum()),
                source_order=[r['name'] for i,r in enumerate(case['frames']) if i!=target],graphs=graphs,
                correspondences=pairs,methods={},controls={},same_pixel_control={},promotion='none')
            truth=None if footprint else images[target]
            for method,r in regs.items():
                require_storage(ROOT)
                stats,evidence=score_transfer(clean,als,mask,target,r,plan['policies'],truth)
                ep=out/(job['region']+'_'+method+'_evidence.npz');save_evidence(ep,evidence)
                result['methods'][method]=dict(policies=stats,registrations=[v.diagnostics for v in r.values()],evidence=ref(ep))
                for control,refs in job['controls'].items():
                    old=json.loads(read_ref(refs['record']).read_text())['methods'][control]['policies']
                    result['controls'][control]=dict(references=refs,policies={k:dict(v,abstention=1-v['coverage']) for k,v in old.items()})
                    with np.load(read_ref(refs['evidence']),allow_pickle=False) as z:cached={k:z[k] for k in z.files}
                    if not np.array_equal(mask,cached['mask']):raise ValueError('Control mask differs')
                    if truth is not None:result['same_pixel_control'].setdefault(method,{})[control]=same_pixel_error(evidence,cached,plan['policies'])
                del evidence,cached;gc.collect()
            result['runtime_seconds']=time.monotonic()-start;immutable_json(dest,result);completed+=1
            print('RESULT',job['region'],{m:v['policies']['candidate'] for m,v in result['methods'].items()},flush=True)
            del clean,als,regs;gc.collect()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--region',action='append')
    p.add_argument('--limit',type=int,default=1);p.add_argument('--weights');p.add_argument('--certificate');args=p.parse_args()
    if not 1<=args.limit<=5:p.error('limit must be 1..5')
    if args.run and (not args.weights or not args.certificate):p.error('Explicit weights and certificate required')
    plan=build_plan();immutable_json(OUT/'execution_plan.json',plan)
    if args.run:run(args,plan)
    else:print(json.dumps(dict(plan=str(OUT/'execution_plan.json'),regions=5,methods=2,model_executed=False)))
