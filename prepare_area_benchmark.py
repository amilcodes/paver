"""Compile reviewed area-source masks and freeze controlled conditions, no model.

The new six anchors remain development/train. Texture strata are scoring-only,
and exact pixel solid angles supplement projected-area counts without altering
any model policy. Existing benchmarks/annotations are never changed.
"""
import argparse,json,math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from benchmark import ROOT,digest,save_json
from repair import polygon_mask,nadir
from prepare_annotated_case_v2 import compile_masks,CATEGORIES
from run_dense import heavy_job,ref,read_ref,save_evidence
from dense_experiment import immutable_json,array_hash
from controlled_occlusions import make_mask,mask_shape,FAMILIES,SEVERITIES
from dataset_admission import effective_splits

OUT=ROOT/'controlled_v2'


def solid_angles(size,fov):
    if type(size) is not int or size<2 or not 0<fov<180:raise ValueError('Invalid pinhole projection')
    f=size/(2*np.tan(np.deg2rad(fov/2)));c=np.arange(size+1,dtype=np.float64)-size/2
    x=c[None,:]/f;y=c[:,None]/f
    F=np.arctan2(x*y,np.sqrt(1+x*x+y*y))
    return F[1:,1:]-F[:-1,1:]-F[1:,:-1]+F[:-1,:-1]


def assemble(a):
    parts,_,_=compile_masks(a);margin=a['ground_margin_px']
    if type(margin) is not int or not 0<=margin<=64:raise ValueError('Invalid ground margin')
    counts={}
    for name in a['frame_order']:
        ground=parts[name+'_ground']
        safe=cv2.erode(ground.astype(np.uint8),np.ones((2*margin+1,2*margin+1),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
        parts[name+'_allowed']=safe&~parts[name+'_excluded']
        if parts[name+'_allowed'].sum()<10000:raise ValueError('Insufficient reviewed ground')
        counts[name]={c:int(parts[name+'_'+c].sum()) for c in CATEGORIES}
        counts[name]['allowed']=int(parts[name+'_allowed'].sum())
    shape=parts['target_allowed'].shape
    for name,polys in a['regions'].items():
        parts['region_'+name]=polygon_mask(shape,polys)&parts['target_allowed']
        if parts['region_'+name].sum()<1000:raise ValueError('Degenerate evaluation basis')
    return parts,counts


def preview(a,parts):
    folder=OUT/'review'/a['scene_id'];sheet=Image.new('RGB',(1800,1300),(20,20,20));d=ImageDraw.Draw(sheet)
    for j,name in enumerate(a['frame_order']):
        index=next(i['index'] for i in json.loads((OUT/'annotation_selection.json').read_text())['targets'] if i['target_id']==a['target_id'])+j-2
        raw=np.asarray(Image.open(folder/f'{index:03d}.jpg').convert('RGB'));over=raw.copy()
        for kind,color in [('allowed',[0,120,0]),('ego',[170,0,0]),('moving',[150,0,150]),('shadow',[150,100,0]),('uncertain',[200,70,0])]:
            mask=cv2.resize(parts[name+'_'+kind].astype(np.uint8),(1200,1200),interpolation=cv2.INTER_NEAREST)>0
            over[mask]=np.clip(.55*raw[mask]+.45*np.asarray(color),0,255)
        if name=='target':
            for r in a['regions']:
                mask=cv2.resize(parts['region_'+r].astype(np.uint8),(1200,1200),interpolation=cv2.INTER_NEAREST)>0
                contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(over,contours,-1,(0,255,255),3)
        im=Image.fromarray(over);im.save(folder/(name+'_overlay.jpg'),quality=95)
        x=j%3*600;y=j//3*650;sheet.paste(im.resize((600,600)),(x,y+25));d.text((x+8,y+6),a['scene_id']+' '+name,fill='white')
    sheet.save(OUT/'review'/(a['scene_id']+'_overlay_sheet.jpg'),quality=95)


def prepare(freeze=False):
    selection=OUT/'annotation_selection.json';selected=json.loads(selection.read_text());effective=effective_splits(ROOT/'dataset_v1')
    graph=json.loads((ROOT/'network_v1/candidate_graph.json').read_text());links={(r['target'],r['donor']) for r in graph['edges']};summaries=[]
    for target in selected['targets']:
        name=f"{target['area']}_{target['index']:03d}";path=OUT/'annotations'/(name+'.json');a=json.loads(path.read_text());parts,counts=assemble(a)
        read_ref(a['source']);read_ref(a['source_manifest']);read_ref(a['selection'])
        if a['split'] not in ['development','train'] or effective['collections'][a['collection']]['split']!=a['split'] or a['collection'] in effective['quarantined_collections']:raise ValueError('Unsafe split')
        for n,r in a['frames'].items():
            read_ref(dict(path=r['image'],sha256=r['sample_sha256']));meta=json.loads(read_ref(dict(path=r['metadata_path'],sha256=r['metadata_sha256'])).read_text())
            if meta['collection']!=a['collection'] or meta['properties']['license']!=r['license'] or r['license']!='etalab-2.0':raise ValueError('Provenance mismatch')
            if n!='target' and (a['target_id'],r['id']) not in links:raise ValueError('Annotated donor outside metadata graph')
        if not freeze:preview(a,parts);continue
        if a['status']!='reviewed_frozen_before_scoring':raise ValueError('Overlay review required')
        weights=solid_angles(**a['projection']);footprint=parts['region_footprint'];specs=[];conditions=[]
        for location in a['regions']:
            basis=parts['region_'+location]
            for shape in ['disc','vehicle','strip']:
                for severity in SEVERITIES:
                    key=f'{location}_{shape}_{severity}';mask=make_mask(basis,shape,severity)
                    if mask.sum()<500:raise ValueError('Degenerate corruption mask')
                    parts['hidden_'+key]=mask
                    specs.append(dict(key=key,location=location,shape=shape,severity=severity,sha256=array_hash(mask),pixels=int(mask.sum()),
                        solid_angle_sr=float(weights[mask].sum()),sphere_fraction=float(weights[mask].sum()/(4*math.pi)),
                        projected_area_ratio_to_obstruction=float(mask.sum()/footprint.sum()),angular_area_ratio_to_obstruction=float(weights[mask].sum()/weights[footprint].sum())))
            for family in FAMILIES:
                for severity in SEVERITIES:conditions.append(dict(key=f'{location}_{family}_{severity}',location=location,family=family,severity=severity,mask_key=f'hidden_{location}_{mask_shape(family)}_{severity}'))
        source=np.asarray(Image.open(ROOT/a['frames']['target']['image']).convert('RGB'));truth=nadir(source,**a['projection']);g=cv2.cvtColor(truth,cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3,scale=1/8);gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3,scale=1/8)
        textured=np.hypot(gx,gy)>=4;folder=OUT/'prepared'/name;folder.mkdir(parents=True,exist_ok=True)
        mp=folder/'masks.npz';save_evidence(mp,parts);sp=folder/'scoring_only_strata.npz';save_evidence(sp,dict(textured=textured))
        for spec in specs:
            mask=parts['hidden_'+spec['key']];spec['scoring_only_textured_fraction']=float(textured[mask].mean())
        manifest=dict(scene_id=name,area=target['area'],target_id=a['target_id'],split=a['split'],annotation=ref(path),masks=ref(mp),
            scoring_only_strata=ref(sp),compiler=ref(Path(__file__)),base_compiler=ref(ROOT/'prepare_annotated_case_v2.py'),corruption_source=ref(ROOT/'controlled_occlusions.py'),
            projection=a['projection'],frames=counts,mask_specs=specs,conditions=conditions,conditions_count=len(conditions),
            source_frame_hashes={n:r['sample_sha256'] for n,r in a['frames'].items()},
            obstruction=dict(pixels=int(footprint.sum()),solid_angle_sr=float(weights[footprint].sum()),interpretation=a['footprint_interpretation']),
            truth_policy='Scoring-only strata/clean RGB are forbidden inputs to graph retrieval, registration, output selection and confidence. No clean truth beneath real ego mask.',model_executed=False)
        immutable_json(folder/'manifest.json',manifest);summaries.append(ref(folder/'manifest.json'))
    if freeze:
        protocol=dict(status='annotations_and_corruptions_frozen_not_evaluated',selection=ref(selection),graph=ref(ROOT/'network_v1/candidate_graph.json'),prepared=summaries,
            anchors=6,reviewed_source_frames=30,conditions=144,distinct_geometry_masks=72,existing_controlled_v1=ref(ROOT/'controlled_v1/execution_plan.json'),
            evaluation='Measure per-area/family/severity RGB/gradient errors, angular/projected coverage, abstention, error tails and scoring-only texture strata. Include unchanged-input and Telea controls on identical accepted pixels.',
            warning='Same routes remain development/train. Shared mask conditions are correlated. This is an annotation expansion, not new recovery evidence.',
            next_step='Freeze a versioned execution plan and runner with exact source/model signatures. Run unchanged source-only baseline before the proposed native-scale matcher ablation. Do not modify previous signed runners.')
        immutable_json(OUT/'benchmark_protocol.json',protocol);print(json.dumps(dict(anchors=6,frames=30,conditions=144,masks=72,model_executed=False)))
    else:print('Prepared6 overlay sheets; review before freeze')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');args=p.parse_args()
    with heavy_job():prepare(args.freeze)
