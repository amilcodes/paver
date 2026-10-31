"""Render the complete matching benchmark without external browser libraries."""
import copy,json,os
from pathlib import Path
import numpy as np
from PIL import Image
from benchmark import ROOT,save_json
from repair import polygon_mask
R=ROOT/'benchmark_v2';config=json.loads((R/'protocol.json').read_text());scenes={}
for key,case in config['cases'].items():
 vehicle=polygon_mask((config['size'],config['size']),case['vehicle'])
 scene=dict(label=case['label'],split=case['split'],kind=case['kind'],methods={})
 for method in ['sift','disk_lightglue']:
  folder=R/'results'/key/method;d=json.loads((folder/'metrics.json').read_text())
  probes=json.loads((folder/'local_probes.json').read_text());d['regions'].update(probes['regions'])
  for name,region in d['regions'].items():
   evidence=np.load(folder/(name+'_evidence.npz'));mask=evidence['mask'];yy,xx=np.nonzero(mask)
   if len(xx):
    # Square crop around the selected mask, with enough visible context for inspection.
    x0,x1,y0,y1=xx.min()/mask.shape[1],xx.max()/mask.shape[1],yy.min()/mask.shape[0],yy.max()/mask.shape[0]
    side=max(x1-x0,y1-y0)*1.5;side=max(side,.18);cx=(x0+x1)/2;cy=(y0+y1)/2
    side=min(side,1);cx=min(1-side/2,max(side/2,cx));cy=min(1-side/2,max(side/2,cy))
    bounds=(cx-side/2,cy-side/2,cx+side/2,cy+side/2)
   else:bounds=(.25,.25,.75,.75)
   alias=folder/(name+'_original.jpg')
   if not alias.exists():os.link(folder/'original.jpg',alias)
   stems=[name+'_original',name+'_input']
   if name!='footprint':stems.append(name+'_telea')
   for policy,rule in config['policies'].items():
    stems.extend([name+'_'+policy,name+'_'+policy+'_support'])
    if name!='footprint':
     im=Image.open(folder/(name+'_'+policy+'.jpg'));a=np.array(im)
     unaccepted=vehicle|(mask&~((evidence['count']>=rule['min_views'])&(evidence['disagreement']<=rule['max_disagreement'])))
     small=np.array(Image.fromarray(unaccepted).resize(im.size,Image.Resampling.NEAREST));a[small]=0
     Image.fromarray(a).save(folder/(name+'_'+policy+'_display.jpg'),quality=94,subsampling=0)
     stems.append(name+'_'+policy+'_display')
   for stem in stems:
    im=Image.open(folder/(stem+'.jpg'));box=tuple(round(v*(im.width if i%2==0 else im.height)) for i,v in enumerate(bounds))
    im.crop(box).save(folder/(stem+'_crop.jpg'),quality=94,subsampling=0)
   # UI keeps compact diagnostics; complete point arrays remain in metrics.json.
   for registration in region['registrations']:
    for field in ['source_inliers','target_inliers','matching_scores']:registration.pop(field,None)
  scene['methods'][method]={'regions':d['regions']}
 pano=Image.open(R/'data'/key/'target.jpg').resize((2048,1024));pano.save(R/'results'/key/'sphere_original.jpg',quality=94,subsampling=0)
 scenes[key]=scene
new=[c for c in scenes.values() if c['split']=='new' and c['kind']=='car']
summary={m:{p:float(np.mean([c['methods'][m]['regions']['footprint']['policies'][p]['coverage'] for c in new])) for p in config['policies']} for m in ['sift','disk_lightglue']}
summary['new_scene_abstentions']=sum(c['methods']['disk_lightglue']['regions']['footprint']['policies']['candidate']['accepted_pixels']==0 for c in scenes.values() if c['split']=='new')
summary['car_scene_count']=len(new)
accuracy=[]
for category,regions in [('Car-shaped masks',['hidden_front','hidden_rear']),('Small local probes',['local_front','local_rear'])]:
 for method,policy in [('sift','candidate'),('disk_lightglue','candidate'),('disk_lightglue','corroborated'),('disk_lightglue','strict')]:
  rows=[c['methods'][method]['regions'][r] for c in new for r in regions];ms=[r['policies'][policy] for r in rows]
  total=sum(r['pixels'] for r in rows);accepted=sum(m['accepted_pixels'] for m in ms)
  def weighted(field,metric):return sum((m[field][metric] or 0)*m['accepted_pixels'] for m in ms)/accepted if accepted else None
  accuracy.append(dict(category=category,method=method,policy=policy,label=category+' · '+('SIFT' if method=='sift' else 'Learned')+' / '+policy,pixels=total,accepted_pixels=accepted,coverage=accepted/total if total else 0,mae=weighted('accepted_error','mae_255'),telea_mae=weighted('telea_same_pixels','mae_255'),over20=weighted('accepted_error','over_20_fraction')))
result=dict(summary=summary,accuracy=accuracy,scenes=scenes)
save_json(R/'summary.json',dict(summary=summary,accuracy=accuracy,per_scene={k:{m:{p:r['regions']['footprint']['policies'][p]['coverage'] for p in config['policies']} for m,r in c['methods'].items()} for k,c in scenes.items()}))
# Reuse the original dependency-free spherical projection viewer.
old=(ROOT/'docs/report_template.html').read_text();sphere=old[old.index('// Minimal equirectangular viewer.'):old.index('</script></body>')]
sphere=sphere.replace("let yaw=0,pitch=-1.48,fov=1.35,ready=false,drag=null,serial=0;","let yaw=0,pitch=-1.48,fov=1.35,ready=false,drag=null,serial=0,sphereRepaired=true;")
sphere=sphere.replace("function load(repaired){const request", "function load(repaired){sphereRepaired=repaired;const request")
sphere=sphere.replace("'Partial transfer on the marked-road case. Original pixels retained wherever no transfer was accepted.'","items[currentKey].label+' · '+$('policy').value+' policy. Original retained outside accepted transfers.'")
sphere=sphere.replace("img.src='preview/panorama_'+(repaired?'partial_repair':'original')+'.png';", "img.src=repaired?`benchmark_v2/results/${currentKey}/disk_lightglue/sphere_${$('policy').value}.jpg`:`benchmark_v2/results/${currentKey}/sphere_original.jpg`;$('sphereDownload').href=img.src;")
sphere=sphere.replace("window.addEventListener('resize',draw);load(true);","window.addEventListener('resize',draw);window.refreshSphere=()=>load(sphereRepaired);load(true);")
html=(ROOT/'docs/benchmark_template.html').read_text().replace('__DATA__',json.dumps(result,separators=(',',':')).replace('</','<\\/')).replace('__SPHERE__',sphere)
(ROOT/'benchmark.html').write_text(html)
print(json.dumps(dict(summary=summary,accuracy=accuracy),indent=2))
