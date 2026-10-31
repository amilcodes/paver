"""Manual source polygons from the six frozen area windows; no model output.

Coordinates refer to normalized163.302-degree nadir views. These conservative
obstruction envelopes are not exact physical vehicle dimensions. Areas outside
the explicitly reviewed road polygons are out of scope, not implicitly clean.
"""
import json,math
from pathlib import Path
from benchmark import ROOT,digest,save_json
from run_dense import ref


def box(x0,y0,x1,y1):return [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
def ellipse(x,y,rx,ry):return [[round(x+rx*math.cos(t*math.pi/20),6),round(y+ry*math.sin(t*math.pi/20),6)] for t in range(40)]

G_EGO=[[[.439,.319],[.455,.312],[.527,.314],[.548,.326],[.552,.549],[.531,.557],[.517,.57],[.402,.582],[.400,.567],[.419,.538],[.427,.519],[.452,.509]],ellipse(.407,.486,.045,.04),box(.465,.576,.495,.639),box(.439,.584,.478,.602)]
B_EGO=[ellipse(.5,.5,.104,.104),[[.437,.403],[.441,.384],[.465,.376],[.525,.376],[.546,.384],[.553,.404]],[[.442,.582],[.551,.582],[.543,.607],[.52,.615],[.468,.615],[.45,.605]],box(.489,.313,.496,.405)]

GROUND={
 'gironde_013':[
  [[.01,.16],[.22,.34],[.33,.35],[.44,.31],[.87,.01],[.98,.01],[.98,.08],[.67,.31],[.56,.43],[.55,.55],[.62,.68],[.91,.98],[.79,.98],[.57,.75],[.43,.62],[.29,.60],[.14,.68],[.01,.76],[.01,.67],[.10,.57],[.12,.48],[.01,.31]],
  [[.01,.385],[.20,.44],[.33,.44],[.41,.36],[.72,.01],[.83,.01],[.65,.25],[.56,.44],[.55,.57],[.62,.69],[.86,.98],[.73,.98],[.55,.79],[.42,.72],[.31,.73],[.20,.78],[.01,.95]],
  [[.01,.585],[.19,.57],[.33,.50],[.43,.41],[.65,.01],[.77,.01],[.64,.26],[.57,.46],[.57,.58],[.60,.67],[.68,.74],[.80,.81],[.82,.98],[.24,.98],[.01,.96]],
  [[.01,.78],[.18,.68],[.35,.58],[.42,.48],[.53,.01],[.66,.01],[.59,.28],[.56,.50],[.59,.68],[.67,.79],[.90,.98],[.01,.98]],
  [[.01,.94],[.18,.81],[.34,.66],[.41,.52],[.49,.01],[.60,.01],[.57,.32],[.56,.53],[.60,.72],[.72,.87],[.88,.98],[.01,.98]]],
 'gironde_129':[
  [[.478,.01],[.552,.01],[.537,.27],[.545,.58],[.58,.98],[.381,.98],[.386,.64],[.40,.32]],
  [[.476,.01],[.555,.01],[.542,.28],[.543,.58],[.568,.98],[.379,.98],[.384,.64],[.399,.32]],
  [[.475,.01],[.566,.01],[.548,.28],[.548,.55],[.575,.98],[.375,.98],[.381,.60],[.397,.32]],
  [[.47,.01],[.555,.01],[.542,.30],[.544,.58],[.57,.98],[.377,.98],[.382,.64],[.397,.32]],
  [[.473,.01],[.554,.01],[.540,.30],[.546,.58],[.573,.98],[.379,.98],[.386,.64],[.399,.32]]],
 'nantes_005':[
  [[.407,.02],[.592,.02],[.676,.30],[.73,.52],[.94,.98],[.43,.98]],
  [[.407,.02],[.60,.02],[.68,.30],[.73,.56],[.88,.98],[.431,.98]],
  [[.406,.02],[.60,.02],[.674,.30],[.71,.50],[.81,.98],[.453,.98],[.437,.63]],
  [[.404,.02],[.60,.02],[.669,.30],[.71,.54],[.788,.98],[.433,.98]],
  [[.402,.02],[.60,.02],[.67,.30],[.71,.56],[.785,.98],[.422,.98]]],
 'nantes_045':[
  [[.238,.02],[.70,.02],[.714,.40],[.715,.98],[.425,.98],[.316,.50]],
  [[.247,.02],[.704,.02],[.713,.4],[.721,.98],[.422,.98],[.328,.5]],
  [[.22,.02],[.704,.02],[.714,.4],[.733,.98],[.433,.98],[.321,.5]],
  [[.20,.02],[.704,.02],[.714,.4],[.717,.98],[.392,.98],[.298,.5]],
  [[.215,.02],[.704,.02],[.714,.4],[.705,.98],[.407,.98],[.303,.5]]],
 'bourges_003':[
  [[.449,.01],[.59,.01],[.596,.5],[.617,.98],[.415,.98],[.433,.5]],
  [[.451,.01],[.59,.01],[.599,.5],[.618,.98],[.418,.98],[.434,.5]],
  [[.449,.01],[.579,.01],[.598,.5],[.621,.98],[.414,.98],[.432,.5]],
  [[.445,.01],[.561,.01],[.592,.5],[.617,.98],[.414,.98],[.431,.5]],
  [[.455,.01],[.572,.01],[.59,.5],[.62,.98],[.418,.98],[.433,.5]]],
 'bourges_045':[
  [[.444,.01],[.574,.01],[.602,.5],[.636,.98],[.446,.98],[.443,.5]],
  [[.45,.01],[.576,.01],[.605,.5],[.637,.98],[.444,.98],[.445,.5]],
  [[.463,.01],[.587,.01],[.609,.5],[.63,.98],[.445,.98],[.447,.5]],
  [[.451,.01],[.585,.01],[.607,.5],[.631,.98],[.431,.98],[.438,.5]],
  [[.451,.01],[.576,.01],[.598,.5],[.618,.98],[.439,.98],[.443,.5]]]
}


def van_ego(name,j):
    if name=='nantes_005':
        top=[.098,.079,.076,.051,.047][j]
        body=[[.386,top+.025],[.414,top+.007],[.454,top],[.524,top],[.564,top+.013],[.586,top+.043],[.611,.352],[.611,.406],[.585,.451],[.548,.482],[.503,.491],[.464,.485],[.434,.456],[.414,.407]]
        boom=[[.558,.393],[.586,.342],[.634,.241],[.674,.173],[.682,.133],[.721,.127],[.743,.148],[.741,.183],[.706,.218],[.676,.288],[.619,.374],[.594,.405]]
    else:
        body=[[.333,0],[.701,0],[.665,.22],[.642,.36],[.624,.415],[.59,.455],[.56,.477],[.49,.483],[.453,.464],[.428,.427],[.408,.37]]
        boom=[[.572,.368],[.603,.298],[.728,.093],[.779,0],[.835,0],[.779,.115],[.675,.302],[.616,.364]]
    return [body,boom]

MOVING={
 'nantes_005':[[box(0,.78,.29,1)],[box(0,.74,.28,1)],[box(0,.69,.26,1)],[box(0,.64,.25,1)],[box(0,.60,.25,.93)]],
 'nantes_045':[[box(.552,.742,.74,1),box(0,.67,.19,.86),box(.10,.94,.24,1)],
  [box(.549,.745,.72,1),box(0,.54,.17,.68),box(.08,.9,.24,1)],
  [box(.537,.775,.681,1),box(0,.83,.27,1)],
  [box(.515,.793,.674,1),box(0,.40,.10,.63)],
  [box(.493,.789,.65,1),box(0,.47,.145,.65),box(0,.73,.242,.94)]]
}

REGIONS={
 'gironde_013':{'cross_junction':[box(.08,.64,.37,.86)],'rear_lane':[box(.40,.70,.73,.93)]},
 'gironde_129':{'forward':[box(.40,.07,.57,.29)],'rear':[box(.38,.66,.58,.94)]},
 'nantes_005':{'near_rear':[box(.44,.575,.74,.775)],'far_rear':[box(.46,.79,.81,.98)]},
 'nantes_045':{'near_rear':[box(.42,.575,.69,.755)],'far_left_rear':[box(.43,.79,.54,.98)]},
 'bourges_003':{'forward':[box(.43,.10,.59,.31)],'rear':[box(.42,.65,.61,.93)]},
 'bourges_045':{'forward':[box(.46,.265,.58,.35)],'rear':[box(.445,.65,.61,.94)]}
}

NOTES={
 'gironde_013':['Rural junction with lane/yield markings; changing yaw and side branch.','Signs, poles, vegetation and verge excluded; no traffic visible in reviewed road polygons.'],
 'gironde_129':['Narrow rural asphalt with central/edge markings and roadside bins.','Roadside bins/fences outside ground polygons; no traffic visible in reviewed road.'],
 'nantes_005':['Divided road entering bridge shadow; large van and mast; opposing moving car.','Use host carriageway only; barrier, opposite lane and vegetation outside ground. Bridge and ego shadows excluded per frame. Thin forward projected structure above roof excluded as uncertain.'],
 'nantes_045':['Marked main carriageway with tar-repair lines; following car and opposite traffic.','Large ego van extends beyond nadir top; footprint is view-clipped. Only host carriageway eligible; shadows/traffic excluded.'],
 'bourges_003':['Rural road with edge paint, large nadir graphic, hood and thin antenna.','Fields/verges excluded; no traffic visible in reviewed road.'],
 'bourges_045':['Residential road with graphic/hood/antenna, roadside walls and speed cushions.','Sidewalks, walls, gates, raised speed cushions and tree/pole shadows excluded; current road-plane model is not valid on the cushions.']}


def shadows(name,j):
    if name=='nantes_005':
        ego=[[.438,.465],[.552,.465],[.527,.551],[.45,.561]]
        boundary=[None,[(0,0),(1,0),(1,.065),(.7,.065),(.4,.15),(0,.25)],[(0,0),(1,0),(1,.23),(.7,.23),(.4,.35),(0,.50)],[(0,0),(1,0),(1,.50),(.75,.50),(.4,.59),(0,.66)],[(0,0),(1,0),(1,.80),(.8,.80),(.4,.87),(0,.91)]][j]
        extra=[box(.68,.41,.74,.49),box(.72,.61,.79,.71)] if j==2 else []
        return [ego]+([boundary] if boundary else [])+extra
    if name=='nantes_045':
        right=[.70,.69,.71,.70,.70][j]
        return [[[.438,.462],[.55,.462],[.527,.551],[.45,.556]],[[right,0],[1,0],[1,1],[right-.018,1],[right-.006,.6]]]
    if name=='bourges_045':
        return [[[[.59,.87],[.66,.83],[.70,1],[.59,1]]],[[[.57,.64],[.65,.64],[.68,1],[.50,1],[.50,.79]]],[[[.56,.24],[.63,.24],[.65,.61],[.58,.61],[.55,.53]]],[[[.53,0],[.64,0],[.64,.17],[.55,.15]]],[[[.55,0],[.64,0],[.64,.30],[.58,.30],[.55,.16]]]][j]
    return []


def build():
    selection=ROOT/'controlled_v2/annotation_selection.json';sel=json.loads(selection.read_text());out=ROOT/'controlled_v2/annotations';out.mkdir(exist_ok=True)
    for t in sel['targets']:
        name=f"{t['area']}_{t['index']:03d}";manifest=json.loads((ROOT/t['source_manifest']['path']).read_text());metadata={r['id']:r for r in manifest['frames']};frames={};names=['prev2','prev1','target','next1','next2']
        for j,(n,r) in enumerate(zip(names,t['frames'])):
            m=metadata[r['id']];uncertain=[]
            if name=='nantes_005':uncertain=[box(.49,0,.56,.125)]
            if t['area']=='nantes':uncertain.append(box(.46,.54,.50,.605))
            if name=='bourges_045' and j==0:uncertain=[box(.44,.75,.65,.96)]
            if name=='bourges_045' and j==2:uncertain=[box(.45,.105,.615,.248)]
            if name=='bourges_045' and j==3:uncertain=[box(.49,.035,.52,.075)]
            frames[n]=dict(id=r['id'],image=r['image_path'],sample_sha256=r['sample_sha256'],metadata_path=m['metadata_path'],metadata_sha256=m['metadata_sha256'],license=r['license'],producer=r['producer'],capture_time=m['capture_time'],
                ground=[GROUND[name][j]],ego=G_EGO if t['area']=='gironde' else B_EGO if t['area']=='bourges' else van_ego(name,j),
                moving=MOVING.get(name,[[],[],[],[],[]])[j],shadow=shadows(name,j),privacy=[],uncertain=uncertain)
        a=dict(version=1,status='draft_pending_overlay_review',scene_id=name,selection=ref(selection),source=ref(Path(__file__)),source_manifest=t['source_manifest'],
            collection=manifest['selection']['collection'] if 'selection' in manifest else manifest['frames'][0]['collection'],split=t['split'],target_id=t['target_id'],
            projection=dict(size=2200,fov=163.302054963582),boundary_margin_px=20,ground_margin_px=8,frame_order=names,frames=frames,regions=REGIONS[name],
            review=dict(reviewer='Codex visual source annotation',views='All five600px grid views plus target1200px raw nadir and target spherical context.',scope='Conservative selected road surfaces only; empty category is no observed nuisance in eligible scope, not an exhaustive whole-sphere label.',notes=NOTES[name]),
            footprint_interpretation='Conservative annotated obstruction envelope in the finite nadir view; not exact physical car dimensions or a clean reference.',no_model_outputs_used=True)
        path=out/(name+'.json')
        if path.exists() and json.loads(path.read_text())['status']!='draft_pending_overlay_review':raise RuntimeError('Already reviewed; create a new version')
        save_json(path,a)
    print('Prepared6 draft annotations for overlay review')


if __name__=='__main__':build()
