"""Internal source-annotation QA only; no matching, recovery or model output."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from benchmark import ROOT,digest
from repair import nadir
from run_dense import heavy_job,ref,read_ref
from dense_experiment import immutable_json


def build():
    selection=ROOT/'controlled_v2/annotation_selection.json';a=json.loads(selection.read_text());out=ROOT/'controlled_v2/review';out.mkdir(exist_ok=True);records=[]
    with heavy_job():
        for t in a['targets']:
            name=f"{t['area']}_{t['index']:03d}";folder=out/name;folder.mkdir(exist_ok=True)
            sheet=Image.new('RGB',(1800,1300),(20,20,20));d=ImageDraw.Draw(sheet)
            for j,r in enumerate(t['frames']):
                source=ROOT/r['image_path']
                if digest(source)!=r['sample_sha256']:raise RuntimeError('Image changed')
                raw=nadir(np.asarray(Image.open(source).convert('RGB')),1200,163.302054963582)
                path=folder/f"{r['index']:03d}.jpg";Image.fromarray(raw).save(path,quality=95)
                im=Image.fromarray(raw).resize((600,600));draw=ImageDraw.Draw(im)
                for k in range(1,10):
                    x=k*60;draw.line((x,0,x,599),fill=(125,125,125),width=1);draw.line((0,x,599,x),fill=(125,125,125),width=1)
                    draw.text((x+2,3),str(k/10),fill='yellow');draw.text((3,x+2),str(k/10),fill='yellow')
                x=(j%3)*600;y=(j//3)*650;sheet.paste(im,(x,y+25));d.text((x+8,y+5),f"{name} frame{r['index']} {'TARGET' if j==2 else 'donor'}",fill='white')
                records.append(dict(area=t['area'],anchor=t['index'],frame=r['index'],source=dict(path=r['image_path'],sha256=r['sample_sha256']),review=ref(path),license=r['license'],producer=r['producer']))
            target=ROOT/t['frames'][2]['image_path'];full=Image.open(target).convert('RGB');full.thumbnail((600,300));sheet.paste(full,(1200,900));d.text((1210,870),'Target sphere / context',fill='white')
            sheet.save(out/(name+'_sheet.jpg'),quality=95)
    immutable_json(out/'provenance.json',dict(selection=ref(selection),source=ref(Path(__file__)),projection=dict(size=1200,fov=163.302054963582),frames=records,purpose='Internal manual annotation review, no model outputs.'))
    print('Prepared30 source views and6 context sheets')


if __name__=='__main__':build()
