"""Build a local, offline experiment viewer and a partial 360-degree export."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from repair import paste_nadir

ROOT=Path(__file__).resolve().parent


def main():
    preview=ROOT/'preview';preview.mkdir(exist_ok=True)
    experiments=[]
    for folder,title in [('wide_results','Marked road · wider view'),
                         ('road_results','Marked road · narrow view'),
                         ('results','Repetitive paving · failure case')]:
        report=json.loads((ROOT/folder/'report.json').read_text())
        # Consistent central 1024-pixel crop across both projection sizes.
        n=report['configuration']['size'];margin=(n-1024)//2
        crop=(margin,margin,margin+1024,margin+1024)
        for name in ['target','prev','next']:
            im=Image.open(ROOT/folder/f'{name}_nadir.jpg').crop(crop)
            im.resize((700,700),Image.Resampling.LANCZOS).save(preview/f'{folder}_{name}.jpg',quality=92)
        for case in report['cases']:
            for mode in ['input','repaired','telea','provenance','evidence']:
                im=Image.open(ROOT/folder/case['name']/f'{mode}.png').crop(crop)
                im.resize((700,700),Image.Resampling.LANCZOS).save(preview/f'{folder}_{case["name"]}_{mode}.jpg',quality=93)
        experiments.append(dict(folder=folder,title=title,report=report))
    wide=experiments[0]['report'];fov=wide['configuration']['fov']
    original=np.array(Image.open(ROOT/'road_sample/target.jpg').resize((2048,1024),Image.Resampling.LANCZOS))
    patch=np.array(Image.open(ROOT/'wide_results/vehicle_footprint/repaired.png'))
    accepted=np.load(ROOT/'wide_results/vehicle_footprint/evidence.npz')['accepted']
    restored,supported=paste_nadir(original,patch,accepted,fov)
    Image.fromarray(original).save(preview/'panorama_original.png')
    Image.fromarray(restored).save(preview/'panorama_partial_repair.png')
    Image.fromarray(supported.astype(np.uint8)*255).save(preview/'panorama_transfer_mask.png')
    # An exportable, labeled figure. No claim of ground truth under the car.
    figure=Image.new('RGB',(1800,740),'#101923');draw=ImageDraw.Draw(figure)
    try:
        font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',24)
        small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',19)
    except OSError:
        font=ImageFont.load_default(size=24);small=ImageFont.load_default(size=19)
    panels=[('Observed target','wide_results_target.jpg'),
            ('Partial transfer · 55.6% of mask','wide_results_vehicle_footprint_repaired.jpg'),
            ('Donor map · orange = unresolved','wide_results_vehicle_footprint_provenance.jpg')]
    for i,(label,file) in enumerate(panels):
        draw.text((i*600+18,18),label,font=font,fill='#e9f1f5')
        figure.paste(Image.open(preview/file).resize((580,580)),(i*600+10,60))
    draw.text((20,660),'Every accepted pixel has one donor only. No ground truth is available beneath the vehicle.',font=small,fill='#e7bc75')
    draw.text((20,696),'Images: Eurométropole de Strasbourg · captured 1 Nov 2024 · Panoramax IGN · Licence Ouverte 2.0 · modified',font=small,fill='#a7b8c4')
    figure.save(ROOT/'comparison.jpg',quality=94)
    template=(ROOT/'docs/report_template.html').read_text()
    (ROOT/'report.html').write_text(template.replace('__EXPERIMENT_DATA__',json.dumps(experiments).replace('</','<\\/')))
    print('Built report.html, comparison.jpg, and preview/panorama_partial_repair.png')


if __name__=='__main__':main()
