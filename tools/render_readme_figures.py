"""Render the README figures from committed, licensed crops and measured evidence.

This regenerates presentation assets, not reconstruction or benchmark scores.
Requires Pillow (in requirements.txt). No network, models or source selection.
"""
import hashlib,json,math
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'docs/figures/beijing'
INK='#20272d';MUTED='#68737c';BG='#ffffff';LINE='#d2d8dd'
BLUE='#397ca6';ORANGE='#b87842';GREEN='#5b8a6e';AMBER='#bc9350';VIOLET='#877c9a';GRAY='#58606a'

def font(size):return ImageFont.load_default(size=size)

def text(draw,xy,value,size=22,fill=INK,anchor=None):draw.text(xy,value,font=font(size),fill=fill,anchor=anchor)

class RetinaDraw:
    def __init__(self,image):self.draw=ImageDraw.Draw(image)
    def coords(self,xy):
        return [(x*2,y*2) for x,y in xy] if isinstance(xy[0],(tuple,list)) else tuple(v*2 for v in xy)
    def text(self,xy,value,**kw):
        kw['font']=font(kw['font'].size*2);self.draw.text(self.coords(xy),value,**kw)
    def line(self,xy,**kw):
        kw['width']=kw.get('width',1)*2;self.draw.line(self.coords(xy),**kw)
    def ellipse(self,xy,**kw):
        kw['width']=kw.get('width',1)*2;self.draw.ellipse(self.coords(xy),**kw)
    def rectangle(self,xy,**kw):self.draw.rectangle(self.coords(xy),**kw)
    def polygon(self,xy,**kw):self.draw.polygon(self.coords(xy),**kw)

def triptych(images,status=None):
    panel=464;gap=18;pad=24;w=3*panel+2*gap+2*pad;h=panel+116
    im=Image.new('RGB',(w,h),BG);d=ImageDraw.Draw(im)
    labels=[('Masked target','35° nadir occlusion'),('Astra reconstruction','Raw donor pixels'),('Clean reference','Withheld during reconstruction')]
    for i,(a,(label,sub)) in enumerate(zip(images,labels)):
        x=pad+i*(panel+gap);text(d,(x,12),label,27);text(d,(x,47),sub,18,MUTED)
        im.paste(a.resize((panel,panel),Image.Resampling.LANCZOS),(x,82))
    if status:text(d,(pad,h-22),status,17,MUTED)
    return im

def dashed(draw,points,fill,width=2):
    for i in range(0,len(points)-1,8):draw.line(points[i:i+5],fill=fill,width=width)

def curve(draw,x0,x1,y,depth,color,dash=False):
    pts=[]
    for i in range(101):
        t=i/100;u=1-t
        pts.append((u*u*u*x0+3*u*u*t*x0+3*u*t*t*x1+t*t*t*x1,y+3*u*t*depth))
    if dash:dashed(draw,pts,color,2)
    else:draw.line(pts,fill=color,width=4)
    x,y=pts[-1];draw.polygon([(x,y-2),(x-5,y+8),(x+5,y+8)],fill=color)

def evidence_figure(e):
    im=Image.new('RGB',(2952,880),BG);d=RetinaDraw(im)
    text(d,(24,16),'a  Donor evidence',27);text(d,(24,55),'Five-frame window · capture order',19,MUTED)
    xs=[70,210,350,490,630];y=148
    for i in [0,4,1,3]:
        x=xs[i]
        curve(d,x,xs[2]+(-8 if i<2 else 8),y+18,145 if i in [0,4] else 90,LINE if i in [0,4] else ORANGE if i==1 else BLUE,i in [0,4])
    for i,x in enumerate(xs):
        color=ORANGE if i==1 else BLUE if i==3 else INK if i==2 else MUTED
        d.ellipse((x-20,y-20,x+20,y+20),fill=BG,outline=color,width=3)
        text(d,(x,y),str(i),22,color,'mm')
        label='Target' if i==2 else 'Donor '+str(i) if i in [1,3] else 'Candidate'
        text(d,(x,y-53),label,18,color,'mm')
    shares={r['donor']:r['mask_area_fraction'] for r in e['donor_breakdown']}
    text(d,(165,236),f'{shares[1]*100:.1f}% of mask',20,ORANGE)
    text(d,(420,236),f'{shares[3]*100:.1f}% of mask',20,BLUE)
    d.line((28,322,74,322),fill=BLUE,width=4);text(d,(88,311),'Used in committed output',18,MUTED)
    dashed(d,[(395+i,322) for i in range(46)],LINE,2);text(d,(454,311),'Retrieved, unused',18,MUTED)
    d.line((720,20,720,352),fill='#e7ebee',width=1)
    text(d,(764,16),'b  Recovery breakdown',27);text(d,(764,55),'Fraction of the missing angular area',19,MUTED)
    f=e['fractions'];parts=[('joint_correct',GREEN),('aligned_RGB_failed',AMBER),('accepted_geometry_failed',VIOLET),('abstained',GRAY)]
    x=764;width=678;y0=109
    for k,color in parts:
        right=x+width*f[k];d.rectangle((round(x),y0,round(right),y0+35),fill=color);x=right
    rows=[('Both checks pass','joint_correct',GREEN),('Aligned; RGB error > 10','aligned_RGB_failed',AMBER),('Geometry / support fails','accepted_geometry_failed',VIOLET),('Abstained','abstained',GRAY)]
    for i,(label,key,color) in enumerate(rows):
        y=180+i*42;d.rectangle((764,y,775,y+11),fill=color);text(d,(789,y-7),label,21)
        value=f'{100*f[key]:.1f}%' if f[key]>.01 else f'{100*f[key]:.2f}%';text(d,(1442,y+3),value,22,color,'rm')
    text(d,(24,383),'Geometry: visibility + source-ray error <= 0.5°     RGB: mean absolute channel error <= 10 / 255',19,MUTED)
    text(d,(24,413),'One Beijing development example. Expert comparator; conditional synthetic-geometry scores.',18,MUTED)
    return im

def main():
    e=json.loads((ASSETS/'evidence.json').read_text())
    for r in e['source_crops']:assert hashlib.sha256((ASSETS/r['path']).read_bytes()).hexdigest()==r['sha256']
    assert abs(sum(e['fractions'].values())-1)<1e-10
    images={n:Image.open(ASSETS/'source'/f'{n}.png').convert('RGB') for n in ['damaged','donor3','reconstruction','reference']}
    triptych([images[n] for n in ['damaged','reconstruction','reference']]).save(ASSETS/'comparison.png',optimize=True)
    evidence_figure(e).save(ASSETS/'evidence.png',optimize=True)
    # Discrete assembly stages; deliberately no invented intermediate solver states.
    states=[('damaged','Before transfer'),('donor3','Donor 3 contributes its visible pixels'),('reconstruction','Donor 1 fills the remaining supported region')]
    frames=[triptych([images['damaged'],images[n],images['reference']],label).resize((1107,435),Image.Resampling.LANCZOS) for n,label in states]
    palette_source=Image.new('RGB',(1107,435*3))
    for i,f in enumerate(frames):palette_source.paste(f,(0,435*i))
    palette=palette_source.quantize(colors=256,method=Image.Quantize.MEDIANCUT)
    frames=[f.quantize(palette=palette,dither=Image.Dither.NONE) for f in frames]
    frames[0].save(ASSETS/'assembly.gif',save_all=True,append_images=frames[1:],duration=[1300,1700,4200],loop=0,optimize=False,disposal=2)
    files=['comparison.png','evidence.png','assembly.gif']
    (ASSETS/'render.json').write_text(json.dumps(dict(renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        evidence_sha256=hashlib.sha256((ASSETS/'evidence.json').read_bytes()).hexdigest(),
        files=[dict(path=n,bytes=(ASSETS/n).stat().st_size,sha256=hashlib.sha256((ASSETS/n).read_bytes()).hexdigest()) for n in files],
        animation='Three assembly states from the committed reconstruction, not solver iterations. GIF palette quantization is display-only; PNG and source crops retain RGB.'),indent=2)+'\n')
    print('Rendered comparison, evidence diagram and three-state assembly GIF.')

if __name__=='__main__':main()
