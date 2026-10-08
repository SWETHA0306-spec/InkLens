from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont, ImageFilter
ROOT=Path(__file__).parent; DATA=ROOT/'data'; DATA.mkdir(exist_ok=True)
try: FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf', 28)
except: FONT=ImageFont.load_default()
def make(name,title,desc,seed):
    im=Image.new('RGB',(1200,720),(246,244,236)); d=ImageDraw.Draw(im)
    d.rectangle((38,35,1160,680),outline=(210,205,193),width=2)
    d.text((76,68),'CLINIC NOTE / '+seed,fill=(92,86,80),font=FONT)
    lines=[('Rx Paracetamol 500mg t.i.d.',(78,185)),('Wrong dosage 1000mg',(78,315)),('follow-up in 7 days',(78,445))]
    for i,(txt,pos) in enumerate(lines):
        d.text((pos[0]+(i%2)*5,pos[1]),txt,fill=(34,39,48),font=FONT,stroke_width=1)
    d.line((68,330,690,300),fill=(170,45,52),width=8); d.line((70,300,680,335),fill=(170,45,52),width=5)
    d.text((930,390),'Check BP\nfirst!',fill=(115,47,74),font=FONT,spacing=8)
    d.line((902,380,1120,365),fill=(115,47,74),width=2)
    im=im.filter(ImageFilter.GaussianBlur(0.35)); im.save(DATA/(name+'.png'))
    (DATA/(name+'.json')).write_text(json.dumps({'id':name,'name':name+'.png','title':title,'description':desc,'url':'/data/'+name+'.png'}),encoding='utf-8')
make('hard_clinic_note','Clinic note / mixed layout','crossed-out dosage + margin note','CASE 01')
make('hard_prescription','Prescription / cramped note','low-quality medical scrawl simulation','CASE 02')
print('Generated 2 hard samples in data/')
