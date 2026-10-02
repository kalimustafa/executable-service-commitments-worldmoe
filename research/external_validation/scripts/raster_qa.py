"""Inspect native Word exports with page geometry and explicit figure/caption checks."""
import json
import fitz
from PIL import Image,ImageDraw
from paths import *
QA=BASE/'qa'
manifest=[]
for pdf in sorted(QA.glob('*FINAL*.pdf')):
    d=fitz.open(pdf);dest=QA/pdf.stem;dest.mkdir(exist_ok=True)
    pages=[];records=[]
    for i,p in enumerate(d):
        image=dest/f'page-{i+1}.png';p.get_pixmap(matrix=fitz.Matrix(100/72,100/72),alpha=False).save(image);pages.append(image)
        words=p.get_text('words');records.append(dict(page=i+1,words=len(words),text=p.get_text(),outside=[list(w[:5]) for w in words if w[0]<18 or w[2]>p.rect.width-18 or w[1]<18 or w[3]>p.rect.height-18]))
    for first in range(0,len(pages),6):
        canvas=Image.new('RGB',(1275,1140),'#DFE3E8');draw=ImageDraw.Draw(canvas)
        for j,image in enumerate(pages[first:first+6]):
            im=Image.open(image);im.thumbnail((425,550));x=(j%3)*425;y=(j//3)*570
            canvas.paste(im,(x,y+20));draw.text((x+8,y+3),f'Page {first+j+1}',fill='black')
        canvas.save(dest/f'overview-{first//6+1}.png')
    save(dest/'page_map.json',records)
    manifest.append(dict(file=pdf.name,pages=len(d),blank_pages=[r['page'] for r in records if r['words']<2],outside_items=sum(len(r['outside']) for r in records)))
word=json.loads((QA/'word_layout_audit.json').read_text(encoding='utf-8-sig'))
for r in word:
    if r['captions']:
        assert len(r['figures'])==len(r['captions'])==6
        for f,cap in zip(r['figures'],r['captions']):assert f['page']==cap['start_page']==cap['end_page'],(r['file'],f,cap)
        assert r['table1']['start_page']==r['table1']['end_page']
        assert r['pages']<30
assert all(not r['blank_pages'] and r['outside_items']==0 for r in manifest)
save(QA/'render_manifest.json',manifest)
print(json.dumps(manifest,indent=2))
