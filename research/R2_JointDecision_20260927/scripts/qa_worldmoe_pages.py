"""Rasterize native Word exports and retain concise layout diagnostics."""
from pathlib import Path
import json, math
import fitz
from PIL import Image, ImageOps, ImageDraw
ROOT=Path(__file__).resolve().parents[2]/'WorldMoE_Revision_20260928'
records=[]
for pdf in (ROOT/'qa').glob('WorldMoE_*.pdf'):
    doc=fitz.open(pdf);folder=pdf.parent/pdf.stem;folder.mkdir(exist_ok=True);thumbs=[];issues=[]
    for i,page in enumerate(doc):
        pix=page.get_pixmap(matrix=fitz.Matrix(1.7,1.7),alpha=False);path=folder/f'page-{i+1:02d}.png';pix.save(path)
        image=Image.open(path).convert('RGB');thumb=ImageOps.contain(image,(306,400))
        canvas=Image.new('RGB',(322,426),'#dce2e6');canvas.paste(thumb,((322-thumb.width)//2,20));ImageDraw.Draw(canvas).text((9,5),f'Page {i+1}',fill='black');thumbs.append(canvas)
        for word in page.get_text('words'):
            x0,y0,x1,y1,text,*_=word
            if x0<24 or x1>page.rect.width-24 or y0<15 or y1>page.rect.height-15:issues.append(dict(page=i+1,text=text,bbox=[x0,y0,x1,y1]))
    for start in range(0,len(thumbs),12):
        part=thumbs[start:start+12];sheet=Image.new('RGB',(322*3,426*math.ceil(len(part)/3)),'white')
        for j,thumb in enumerate(part):sheet.paste(thumb,((j%3)*322,(j//3)*426))
        sheet.save(folder/f'contact-{start//12+1:02d}.png')
    records.append(dict(file=pdf.name,pages=len(doc),page_images=str(folder),margin_exceptions=issues))
(ROOT/'qa/page_layout.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records,indent=2))
