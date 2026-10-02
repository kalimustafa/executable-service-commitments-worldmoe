from pathlib import Path
import json
import fitz
from PIL import Image,ImageDraw,ImageOps
ROOT=Path(__file__).resolve().parents[1];QA=ROOT/'qa';stats=[]
for pdf in QA.glob('*.pdf'):
    label='main' if pdf.name.startswith('NatureEnergy') else 'supplement'
    folder=QA/label;folder.mkdir(exist_ok=True);cards=[]
    with fitz.open(pdf) as d:
        for i,page in enumerate(d):
            image_path=folder/f'page-{i+1}.png';pix=page.get_pixmap(dpi=144,alpha=False);pix.save(image_path)
            im=Image.open(image_path).convert('RGB');thumb=ImageOps.contain(im,(425,550));card=Image.new('RGB',(445,580),'#eeeeee');card.paste(thumb,((445-thumb.width)//2,20));ImageDraw.Draw(card).text((10,3),f'{label} {i+1}',fill='black');cards.append(card)
            stats.append(dict(document=label,page=i+1,text_chars=len(page.get_text()),images=len(page.get_images()),text=page.get_text()))
    for start in range(0,len(cards),6):
        part=cards[start:start+6];sheet=Image.new('RGB',(445*3,580*((len(part)+2)//3)),'white')
        for j,card in enumerate(part):sheet.paste(card,((j%3)*445,(j//3)*580))
        sheet.save(QA/f'{label}-contact-{start//6+1}.png')
(QA/'page_text.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
print({label:sum(x['document']==label for x in stats) for label in ['main','supplement']})
