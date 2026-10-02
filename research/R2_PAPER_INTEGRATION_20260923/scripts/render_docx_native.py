"""Windows native Word PDF rendering when the bundle contains no LibreOffice.

The packaged render_docx.py was tried and failed with missing soffice.exe.
This explicitly records the alternative engine and creates the same page-PNG
inspection output without installing or using desktop LibreOffice.
"""
from pathlib import Path
import argparse,subprocess,json
import fitz
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output_dir',required=True);p.add_argument('--dpi',type=int,default=144);a=p.parse_args()
source=Path(a.source).resolve();out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
pdf=out/(source.stem+'.pdf')
result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-File',str(Path(__file__).with_name('export_word_pdf.ps1')),
    '-Source',str(source),'-Pdf',str(pdf)],capture_output=True,text=True,encoding='utf-8',errors='replace')
if result.returncode:raise RuntimeError(result.stdout+'\n'+result.stderr)
pages=[]
with fitz.open(pdf) as doc:
    for i,page in enumerate(doc):
        target=out/f'page-{i+1}.png';page.get_pixmap(dpi=a.dpi,alpha=False).save(target)
        pages.append(dict(page=i+1,image=str(target),width_pt=page.rect.width,height_pt=page.rect.height,text_chars=len(page.get_text())))
(out/'render_manifest.json').write_text(json.dumps(dict(source=str(source),engine='Microsoft Word native export',
    fallback_reason='Packaged render_docx.py: missing LibreOffice soffice.exe; no bundled Windows LibreOffice',pages=pages),indent=2))
print('RENDERED',len(pages),'PAGES',flush=True)
