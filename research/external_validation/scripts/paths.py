from pathlib import Path
import hashlib,json
WORK=Path(__file__).resolve().parents[2]
BASE=WORK/'external_validation'
PAPER=WORK/'新论文_市场可交付灵活性_20260917'
REPO=WORK/'power-dc-vpp'
FINAL=WORK/'EXTERNAL_VALIDATION_FINAL_20260922'
DATA=BASE/'data'
RESULTS=FINAL/'01_results'
REPORTS=FINAL/'06_reports'
for p in [DATA,RESULTS,REPORTS,BASE/'runs',BASE/'qa',FINAL/'02_figures/source_data',FINAL/'03_scripts',FINAL/'04_manuscript',FINAL/'05_supplementary']:p.mkdir(parents=True,exist_ok=True)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()
def save(p,value):Path(p).write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str,allow_nan=False),encoding='utf-8')
