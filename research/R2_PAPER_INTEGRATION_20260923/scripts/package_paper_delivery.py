"""Package measured paper artifacts without copying large world checkpoints."""
from pathlib import Path
import hashlib,json,re,shutil,zipfile
from datetime import datetime,timezone
from lxml import etree
from docx import Document

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parent
PAPER=WORK/'新论文_市场可交付灵活性_20260917'
OUT=PAPER/'交付稿_20260924_R2_APIC_完整修订'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')

freeze=read(ROOT/'frozen_candidate/EVALUATION_FREEZE.json')
for item in freeze['files']:
    assert digest(item['path'])==item['sha256'],'Frozen artifact changed: '+item['path']
assert read(ROOT/'results/PAPER_SUPPORT_STATUS.json')['status']=='complete'
main=OUT/'NatureEnergy_ExecutableFlexibility_R2_APIC_EN.docx'
supp=OUT/'Supplementary_ExecutableFlexibility_R2_APIC_EN.docx'
reports=[]
for path,figures,tables,render in [(main,6,1,'main_final_pages'),(supp,0,21,'supplement_final_pages')]:
    doc=Document(path)
    assert len(doc.inline_shapes)==figures
    assert len(doc.tables)==tables
    with zipfile.ZipFile(path) as z:
        texts=[]
        for name in z.namelist():
            if name.startswith('word/') and name.endswith('.xml'):
                xml=etree.fromstring(z.read(name))
                texts.extend(xml.xpath('//*[local-name()="t"]/text()'))
        assert not re.search('dart',''.join(texts),re.I),str(path)
    manifest=read(ROOT/'tmp'/render/'render_manifest.json')
    assert Path(manifest['source'])==path
    reports.append(dict(path=str(path),sha256=digest(path),pages=len(manifest['pages']),figures=figures,tables=tables,author_visible_old_acronym=0))

stems=['Fig1_new_R2_architecture','Fig2_service_resource','Fig3_cost_and_information','Fig4_value_and_shifted_energy','Fig5_service_recourse','Fig6_external_mechanisms']
for stem in stems:
    svg=(ROOT/'figures'/f'{stem}.svg').read_text(encoding='utf-8')
    assert 'DART' not in svg
    for suffix in ['png','pdf','svg']:assert (ROOT/'figures'/f'{stem}.{suffix}').is_file()

reproduce='''# R2 / APIC paper delivery — 24 September 2026

The main manuscript, supplement, figures and table labels use one terminology contract.
APIC means anchored price-identity coupling. D denotes the DA–RT spread.
R2 is the proposed price-and-cost module; Previous R2 names the earlier frozen price-only comparator.

## Contents

- Main manuscript: 17 pages, six figures and one main table.
- Supplement: 11 pages and 21 measured supporting tables.
- Figures_and_Source_Data.zip: six figures as PNG, editable SVG and PDF; all panel-source CSVs; captions; palette and naming files.
- R2_APIC_Research_Companion.zip: experiment and authoring code, measured result tables and run records, three selected R2 heads and native calibration, frozen configuration and lineage.

## Evidence chronology

Price-model training uses 2021–2023. Checkpoint selection, calibration and operating-configuration selection use the registered 2024 splits. Primary seed: 20260920; retained replication seeds: 20260921 and 20260922. Selected epochs: 11, 0 and 3. Epoch 0 is the fitted Base-equivalent state selected by the same proper-score rule; it is retained in all summaries.

The freeze occurred at 2026-09-23 13:02:40 UTC. The user then-authorized 2025 annual replay and necessary observed-price early-2026 settlement tail use fixed models, calibrations and policy. The study describes 2025 as retrospective final-year evaluation; 2024 replications remain development sensitivities. The withdrawn 15% cost-search target is not an acceptance condition for this delivery.

The primary fixed policy is the 24 h interval configuration with the reservation-state cost head and a causal update of issued PV paths using the preceding four completed hours. Execution and accounting functions are unchanged. All three seeds meet primary deadlines with zero closing IT work. Expanded mixed-contract stress cases retain their measured residual violations in the tables and heatmaps.

## Rebuild in the registered workspace

Run from `C:/Users/Kali/Desktop/电力市场`. Scientific packages use `C:/Users/Kali/anaconda3/python.exe` (Python 3.13.5, PyTorch 2.9.0+cu126). Word authoring uses the bundled Codex Python named below.

```powershell
& './R2_PAPER_INTEGRATION_20260923/scripts/run.ps1' -Script 'scripts/collect_paper_evidence.py'
& './R2_PAPER_INTEGRATION_20260923/scripts/run.ps1' -Script 'scripts/build_architecture_figure.py'
& './R2_PAPER_INTEGRATION_20260923/scripts/run.ps1' -Script 'scripts/build_dense_figures.py' --figures 2 3 4 5 6
& './R2_PAPER_INTEGRATION_20260923/scripts/run.ps1' -Script 'scripts/summarize_forecast_scores.py'
$env:PYTHONUTF8='1'
& 'C:/Users/Kali/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' './R2_PAPER_INTEGRATION_20260923/scripts/build_manuscripts.py'
```

The code retains the registered workspace dependencies: `power-dc-vpp`, the earlier paper revisions, `external_validation`, the original Base/control branch, and the primary-seed redesign branch. Input and checkpoint paths/hashes are retained in the freeze, source manifests and `src/runtime.py`. The companion is an inspectable research snapshot, not a standalone copy of the entire data repository. Large WorldMoE weights, weather/market corpora, prediction caches and cohort trajectory files stay at their identified local source paths. `LOCAL_INPUTS.json` identifies these locations. The three small selected R2 heads and calibration files are included byte-for-byte.

Code and frozen checkpoint keys retain their internal historical schemas for compatibility. Author-visible manuscript text, native equations, figure labels, captions and displayed metrics use APIC and spread terminology. Numerical arrays and model parameters were not renamed or retuned.

## Figures and measurements

Fig. 2 uses 11 measured capacity fractions and 19 integer-hour service windows. Figs. 3–5 retain the hourly contract grid; Fig. 3 also has 53 weekly score observations and paired seven-day block intervals. Fig. 5 uses hourly cumulative lateness and actual deadline-share tests. Fig. 6 contains four trace/year comparisons, 11 capacity fractions, 19 service windows and measured late-work heatmaps. Difference strips and marker styles expose overlaps without shifting data coordinates.

Every series color comes from the supplied eight-color card: #B6B3D6, #CFCCE3, #D5D3DE, #D5D1D1, #F6DFD6, #F8B2A2, #F1837A, #E9687A. Axes and fine keylines use neutral ink. Figure captions distinguish primary-seed, three-seed ranges and paired confidence intervals.

## Document rendering

Native Microsoft Word exported the final DOCX files; bundled PyMuPDF rendered their pages for visual inspection. This is the Windows fallback because the packaged document renderer has no bundled `soffice.exe`. QA page images are kept in the experiment's `tmp` directory, outside the delivery. Original manuscript and previous delivery versions are preserved.
'''
(OUT/'REPRODUCE.md').write_text(reproduce,encoding='utf-8')
terms=read(ROOT/'configs/terminology_contract.json')
guide='# 统一命名 / Terminology\n\n| 对象 | 统一名称 |\n|---|---|\n| 框架 | R-ECMoE-EF |\n| 新模型 | R2 |\n| 冻结世界主干 | WorldMoE (W1) |\n| 价格耦合方法 | APIC — anchored price-identity coupling |\n| 价差量 | DA–RT spread，符号 D，D = DA − RT |\n'
for key,value in terms['display_labels'].items():guide+=f'| {key} | {value} |\n'
guide+='\n指标名称：service-adjusted electricity cost；electricity cost saving；shifted IT energy；late IT energy；closing IT work；service-adjusted daily-loss CVaR95；spread pinball loss。\n\n正文首次写全称，后续使用统一简称。APIC 是方法名，D 是价差量。表格列宽有限时采用明确缩写，单位在表头、表注或补充材料定义处给出。\n'
(OUT/'TERMINOLOGY.md').write_text(guide,encoding='utf-8')

inputs={
 'workspace':str(WORK),'experiment':str(ROOT),
 'original_main':r'C:\Users\Kali\Downloads\NatureEnergy_ExecutableFlexibility_FINAL_REFINED_20260923.docx',
 'original_supplement':str(WORK/'EXTERNAL_VALIDATION_FINAL_20260922/05_supplementary/Supplementary_ExecutableFlexibility_20260922_EXTVALID_FINAL.docx'),
 'physical_and_external_inputs':str(WORK/'external_validation'),
 'predictions':str(ROOT/'results/predictions_2025'),
 'cohort_trajectories':str(ROOT/'results/economic_trajectories'),
 'frozen_artifacts':freeze['files'],
 'executor_sources':read(ROOT/'configs/paper_executor_sources.json'),
 'world_lineage':read(ROOT/'configs/selected_w1.json')}
save(OUT/'LOCAL_INPUTS.json',inputs)

with zipfile.ZipFile(OUT/'Figures_and_Source_Data.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for stem in stems:
        for ext in ['png','svg','pdf']:
            p=ROOT/'figures'/f'{stem}.{ext}';z.write(p,'figures/'+p.name)
    for p in sorted((ROOT/'figures/source_data').glob('*.csv')):z.write(p,'source_data/'+p.name)
    for name in ['visual_palette.json','terminology_contract.json','concise_figure_captions.json']:
        z.write(ROOT/'configs'/name,'style/'+name)
    for name in ['build_dense_figures.py','build_architecture_figure.py']:z.write(ROOT/'scripts'/name,'scripts/'+name)
    z.write(OUT/'REPRODUCE.md','REPRODUCE.md');z.write(OUT/'TERMINOLOGY.md','TERMINOLOGY.md')

with zipfile.ZipFile(OUT/'R2_APIC_Research_Companion.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    added=set()
    def include(p,arc):
        p=Path(p);arc=str(arc).replace('\\','/')
        if arc not in added and p.is_file():z.write(p,arc);added.add(arc)
    for folder in ['src','vendor','scripts','configs','frozen_candidate']:
        for p in sorted((ROOT/folder).rglob('*')):
            if '__pycache__' not in p.parts and p.suffix in ['.py','.ps1','.json','.md','.yaml','.yml']:
                include(p,p.relative_to(ROOT))
    for p in sorted((ROOT/'results').glob('*')):
        if p.suffix in ['.csv','.json']:include(p,p.relative_to(ROOT))
    for p in sorted((ROOT/'results/paper_runs').glob('*.json')):include(p,p.relative_to(ROOT))
    for p in sorted((ROOT/'figures/source_data').glob('*.csv')):include(p,p.relative_to(ROOT))
    for p in (ROOT/'cache').glob('*.json'):include(p,p.relative_to(ROOT))
    for item in freeze['files']:
        p=Path(item['path'])
        if p.parent.name=='checkpoints':include(p,'checkpoints/'+p.name)
    for name in ['frozen_2025_manifest.json','PROTOCOL.json']:include(ROOT/name,name)
    for name in ['REPRODUCE.md','TERMINOLOGY.md','LOCAL_INPUTS.json']:include(OUT/name,name)

audit=dict(status='complete',updated_utc=datetime.now(timezone.utc).isoformat(),documents=reports,
    frozen_artifact_count=len(freeze['files']),frozen_hashes_match=True,
    figures=6,figure_formats=['PNG','SVG','PDF'],supplementary_tables=21,
    final_visual_review='All 17 main and 11 supplementary pages inspected; subsequent changed pages inspected separately.',
    support=read(ROOT/'results/PAPER_SUPPORT_STATUS.json'),primary=read(ROOT/'results/PAPER_PRIMARY_RESULT.json'))
save(ROOT/'results/FINAL_DELIVERY_AUDIT.json',audit)
files=[dict(name=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(OUT.iterdir()) if p.is_file()]
save(OUT/'DELIVERY_MANIFEST.json',dict(status='complete',documents=reports,files=files))
current=PAPER/'CURRENT_MANUSCRIPT.json'
backup=ROOT/'results/PREVIOUS_CURRENT_MANUSCRIPT.json'
if not backup.exists():shutil.copy2(current,backup)
save(current,dict(status='complete',title='Executable service commitments unlock the market value of computing flexibility',
 directory=str(OUT),delivery=str(OUT),revision=str(ROOT),study=str(ROOT),source_experiment=str(ROOT/'results'),
 scope='frozen_R2_APIC_three_seed_and_external_evaluation',manuscript=str(main),main_file=str(main),
 supplement=str(supp),supplement_file=str(supp),primary_language='EN',
 previous_main_file=inputs['original_main'],R2_seeds=freeze['seeds'],R2_primary_seed=20260920,
 R2_epochs=[11,0,3],world_seed=20240829,w1_epoch=80,pages={'main':reports[0]['pages'],'supplement':reports[1]['pages']},
 scientific_counts={'main_figures':6,'main_tables':1,'supplementary_tables':21,'main_references':15},
 original_W0_W1_training_modified=False,checkpoints_modified=False,files=files))
print(json.dumps(dict(status='complete',delivery=str(OUT),pages=[r['pages'] for r in reports],frozen_hashes=len(freeze['files']),files=[f['name'] for f in files]),ensure_ascii=False,indent=2))
