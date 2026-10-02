"""Assemble the compact final delivery without checkpoints or QA intermediates."""
import json,shutil,zipfile
from datetime import datetime,timezone
from paths import *
def copy(src,dest):
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
def main():
    qa=json.loads((BASE/'qa/render_manifest.json').read_text())
    word=json.loads((BASE/'qa/word_layout_audit.json').read_text(encoding='utf-8-sig'))
    build=json.loads((REPORTS/'document_build_manifest.json').read_text())
    assert all(not q['blank_pages'] and q['outside_items']==0 for q in qa)
    for p in (BASE/'scripts').glob('*'):
        if p.is_file() and p.suffix in ['.py','.ps1']:copy(p,FINAL/'03_scripts'/p.name)
    for name in ['PROTOCOL.json','frozen_2025_manifest.json']:copy(BASE/name,REPORTS/name)
    for name in ['calendar_and_physical_audit_2024.json','forcing_source_manifest.json','frontier_parse_quality.json','physical_day_provenance_2024.csv']:
        copy(DATA/name,REPORTS/name)
    for year in [2024,2025]:
        folder=DATA/f'frontier_{year}'
        for name in [f'frontier-{year}.zip','LICENSE.txt','header.txt','FIELDS.md','download_manifest.json','parse_quality.json','mapped_arrivals.npz','arrival_series.parquet']:
            copy(folder/name,FINAL/'07_data'/f'frontier_{year}'/name)
    for name in ['forcing_2024.npz','forcing_2025.npz','frontier_archive_description.html']:
        copy(DATA/name,FINAL/'07_data'/name)
    acceptance=dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
        experiments=json.loads((RESULTS/'replication_completed.json').read_text()),
        year2024_tail=json.loads((RESULTS/'tail_2024_completed.json').read_text()),
        document_build=build,word_layout=word,pdf_geometry=qa,
        figure1_unchanged=True,editable_equations_retained=True,all_figures_caption_same_page=True,table1_single_page=True,
        new_figure_formats=['SVG','PDF','PNG 600 dpi'],all_figures_source_csv=True,
        grayscale_inspection='Figure 6 native PDF rendered to greyscale and inspected; symbols, open/filled years, neutral line styles and direct labels retain identification',
        visual_inspection='Figures 2-6, main manuscript page overviews and external-validation supplement pages inspected; observed caption/legend/reference pagination issues corrected',
        claims_boundary='2024 execution mechanism only; counterfactual Frontier-NYISO pairing; no independent R2 2024 claim',
        retained_non_replications='negative 2025 24 h persistence increments; Eagle-2025 7 h online deadline failures; zero-to-zero Frontier-2024 mixed case; resource-proxy sensitivity')
    save(REPORTS/'DELIVERY_ACCEPTANCE.json',acceptance)
    for r in json.loads((BASE/'frozen_2025_manifest.json').read_text()):assert sha(r['path'])==r['sha256']
    pagecounts={q['file'].replace('.pdf','.docx'):q['pages'] for q in qa}
    index=['# Final delivery index','',
        'This package completes the requested external-workload and second-market-year mechanism replication, manuscript update and figure system. The English and Chinese papers include all supported findings and the retained counterexamples.','',
        '## Manuscripts','']
    for folder in ['04_manuscript','05_supplementary']:
        for p in sorted((FINAL/folder).glob('*.docx')):index.append(f'- {folder}/{p.name} — {pagecounts[p.name]} pages.')
    index+=['','## Other folders','',
        '- 01_results: complete primary, sensitivity, recourse, floor, cost and boundary CSVs.',
        '- 02_figures: Figures 2–6 in vector SVG/PDF and 600 dpi PNG; source_data contains the five panel-indexed CSVs.',
        '- 03_scripts: all added parsing, forcing, execution, figure, document and packaging scripts.',
        '- 06_reports: the eight-question result report, data provenance, style guide, manuscript changes, run commands and acceptance/manifests.',
        '- 07_data: original official Frontier ZIPs containing complete raw SWF files, licences, headers, mapped arrivals and year forcing snapshots. Full expanded logs and detailed cohort trajectories additionally remain in the working project under external_validation.',
        '',
        'Figure 1 is kept inside each paper unchanged. It is not redrawn or separately re-exported in this package. The original full project remains a dependency for reproduction; neural checkpoints are not redistributed. Internal QA page images and PDFs are intentionally omitted.','']
    (FINAL/'FILE_INDEX.md').write_text('\n'.join(index),encoding='utf-8')
    files=[dict(path=p.relative_to(FINAL).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(FINAL.rglob('*')) if p.is_file() and p.name!='DELIVERY_MANIFEST.json']
    save(FINAL/'DELIVERY_MANIFEST.json',dict(status='complete',files=files))
    target=WORK/'EXTERNAL_VALIDATION_FINAL_20260922.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(FINAL.rglob('*')):
            if p.is_file():z.write(p,FINAL.name+'/'+p.relative_to(FINAL).as_posix())
    save(BASE/'progress.json',dict(status='complete',delivery=str(FINAL),archive=str(target),archive_bytes=target.stat().st_size))
    print(json.dumps(dict(directory=str(FINAL),archive=str(target),bytes=target.stat().st_size,files=len(files)+1,pages=pagecounts),indent=2,ensure_ascii=False))
if __name__=='__main__':main()
