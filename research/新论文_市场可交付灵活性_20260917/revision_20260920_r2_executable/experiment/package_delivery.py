"""Package the completed, visually approved research outputs."""
from pathlib import Path
import hashlib,json,shutil,zipfile
ROOT=Path(__file__).resolve().parents[1];PAPER=ROOT.parent;REPO=ROOT.parents[1]/'power-dc-vpp'
def main():
    qa=json.loads((ROOT/'qa/final_visual_qa.json').read_text(encoding='utf-8'));assert qa['status']=='PASS'
    status=json.loads((ROOT/'results/analysis_completed.json').read_text());assert status['status']=='complete'
    out=PAPER/'交付稿_20260920_R2可执行柔性';out.mkdir(exist_ok=True)
    names=['NatureEnergy_ExecutableFlexibility_R2_EN','Supplementary_ExecutableFlexibility_R2_EN','NatureEnergy_ExecutableFlexibility_R2_CN']
    for name in names:shutil.copy2(ROOT/'manuscript'/(name+'.docx'),out/(name+'.docx'))
    for name in ['FINAL_R2_REPORT.md','MANUSCRIPT_R2_PATCH.md','W0_W1_SELECTION.md','R2_INTEGRATION_PLAN.md','W1_OOD_DIAGNOSTIC.md','FROZEN_FEATURE_CACHE_MANIFEST.json','REPRODUCE.md']:
        shutil.copy2(ROOT/name,out/name)
    for name in ['policy_comparison_2025.csv','service_window_R2_results.csv','heterogeneous_deadline_R2_results.csv','attribution_waterfall.csv','R2_forecast_metrics.csv','R2_calibration_metrics.csv','paired_policy_differences.csv']:
        shutil.copy2(ROOT/'results'/name,out/name)
    figure_archive=out/'Figures_and_Source_Data.zip'
    with zipfile.ZipFile(figure_archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in sorted((ROOT/'figures').rglob('*')):
            if p.is_file():archive.write(p,p.relative_to(ROOT))
        archive.write(ROOT/'experiment/build_figures.py','build_figures.py')
    reproducibility=out/'R2_Research_Companion.zip'
    with zipfile.ZipFile(reproducibility,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for folder in ['experiment','results','checkpoints','forecast_2025','validation_economics']:
            for p in sorted((ROOT/folder).glob('*')):
                if p.is_file() and p.suffix in ['.py','.csv','.json','.pt','.parquet','.md']:
                    archive.write(p,p.relative_to(ROOT))
        for name in ['selected_w0_w1.json','frozen_parameter_report.json','training_config.json','EVALUATION_FREEZE.json','FROZEN_FEATURE_CACHE_MANIFEST.json','REPRODUCE.md','w1_ood_metrics.csv','W1_OOD_DIAGNOSTIC.md','R2_INTEGRATION_PLAN.md']:
            archive.write(ROOT/name,name)
        for p in sorted((ROOT/'cache').glob('*.json')):archive.write(p,p.relative_to(ROOT))
        for p in sorted((ROOT/'manuscript').glob('*.md')):
            if not p.name.endswith('_typeset.md'):archive.write(p,p.relative_to(ROOT))
        for name in ['build_documents.py','render_word.ps1','render_pages.py']:
            archive.write(ROOT/'manuscript'/name,'manuscript/'+name)
        archive.write(REPO/'src/dcvpp/vpp_world_moe/r2_market.py','repository_module/r2_market.py')
        for name in ['run_screen.py','deepen_scheduling.py','repair_reservations.py']:
            archive.write(PAPER/'revision_20260920_market_expansion/experiment'/name,'inherited_execution/'+name)
        for name in ['multiseed_20260910_20240829_w1.yaml','multiseed_20260910_20240829_r_full.yaml']:
            archive.write(REPO/'configs'/name,'registered_configs/'+name)
    manifest=dict(status='complete',directory=str(out),manuscript=str(out/(names[0]+'.docx')),supplement=str(out/(names[1]+'.docx')),chinese=str(out/(names[2]+'.docx')),
        study=str(ROOT),world_seed=20240829,w0_epoch=80,w1_epoch=80,R_epoch=78,R2_primary='full_native',
        pages=qa['pages'],scientific_counts=json.loads((ROOT/'manuscript/manuscript_structure.json').read_text()),
        files=[dict(name=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(out.iterdir()) if p.is_file()])
    (out/'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    pointer=PAPER/'CURRENT_MANUSCRIPT.json'
    prior=json.loads(pointer.read_text(encoding='utf-8')) if pointer.exists() else {}
    if pointer.exists():shutil.copy2(pointer,ROOT/'previous_manuscript_pointer.json')
    current={**manifest,'revision':str(ROOT),'delivery':str(out),'source_experiment':str(ROOT/'results'),
        'scope':'frozen_world_R2_market_forecasting_continuous_executable_flexibility',
        'training_code_modified':True,'original_W0_W1_training_modified':False,'checkpoints_modified':False,
        'new_R2_checkpoints_added':True,'primary_language':'EN','main_file':manifest['manuscript'],
        'supplement_file':manifest['supplement'],'chinese_file':manifest['chinese'],
        'previous_main_file':prior.get('main_file')}
    pointer.write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'delivery':str(out),'files':len(manifest['files']),'pages':qa['pages']},ensure_ascii=False))
if __name__=='__main__':main()
