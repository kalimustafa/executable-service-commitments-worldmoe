"""Package final documents, all figures, source rows and reproducibility code."""
from pathlib import Path
import json, shutil, zipfile, hashlib

RUN=Path(__file__).resolve().parents[1];WORK=RUN.parent;ROOT=WORK/'WorldMoE_Revision_20260928'
summary=json.loads((ROOT/'results_summary.json').read_text())
manuscript=json.loads((ROOT/'manuscript_manifest.json').read_text())
assert summary['status']=='complete' and manuscript['main_figures']==6
render=json.loads((ROOT/'qa/render_counts.json').read_text(encoding='utf-8-sig'))

plot_scripts=['redraw_worldmoe_figures.py','write_worldmoe_paper.py','render_worldmoe_documents.ps1','qa_worldmoe_pages.py','package_worldmoe_revision.py']
for name in plot_scripts:shutil.copy2(RUN/'scripts'/name,ROOT/'scripts'/name)
code=ROOT/'code/decision_module'
for source in (RUN/'src').rglob('*.py'):
    if '__pycache__' in source.parts:continue
    target=code/'src'/source.relative_to(RUN/'src');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
for name in ['train_deep_rollout.py','finish_cdf_primary.py','evaluate_scenario_recourse.py','prepare_rollout_states.py','cache_rollout_forecasts.py','run_cdf_extensions.py','summarize_worldmoe_final.py']:
    target=code/'scripts'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(RUN/'scripts'/name,target)
extension=WORK/'ServiceCommitment_1to5_20260927'
for name in ['scripts/run_bounds.py','src/context.py','src/controllers.py','paths.py']:
    target=ROOT/'code/service_constraints'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(extension/name,target)
for source in (extension/'vendor').glob('*.py'):
    target=ROOT/'code/service_constraints/vendor'/source.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)

models=ROOT/'models';models.mkdir(exist_ok=True)
results=ROOT/'results';results.mkdir(exist_ok=True)
for seed,epoch in zip(summary['seeds'],summary['selected_epochs']):
    name=f'CdfR2_seed{seed}_epoch{epoch:02d}.pt';shutil.copy2(RUN/'checkpoints'/name,models/name)
    selection='cdf_selection.json' if seed==20260920 else f'cdf_selection_s{seed}.json'
    selected=json.loads((RUN/'results'/selection).read_text())
    (results/selection).write_text(json.dumps(selected,indent=2))
    for suffix in ['','_postonly']:
        name=f'CdfR2_Eagle_2025_s{seed}_ep{epoch:02d}_coupled_cdf{suffix}.json'
        shutil.copy2(RUN/'results'/name,results/name)

# Export the scientific protocol rather than conversational authorization text.
auth=json.loads((RUN/'results/replication_authorization.json').read_text())
keep=['shared_initial_checkpoint','shared_initial_checkpoint_sha256','replication_scope','training_years','physical_training_years','selection_year','evaluation_year','epochs_max','batch','qp_regularization','early_stopping','controller']
provenance=json.loads((ROOT/'provenance.json').read_text())
provenance['shared_initialization']={k:auth[k] for k in keep}
provenance['model_name']='WorldMoE'
provenance['internal_run_family']='CdfR2'
provenance['evaluation_scope']=summary['evaluation_scope']
provenance['module_checkpoints']=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in models.glob('*.pt')]
(ROOT/'provenance.json').write_text(json.dumps(provenance,indent=2))
summary.pop('threshold_waived_by_user',None);summary['target_80_role']='descriptive engineering target; not reached and not a condition for reporting the retained three seeds'
(ROOT/'results_summary.json').write_text(json.dumps(summary,indent=2))

main_pages=next(x['Pages'] for x in render if 'Revised_Manuscript' in x['File'])
supp_pages=next(x['Pages'] for x in render if 'Supplementary' in x['File'])
readme=f'''# WorldMoE 论文与图表

本包包含英文正文（{main_pages} 页）、补充材料（{supp_pages} 页）、6 张全部重绘的主文图、1 张主文表和 13 张补充表。论文统一使用 WorldMoE 名称，将预测与状态成本联合决策模块作为框架内部组成。

包含创新点 1、2、3、5：认证节约上界、信息时序价值、电网约束下的履约、事件影响与稳健性。不包含新版与旧版 WorldMoE 的机制对照。

三种子平均节约 105,271.06 美元，相当于基准成本的 18.9620%，以及认证节约上界的 79.3867%。认证上界为 132,605.42 美元，原 80% 对应 106,084.34 美元；实际结果没有达到该数值。

- 种子 20260920：选择第 6 轮，2025 节约 105,565.67 美元。
- 种子 20260921：训练 6 轮后早停，选择第 2 轮，节约 105,003.52 美元。
- 种子 20260922：训练 6 轮后早停，选择第 0 轮，节约 105,244.00 美元。第 0 轮指共享的已训练初始化检查点。

三次均以 2024 年选择检查点。2021–2023 年拟合参数，物理标签来自 2022–2023 年；2025 年不参与拟合或逐种子检查点选择。三种子共享预训练初始化和冻结骨干缓存，反映优化过程的变异。此前开发已检查过 2025 年结果，文稿保留这一评价范围说明。

`manuscript/` 是两个最终 DOCX。`figures/` 包含每张图的 PNG、SVG 和矢量 PDF；SVG 文本可编辑。`source_data/` 包含逐种子、逐日、信息限制、电网约束、预测分数与事件分析数据。`source_data/certificates/` 包含 48 个站点级原始证书和诊断数据。

`models/` 是三个最终决策模块检查点。CdfR2 为代码内部标识，文稿名称为 WorldMoE。`code/` 保留模型、训练、回放及约束代码；`provenance.json` 保存选中检查点、共同初始化和完整哈希。

图表重绘使用带 NumPy、pandas、Matplotlib 的 Python：运行 `scripts/redraw_worldmoe_figures.py`。文稿编写使用带 python-docx、NumPy、pandas 的 Python：运行 `scripts/write_worldmoe_paper.py`。工作区中图表使用 Anaconda Python，DOCX 使用 Codex 随附 Python。程序只读取已完成的源数据，不会重新训练模型。

完整训练及全年回放仍需要原工作区的冻结 WorldMoE/R2 缓存、物理 forcing 和数据档案；本包不包含这些大型输入。原运行目录为 `R2_JointDecision_20260927`，按训练、冻结选择、测试、扩展、汇总的顺序执行。避免在唯一保留的结果目录中覆盖重跑。

主模型三种子测试均无实质迟延、期末工作为零。附加电网约束实验中的跨年工作单列并计价；截止期保护消除了实质迟延，并不把跨年义务删除。所有金融事件剔除仅改变汇总，不改变原始连续轨迹。
'''
(ROOT/'README.md').write_text(readme,encoding='utf-8')
files=[p for p in ROOT.rglob('*') if p.is_file() and 'qa' not in p.relative_to(ROOT).parts and '__pycache__' not in p.parts]
inventory=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size) for p in files]
(ROOT/'package_manifest.json').write_text(json.dumps(dict(files=inventory,main_pages=main_pages,supplement_pages=supp_pages),indent=2))
files.append(ROOT/'package_manifest.json')
archive=WORK/'WorldMoE_Revision_20260928.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for p in files:z.write(p,p.relative_to(WORK))
print(json.dumps(dict(archive=str(archive),archive_bytes=archive.stat().st_size,files=len(files),main_pages=main_pages,supplement_pages=supp_pages),indent=2))
