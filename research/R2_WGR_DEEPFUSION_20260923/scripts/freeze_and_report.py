from src.runtime import ROOT
from src.data.access_guard import install_guard
install_guard()
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
import yaml

MODELS=['A0_Base','A1_Shallow','A2_Deep_NoGate','A3_WGR']
LABELS={'A0_Base':'Base','A1_Shallow':'Shallow','A2_Deep_NoGate':'Deep-NoGate','A3_WGR':'WGR','RASC':'RASC'}
SEEDS=[20260920,20260921,20260922]

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def md_table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(str(x) for x in row)+' |' for row in rows])

def main():
    selection=pd.read_csv(ROOT/'results/checkpoint_selection_2024.csv')
    forecast=pd.read_csv(ROOT/'results/forecast_metrics_2024.csv')
    economics=pd.read_csv(ROOT/'results/economic_dev_metrics_2024.csv')
    chosen=pd.read_csv(ROOT/'results/economic_policy_selection_2024.csv')
    gate=pd.read_csv(ROOT/'results/gate_summary_2024.csv')
    ablation=pd.read_csv(ROOT/'results/world_ablation_2024.csv')
    complexity=pd.read_csv(ROOT/'results/model_complexity.csv')
    tests=json.loads((ROOT/'results/required_tests.json').read_text())
    assert tests['passed'] and tests['tests_run']==10
    assert len(selection)==12 and selection.split_hash.nunique()==1
    assert len(economics)==62 and economics.evaluation_protocol.eq('fp32_v1').all()
    assert set(ablation.ablation)=={'normal','force_gate_0','force_gate_1','no_expert_tokens','late_scale_only'}
    assert forecast.quantile_crossing_count.max()==0
    stability=pd.read_csv(ROOT/'results/batch_stability_fp32.csv')
    assert stability.max_rt_quantile_delta.max()<.001
    for name in MODELS:
        for seed in SEEDS:
            done=json.loads((ROOT/'logs'/f'{name}_seed{seed}_calibration_complete.json').read_text())
            assert done['evaluation_protocol']=='fp32_v1'
    f=forecast[forecast.split.eq('development_h2')&forecast.calibration.eq('native')&forecast.publication.eq('pooled')]
    assert len(f)==12
    means=f.groupby('model').mean(numeric_only=True)
    w=means.loc['A3_WGR'];b=means.loc['A0_Base'];s=means.loc['A1_Shallow'];d=means.loc['A2_Deep_NoGate']
    # The present finding warrants NO-GO directly: higher mean loss than Base
    # and lower tail coverage than both simpler controls. No blended score.
    assert w.pinball>b.pinball and w.tail_coverage<min(b.tail_coverage,s.tail_coverage), 'Reassess verdict from the changed evidence'
    primary_gate=gate[gate.seed.eq(SEEDS[0])&gate.split.eq('exploratory_test')].set_index('group')
    econ=economics[economics.window_h.isin([12,24])]
    emean=econ.groupby(['model','window_h','policy']).mean(numeric_only=True)
    delta=pd.read_csv(ROOT/'results/matched_seed_economic_deltas.csv')
    twelve=delta[delta.comparison.eq('A3_WGR minus A0_Base')&delta.window_h.eq(12)&delta.policy.eq('interval')]
    twentyfour=delta[delta.comparison.eq('A3_WGR minus A0_Base')&delta.window_h.eq(24)&delta.policy.eq('interval')]
    pdelta=pd.read_csv(ROOT/'results/matched_seed_forecast_deltas.csv')
    relative_base=100*(w.pinball/b.pinball-1);relative_shallow=100*(w.pinball/s.pinball-1)
    qs=[
        ('Q1. 新的 deep fusion 是否优于当前 shallow fusion？',f'没有稳定优于。统一校准后的 2024 H2 开发集，WGR 的 RT pinball 三 seed 均值为 {w.pinball:.6f}，Shallow 为 {s.pinball:.6f}（WGR 高 {relative_shallow:.2f}%）；tail coverage 为 {100*w.tail_coverage:.2f}% 对 {100*s.tail_coverage:.2f}%。Deep-NoGate 的 pinball 为 {d.pinball:.6f}、tail coverage 为 {100*d.tail_coverage:.2f}%，也没有形成稳定的多维优势。'),
        ('Q2. 是否优于 no-world Base？',f'不支持。Base 的 RT pinball 均值 {b.pinball:.6f}，WGR 为 {w.pinball:.6f}，高 {relative_base:.2f}%；tail coverage 分别为 {100*b.tail_coverage:.2f}% 和 {100*w.tail_coverage:.2f}%。'),
        ('Q3. World information 主要在哪些 market regime 中被 gate 调用？',f"主 seed 的 normal / moderate / tail 平均 gate 为 {primary_gate.loc['normal_regime','mean_gate']:.4f} / {primary_gate.loc['moderate_regime','mean_gate']:.4f} / {primary_gate.loc['tail_regime','mean_gate']:.4f}。低/高历史波动为 {primary_gate.loc['low_volatility','mean_gate']:.4f} / {primary_gate.loc['high_volatility','mean_gate']:.4f}，pre/post-DA 为 {primary_gate.loc['pre_DA','mean_gate']:.4f} / {primary_gate.loc['post_DA','mean_gate']:.4f}。这说明 gate 与状态有关，但不能据此认定 world 信息提升了预测或收益。"),
        ('Q4. 更深的 multi-scale fusion 是否真的提高 tail information？',f'当前结果不支持稳定提高。WGR 平均 tail coverage 比 Base 低 {100*(b.tail_coverage-w.tail_coverage):.2f} 个百分点，比 Shallow 低 {100*(s.tail_coverage-w.tail_coverage):.2f} 个百分点。主 seed 的 token/gate 干预结果列于后表，只用于诊断，未据此新增架构或挑 seed。'),
        ('Q5. Overall pinball 是否恶化？',f'是，按预先明确的 RT 七分位 pinball 指标，WGR 的三 seed 均值比 Base 高 {relative_base:.2f}%。相对 Base，{int((pdelta[pdelta.comparison.eq("A3_WGR minus A0_Base")].pinball_delta>0).sum())}/3 个 seed 的 pinball 更高；方向不满足稳定提升。DA/DART 和三头等权均值另列在完整 CSV 中，不与 RT pinball 混用。'),
        ('Q6. 2024 development economics 是否改善？',f'只出现局部改善。12h interval 相对 Base 的三 seed 平均增量为 {twelve.value_delta_usd.mean():+.2f} 美元，{int((twelve.value_delta_usd>0).sum())}/3 个 seed 为正；24h interval 为 {twentyfour.value_delta_usd.mean():+.2f} 美元，{int((twentyfour.value_delta_usd>0).sum())}/3 为正。所有 point/interval 和风险选择分数都保留，不能用某一窗口的均值声称整体占优。'),
        ('Q7. 三 seed 是否稳定？',f'没有表现出稳健优势。WGR RT pinball 的 seed 标准差为 {f[f.model.eq("A3_WGR")].pinball.std(ddof=1):.6f}，Base 为 {f[f.model.eq("A0_Base")].pinball.std(ddof=1):.6f}；tail coverage 和经济增量也随 seed 改变。20260920 仍是预先固定的主 seed，20260921/22 全部保留。'),
        ('Q8. 是否值得未来替换论文中的 current R2？','当前不值得替换，也不建议将此冻结候选作为优先进入新 holdout 的版本。复杂度增加没有换来稳定的预测、尾部和决策联合改进。原论文、旧 R2、W0/W1、reservation 和 recourse 均未修改。')]
    forecast_table=md_table(['模型','RT pinball 均值 ± seed SD','DA pinball','DART pinball','50% coverage','90% coverage','Tail coverage','90% width'],[
        [LABELS[name],f"{means.loc[name,'pinball']:.6f} ± {f[f.model.eq(name)].pinball.std(ddof=1):.6f}",f"{means.loc[name,'da_pinball']:.6f}",f"{means.loc[name,'dart_pinball']:.6f}",f"{100*means.loc[name,'coverage50']:.2f}%",f"{100*means.loc[name,'coverage90']:.2f}%",f"{100*means.loc[name,'tail_coverage']:.2f}%",f"{means.loc[name,'interval_width']:.3f}"] for name in MODELS])
    seed_table=md_table(['模型','Seed','Best epoch','2024 selection proper score','最终 RT pinball','Tail coverage'],[
        [LABELS[row.model],int(row.seed),int(row.best_epoch),f'{row.best_price_score:.9f}',
         f"{f[(f.model==row.model)&(f.seed==row.seed)].iloc[0].pinball:.6f}",
         f"{100*f[(f.model==row.model)&(f.seed==row.seed)].iloc[0].tail_coverage:.2f}%"] for row in selection.itertuples()])
    econ_table=md_table(['模型','12h point USD','12h interval USD','24h point USD','24h interval USD'],[
        [LABELS[name]]+[f"{emean.loc[(name,window,policy),'value_usd']:.2f}" for window,policy in [(12,'point'),(12,'interval'),(24,'point'),(24,'interval')]] for name in ['RASC']+MODELS])
    abo=ablation[(ablation.split=='development_h2')&(ablation.window_h==12)].set_index('ablation')
    abo_rows=[]
    for variant in ['normal','force_gate_0','force_gate_1','no_expert_tokens','late_scale_only']:
        af=abo.loc[variant]
        a24=ablation[(ablation.split=='development_h2')&(ablation.window_h==24)&ablation.ablation.eq(variant)].iloc[0]
        abo_rows.append([variant,f'{af.pinball:.6f}',f'{100*af.tail_coverage:.2f}%',f'{af.fixed_normal_policy_selection_metric:.2f}',f'{a24.fixed_normal_policy_selection_metric:.2f}'])
    ablation_table=md_table(['主 seed 干预','RT pinball','Tail coverage','12h 固定正常策略选择分数','24h 固定正常策略选择分数'],abo_rows)
    complexity_table=md_table(['模型','R2 总参数','可训练参数上限','冻结 W1 参数','FP32 ms / 512 rows'],[
        [LABELS[r.model],f'{r.head_total_parameters:,}',f'{r.trainable_parameters:,}',f'{r.frozen_world_parameters:,}',f'{r.inference_ms_per_batch:.3f}'] for r in complexity.itertuples()])
    lines=['# R2-WGR 最终实验报告','', '**最终判断：NO-GO。当前候选不替换论文 R2，不建议优先进入未来 holdout evaluation。**','']
    for question,answer in qs:lines += ['## '+question,'',answer,'']
    lines += ['## 结果口径与时间边界','',
        '下表主预测比较为 2024-07-01 至 2024-12-31 的本地目标日期，统一 native affine calibration，pre/post-DA 等权合并。pinball 列明确指 RT 七分位损失，单位 USD/MWh；另有 DA、DART 和三头均值。tail event 定义为 |DART| 超过该 zone 的 2021–2023 训练分布 90% 分位，tail coverage 是这些行上 RT 中央 90% 区间的覆盖率。误差棒/标准差只描述三个已运行 seed，不是总体置信区间。',
        '全部 2024 结果都是 development / model-selection evidence。W1 本身历史上已按全 2024 验证选择。原字段 exploratory_test 在本报告统一称 development H2，不作 independent test 宣称。',
        '训练价格有效行是 2021-02-01 至 2023-12-31，共 102,144 行；最初一月因正式 lag 构造而不在可用训练面板内。Selection 为 1 月 3 日至 3 月 30 日，8,444 行；calibration 为 4 月 1 日至 6 月 29 日，8,640 行；embargo 为 1 月 1–2 日、3 月 31 日、6 月 30 日，共 384 行；H2 开发为 17,668 行。行键与 split 文件哈希见 configs/splits_2024_frozen.json。',
        '经济主块严格为 2024-08-21 00:00 至 2024-12-31 23:45 UTC（133 个 UTC 日，134 个跨时区本地日期），由既有连续物理数据可用性确定。12h/24h 是主窗口，7h 仅服务边界诊断。','',
        '## 2024 预测对照','',forecast_table,'','## 各 seed checkpoint','',seed_table,'',
        '## 2024 development-year decision value','',econ_table,'',
        f"以上是全部三个 seed 的算术均值；RASC 为一个已有固定开发对照。全部 {len(economics)} 个正常经济运行的最大 late IT 为 {economics.late_it_mwh.max():.9g} MWh。逐 seed 的 CVaR95、shifted IT、closing work 和收益见 economic_dev_metrics_2024.csv。选择使用原式 value − 0.1 × 本地日期数 × service-adjusted daily CVaR95，并先要求 zero-late；没有混合预测和经济指标构造新总分。",
        '所有模型只使用相同的 native affine calibration；没有为 WGR 额外搜索校准参数化或 gate threshold。逐 seed、逐窗口的 point/interval 选择保存在 economic_policy_selection_2024.csv，冻结全部三个 seed。','',
        '## Gate 与 world contribution','',ablation_table,'',
        '所有干预只对主 seed 的同一冻结 checkpoint 和同一冻结 calibrator 做推理，不重新校准或重训。表中经济分数固定使用 normal 模型已选策略；CSV 同时列出每个干预按原规则重选策略的分数，防止策略切换掩盖变化。late-scale-only 只移除 early/middle，保留 final site/global/tail 与 expert tokens；no-expert 同时移除 token bank 和显式 expert 选择分支。',
        'force gate=0 返回该 WGR 内部市场路径。因为 world 不可用时须精确退化为 Base，它同时跳过融合 LayerNorm；因此该干预不是只删除 world 内容的无混杂因果实验。它也不等同于独立训练的 A0。对 world 的价值判断以四架构对照和所有干预的共同证据为准。',
        '专家图仅显示编号 expert 的注意力剖面，没有将编号强行解释为天气、光伏或工作负载专家，也不把相关性表述为因果机制。逐行 gate、latent groups、entropy、selected experts 已完整保存。','',
        '## W1 基模选择与实现','',
        '按用户追加要求，对最初发布的 R5-V2.2 W1（epoch 56）和后三 seed 一并比较。正式最差 horizon proper score 分别为 0.616112634、0.570742692、0.568950799、0.622669942。选择后来 seed 20240829、epoch 80；它在 h1/h4/h16/h96 均优于初版，主分数低约 7.65%。四者数据/模型配置与记录有效样本数一致，W1 native gate 均 PASS。详细证据见 W1_FOUNDATION_SELECTION.md 和 selected_w1.json。',
        '保留其完整 W1 与原 W0 parent 血缘，不拼接别的 W0、不解冻 W1。后续 W0 parent 的原生 gate FAIL 状态如实保留；h96 latent free-bit floor fraction=1.0 的继承限制也如实保留。严格加载的 W1 为 59,156,961 参数。',
        '三个真实 interaction state 对应 block indices 2/5/9（自然编号 3/6/10），每层时间归约后保留 40 个 slot token；另保留 site/global/tail/expert/router 信息。训练期 world cache 1,086 天，FP16 存储约 72.7 MB，来自 BF16 frozen forward。',
        'A0/A1 使用原正式架构副本；A2/A3 从同 seed A0 初始化市场编码器和 decoder。8 epochs warmup 冻结市场 encoder，之后 base LR=3e-4、fusion LR=2e-3。只有一个预声明 WGR 设置：gate penalty 1e-4、residual penalty 1e-5、world dropout 0.10。最大 100 epoch、joint-stage patience 12；warmup 的较好 checkpoint 也可以被正式 score 选中。没有追加超参或架构。','',complexity_table,'',
        '参数表的 trainable 是 Stage B 可训练上限；部分最佳 checkpoint 出现在 warmup，其当时可训练计数另存于 model_complexity.csv。A0 保留原 no-world 代码中没有实际梯度贡献的 world 模块参数，避免为了参数表漂亮改写基线。计时仅包括缓存后的 R2 推理，不包括 W1 或磁盘 I/O。','',
        '## 数值适配与必需测试','',
        '原注册 .venv 启动器缺失，其 PyTorch 2.13 包在完整 W1 构造中反复产生原生访问异常。隔离分支复用现有 Anaconda Python 3.13.5 / PyTorch 2.9.0+cu126，未改旧环境。原 Base/Shallow selection score 复现误差分别为 0 和 5.96e-8；原 2024 RASC 的 12h point/interval 收益复现误差低于 1e-12 USD。',
        'W1 提取和训练/selection 保留 BF16。部署 batch 检查发现 BF16 的 R2 分位数随 batch shape 变化，因此所有最终校准、预测导出、经济与干预统一使用 FP32，保持相同校准自由度、边界、objective 和 250 iterations；未重训或重新挑 checkpoint。日志明确保留被替代的 BF16 汇总，最终聚合只读取 fp32_v1 经济记录。',
        f"十项必需测试全部通过；最终 FP32 在 batch 512/64 的最大 RT 分位数差为 {stability.max_rt_quantile_delta.max():.9g} USD/MWh。全部报告预测 quantile crossing=0，post-DA anchor 和 RT=DA−DART 均满足原实现恒等式。年份入口、Parquet 日期预读、NPZ 日期与源目录只读保护已启用。",'',
        '## 冻结范围与后续边界','',
        'frozen_candidate/ 保存三个 WGR checkpoint、固定配置、划分、校准、架构、选择记录、README 和 SHA256 清单。20260920 是预先固定主 seed。包已冻结，但 NO-GO 表示当前证据不支持优先进入新 holdout，更不支持替换论文模型。',
        '**2025 not accessed**：本分支未读取 2025 目标价格、R2 预测、经济结果或 world cache；也未使用 2026 目标。唯一目录名含 forecast_2025 的数据引用是文件名明确为 development_forecast_only_2024h2.parquet 的既有 RASC 开发对照，读取目标日期列验证为 2024 后才读取预测。早期 W1 目录名含 2025final 的历史权重/2024 元数据也单独登记允许，均不是 2025 结果。',
        '原项目只读，没有修改当前 R2/W0/W1/RASC/reservation/recourse、2025 预测或经济结果、论文 CSV/Word/Supplementary。原始研究结果已影响架构动机，所以未来即使评价 2025，也不能称其为 previously unseen blind test for R2-WGR architecture。最严格的新证据需要另一个预先封存、未参与架构设计的窗口（例如未来 2026 holdout）；本任务未打开它。','',
        '**结论：deep fusion does not add robust value in this preregistered development experiment.**','', '**NO-GO**']
    report='\n'.join(lines)+'\n'
    (ROOT/'reports/R2_WGR_FINAL_REPORT.md').write_text(report,encoding='utf-8')
    frozen=ROOT/'frozen_candidate';calibrations={};records=[]
    for seed in SEEDS:
        source=ROOT/'checkpoints'/f'A3_WGR_seed{seed}_best.ckpt'
        target=frozen/f'r2_wgr_seed{seed}_best.ckpt';shutil.copy2(source,target)
        calibrations[str(seed)]=json.loads((ROOT/'checkpoints'/f'A3_WGR_seed{seed}_calibration.json').read_text())
        records.append(selection[(selection.model=='A3_WGR')&(selection.seed==seed)].iloc[0].to_dict())
    shutil.copy2(ROOT/'configs/splits_2024_frozen.json',frozen/'splits_2024_frozen.json')
    cfg=yaml.safe_load((ROOT/'configs/experiment.yaml').read_text());cfg.update(status='frozen',recommendation='NO-GO',w1_source='../configs/selected_w1.json',
        source_code_sha256={str(p.relative_to(ROOT)):digest(p) for folder in ['src','vendor'] for p in sorted((ROOT/folder).rglob('*.py'))})
    (frozen/'config_frozen.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
    (frozen/'calibration_frozen.json').write_text(json.dumps(dict(kind='native_affine',parameters_per_model=24,precision='FP32',seeds=calibrations),indent=2))
    architecture='''# Frozen R2-WGR architecture

Base market latent (exact A0 market path, width 128) + vector-gated bounded
world residual, followed by LayerNorm only for active world fusion. Gate zero
and unavailable world exactly use the internal Base path. Frozen W1 blocks
3/6/10 supply 40 slot tokens each; final site/global/tail and seven routed
expert representations remain distinct. Independent Linear/GELU/LayerNorm
adapters, scale embeddings, market query (zone/lead/publication/anchor), two
4-head cross-attention layers, query-conditioned expert selection with log
router prior, bounded residual .5*tanh(LayerNorm(MLP)), and 128-vector gate.

Exact formal ordered DA/DART quantiles, Gaussian copula, regime logits and
RT identity are reused. Published DA is fixed. Seven quantiles: .05, .10,
.25, .50, .75, .90, .95. No raw-quantile sorting repair. W1 is frozen and
excluded from optimizers. Stage B trainable head capacity: 922,418.

Training/selection BF16; final calibration/deployment FP32 for batch stability.
No W1 retraining, no seed dropping, no additional candidates.
'''
    (frozen/'architecture_frozen.md').write_text(architecture,encoding='utf-8')
    selection_text='# Checkpoint and policy selection record\n\n'+seed_table+'\n\nAll checkpoint decisions use the original 2024 proper price score, with the same frozen rows. Calibration is identical in freedom and bounds. Economic choice uses the original zero-late-first selector; all three seeds retained and primary fixed at 20260920.\n\n'+econ_table+'\n\nRecommendation: NO-GO for prioritizing a new holdout; no replacement of current R2.\n'
    (frozen/'selection_record.md').write_text(selection_text,encoding='utf-8')
    ready=['# READY_FOR_LOCKED_EVAL','', 'Package status: fully frozen. Recommendation: **NO-GO**; this filename records freeze readiness, not endorsement or authorization to open any holdout.','',
        '1. Architecture: R2-WGR as specified in architecture_frozen.md, frozen W1 seed 20240829 epoch 80.',
        '2. Checkpoints: each seed selected independently by the original 2024 proper price score; no training-loss or economic seed selection.',
        '3. Years: price fitting 2021–2023; W1 representations 2022–2024; selection/calibration/economics 2024.',
        '4. **2025 not accessed**. No 2025 target/prediction/economic/world-cache data; no 2026 targets. Historical W1 metadata and explicitly date-checked 2024 RASC comparator paths are documented in SOURCE_PATHS.yaml.',
        '5. Best epochs: '+', '.join(f"{int(r['seed'])}: {int(r['best_epoch'])}" for r in records)+'.','',
        '6. 2024 forecasting (development H2; FP32 native affine):','',forecast_table,'',
        '7. 2024 development economics (USD; three-seed means):','',econ_table,'',
        '8. Versus Base/Shallow: overall WGR pinball and average tail coverage do not improve robustly. The 12h interval mean has a local improvement, while 24h and point policy directions are not robust.',
        f"9. Gate: primary normal/tail means {primary_gate.loc['normal_regime','mean_gate']:.4f}/{primary_gate.loc['tail_regime','mean_gate']:.4f}; nonconstant context use does not establish incremental value.",
        '10. No priority recommendation for future holdout evaluation of this candidate; current paper R2 remains unchanged.',
        '', '**Methodological boundary:** architecture motivation was informed by previously observed study outcomes. A future 2025 evaluation must not be called a previously unseen blind test for the R2-WGR architecture. A genuinely new prelocked window, such as future 2026 holdout, would be needed for strict new evidence. This package does not authorize accessing it.']
    (frozen/'READY_FOR_LOCKED_EVAL.md').write_text('\n'.join(ready)+'\n',encoding='utf-8')
    checksums={p.name:digest(p) for p in sorted(frozen.iterdir()) if p.is_file() and p.name!='SHA256SUMS.json'}
    (frozen/'SHA256SUMS.json').write_text(json.dumps(checksums,indent=2))
    status=dict(status='complete',verdict='NO-GO',architectures=4,seeds=SEEDS,checkpoints=12,required_tests=10,
                world_days=1086,forecast_precision='FP32',economic_normal_runs=62,world_ablation_variants=5,
                selected_w1='multiseed_20240829_epoch80',wgr_best_epochs={str(int(r['seed'])):int(r['best_epoch']) for r in records},
                forecast_h2_means={name:{k:float(means.loc[name,k]) for k in ['pinball','tail_coverage','mean_gate']} for name in MODELS},
                original_project_writes=0,locked_targets_accessed=False,report='reports/R2_WGR_FINAL_REPORT.md')
    (ROOT/'results/EXPERIMENT_COMPLETED.json').write_text(json.dumps(status,indent=2))
    (ROOT/'CURRENT_STAGE_STATUS.md').write_text('# Completed\n\nAll requested experimental stages complete. NO-GO. No further training, tuning, original-project writes, or locked data access is authorized by this run.\n',encoding='utf-8')
    print(json.dumps(status,indent=2),flush=True)

if __name__=='__main__':main()
