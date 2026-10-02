from src.runtime import ROOT,FAILED
from src.data.access_guard import install_guard
install_guard()
import json
import hashlib
import pandas as pd
import yaml

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(str(v) for v in row)+' |' for row in rows])

def main():
    checkpoints=pd.read_csv(ROOT/'results/checkpoint_selection_2024.csv')
    assert set(checkpoints.seed)=={20260920} and len(checkpoints)==2
    forecasts=pd.read_csv(ROOT/'results/forecast_metrics_2024.csv')
    old=pd.read_csv(FAILED/'results/forecast_metrics_2024.csv')
    old=old[old.model.isin(['A0_Base','A1_Shallow'])&old.seed.eq(20260920)]
    forecasts=pd.concat([old,forecasts],ignore_index=True)
    metric=forecasts[forecasts.split.eq('development_h2')&forecasts.calibration.eq('native')&forecasts.publication.eq('pooled')].set_index('model')
    costs=pd.read_csv(ROOT/'results/pilot_economic_metrics_2024.csv')
    success=pd.read_csv(ROOT/'results/pilot_cost_success_check.csv')
    gates=pd.read_csv(ROOT/'results/pilot_gate_summary_2024.csv')
    repaired=json.loads((ROOT/'results/redesign_tests.json').read_text());assert repaired['passed']
    statuses=[]
    for name in ['C0_Repaired','C1_ProfileResidual']:
        row=success[success.model.eq(name)&success.window_h.eq(24)].iloc[0]
        ni=100*(metric.loc[name,'pinball']/metric.loc['A0_Base','pinball']-1)
        passed=row.cost_reduction_pct>=1 and row.late_it_mwh<1e-8 and row.selection_score_delta>=0 and ni<=1
        statuses.append(dict(model=name,seed=20260920,cost_advantage_passed=bool(row.cost_reduction_pct>=1),
            total_success_passed=bool(passed),cost_reduction_pct=float(row.cost_reduction_pct),cost_reduction_usd=float(row.cost_reduction_usd),
            forecast_pinball_change_pct=float(ni),other_seeds_started=False))
    names=['A0_Base','A1_Shallow','C0_Repaired','C1_ProfileResidual']
    ftable=table(['模型','RT pinball ↓','Tail coverage','90% coverage','Mean gate'],[
        [name,f"{metric.loc[name,'pinball']:.6f}",f"{100*metric.loc[name,'tail_coverage']:.2f}%",f"{100*metric.loc[name,'coverage90']:.2f}%",f"{metric.loc[name,'mean_gate']:.4f}"] for name in names])
    ctable=table(['模型','12h point 净成本','12h interval 净成本','24h point 净成本','24h interval 净成本'],[
        [name]+[f"{costs[(costs.model==name)&(costs.window_h==h)&(costs.policy==p)].iloc[0].net_operating_cost_usd:,.2f}" for h,p in [(12,'point'),(12,'interval'),(24,'point'),(24,'interval')]] for name in ['RASC']+names])
    choices=table(['模型','window','原规则选中策略','最强现有对照','净成本差（节省为正）','降幅'],[
        [r.model,int(r.window_h),r.selected_policy,r.comparator_model+' / '+r.comparator_policy,f'{r.cost_reduction_usd:+.2f}',f'{r.cost_reduction_pct:+.4f}%'] for r in success.itertuples()])
    stages=[]
    for row in checkpoints.itertuples():
        trans=json.loads((ROOT/'logs'/f'{row.model}_stage_transition.json').read_text())
        stages.append([row.model,int(row.best_epoch),f'{row.initial_price_score:.9f}',f'{row.best_price_score:.9f}',int(trans['restored_epoch']),int(trans['restored_fusion_states']),int(row.epochs_run)])
    stage_table=table(['模型','best epoch','epoch 0 FP32 score','best FP32 score','Stage B 恢复 epoch','恢复 Adam 状态数','完成 epoch'],stages)
    verdict='PILOT COST THRESHOLD MET' if statuses[1]['total_success_passed'] else 'PILOT COST THRESHOLD NOT MET'
    text=['# R2-BPSR 单 seed 修复与成本试验','',f'**{verdict}**','',
        '本轮仅训练固定主 seed 20260920。20260921/20260922 未启动。上一轮被用户定义为失败；其成绩不能用来淘汰 WGR 思路。本轮结论也只适用于这两个预声明修复方案与当前开发窗口。',
        '', '## 成本目标是否达到','']
    for s in statuses:text.append(f"- {s['model']}：24h 净成本相对最强现有可行对照变化为节省 {s['cost_reduction_usd']:+.2f} USD（{s['cost_reduction_pct']:+.4f}%）；预声明的大幅优势门槛为至少 1%。RT pinball 相对 Base 变化 {s['forecast_pinball_change_pct']:+.3f}%。综合门槛 {'达到' if s['total_success_passed'] else '未达到'}。")
    text += ['', '## 真正修复了什么','',
        '1. C0 移除新加的整段 latent LayerNorm，采用 z_base + g*delta；最后残差层零初始化。C1 在 Base 原生 median/increment logits 上施加门控、零初始化投影后的有界残差，再通过原 monotone decoder，未直接修改最终分位数。两者任意 gate 的初始化输出均与 Base 完全一致。',
        '2. 连续性测试在残差已被扰动为非零后执行。gate 每缩小十倍，输出差近似缩小十倍，最终 gate=0 精确返回 Base；不再依赖不连续旁路。',
        '3. Warm-up 冻结 Base encoder、DA/DART/copula/regime heads 全部参数，并关闭 Base dropout。Stage B 恢复最佳 warm-up 权重及对应 AdamW moments，再加入 encoder LR=2e-5、heads LR=5e-5 的参数组；融合 moments 不重置。',
        '4. 全部 epoch（包括初始化 epoch 0）的 checkpoint selection、校准及最终评价统一使用 FP32；仅训练使用 BF16。不存在 BF16 选点与 FP32 最终评价之间未核对的排序问题。',
        '5. C1 使用同一发行时点已知的完整日内数值特征曲线，以及 post-DA 才开放的已公布 DA 曲线，以表达时段相对价格结构。pre-DA 全曲线及其均值先屏蔽，扰动未来 DA 不改变 pre-DA 预测的测试通过。',
        '', '## 训练与 checkpoint','',stage_table,'',
        '只运行两个预声明候选，没有按结果新增超参。原价格 loss、regime CE、轻量 gate/residual 正则、2024 划分、校准自由度和 executor 均保持原约束。8 项定向测试通过，包含初始化等价、非零残差的小 gate 连续性、完整 Base 冻结、残差可学习性、最优状态/优化器恢复、DA 因果性、价格恒等式和 FP32 batch 稳定性。',
        '', '## 2024 H2 预测开发指标','',ftable,'',
        'Pinball 指 RT 七分位损失，pre/post-DA 等权；tail event 是训练期各 zone |DART| 的 90% 阈值，tail coverage 为其 RT 中央 90% 区间覆盖率。全部是 development evidence，不是独立外部测试。',
        '', '## 同一执行器下的开发块运行成本','',ctable,'',choices,'',
        '单位均为 USD。参考执行成本为 116,723.262346 USD；净运行成本 = 参考成本 − 原执行器 value_usd，因此原手续费、终端工作计价均包含在内。物理容量、任务、截止时间、zero-late 条件、ranking provider 和 reservation 算法未修改。模型策略仍按既有 feasibility-first + risk-adjusted economic score 选择，并且两种策略都完整报告。',
        '这是相同反事实物理执行器与历史价格的开发期决策回放，不是已实现的真实设施降本。没有用一个较小的 savings 分母放大“成本优势”。',
        '', '## 成本可改进空间的诊断','',
        '独立记录的事后真实价格诊断，在相同贪心执行器下 12h/24h interval value 约为 4,169.70 / 5,092.42 USD。它不是可部署预测，也不是数学最优上界，更未作为训练输入。它表明此前多省几十美元不能称为大幅运行成本优势，也不能保证任何架构都能达到目标。',
        '', '## 下一阶段边界','',
        '本轮停在主 seed 评价点，不启动另外两个 seed。若预声明门槛未达到，就如实保留“成本目标尚未实现”；不降低门槛、不把修复正确等同于降本成功、不用本次结果否定整个 WGR 思路。若后续继续追求成本优势，需要根据这次诊断明确新的受控假设，而不是直接多训 seed 或扩大无边界搜索。',
        'W1 仍为 frozen seed20240829 epoch80，旧项目与失败分支只读。2025/2026 目标、预测、经济结果和 world cache 未访问。2024 已多次参与设计和开发，后续仍需要新的独立封存窗口才能评价泛化。']
    (ROOT/'reports/PRIMARY_SEED_RESULT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    record=dict(status='primary_seed_pilot_complete',seed=20260920,other_seeds_started=False,verdict=verdict,models=statuses,
                cost_goal_achieved=bool(statuses[1]['total_success_passed']),old_run_designated_failed=True,
                locked_targets_accessed=False,report='reports/PRIMARY_SEED_RESULT.md')
    (ROOT/'results/PILOT_COMPLETED.json').write_text(json.dumps(record,indent=2))
    (ROOT/'CURRENT_STAGE_STATUS.md').write_text('# Primary seed pilot complete\n\n'+verdict+'\n\nOnly seed 20260920 ran. Seeds 20260921 and 20260922 remain unstarted. Do not expand automatically. The large operating-cost objective is '+('met at pilot level only' if record['cost_goal_achieved'] else 'not achieved')+'.\n',encoding='utf-8')
    manifest=dict(protocol_sha256=hashlib.sha256((ROOT/'configs/pilot_protocol.yaml').read_bytes()).hexdigest(),
                  split_sha256=hashlib.sha256((ROOT/'configs/splits_2024_frozen.json').read_bytes()).hexdigest(),
                  source_reference=str(FAILED),original_project_access='READ ONLY',world_cache_copy=False,
                  model_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/models').glob('*.py')})
    (ROOT/'SOURCE_PATHS.yaml').write_text(yaml.safe_dump(manifest,allow_unicode=True,sort_keys=False),encoding='utf-8')
    print(json.dumps(record,indent=2),flush=True)

if __name__=='__main__':main()
