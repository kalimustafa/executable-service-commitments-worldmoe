"""Answer the master goal directly, retaining gains and non-improvements."""
from pathlib import Path
import json,pandas as pd
ROOT=Path(__file__).resolve().parents[1];M=ROOT/'manuscript';O=ROOT/'results'
def main():
    assert json.loads((O/'analysis_completed.json').read_text())['status']=='complete'
    f=pd.read_csv(O/'policy_comparison_2025.csv');fm=pd.read_csv(O/'R2_forecast_metrics.csv');mix=pd.read_csv(O/'heterogeneous_deadline_R2_results.csv')
    def row(model,policy,w=12):return f[f.model.eq(model)&f.policy.eq(policy)&f.window_h.eq(w)].iloc[0]
    rp=row('R2_full_native','point');ri=row('R2_full_native','robust');op=row('Old_RASC','point');oi=row('Old_RASC','robust');da=row('Published_DA','point');nr=row('Old_R_action','native');nw=row('R2_no_world_native','robust')
    eligible=f[f.window_h.eq(12)&f.deadline_feasible];primary=eligible[eligible.primary_comparison];best=primary.loc[primary.value_usd.idxmax()];eff=primary.loc[primary.value_per_shifted_mwh.idxmax()];risk=primary.loc[primary.mtm_daily_loss_cvar95_usd.idxmin()]
    mre=mix[mix.model.eq('R2_full_native')&mix.policy.eq('robust')];fixed=mre[~mre.recourse].iloc[0];rec=mre[mre.recourse].iloc[0]
    counts=json.loads((M/'manuscript_structure.json').read_text());neural=json.loads((ROOT/'neural_2025/completed.json').read_text())
    text=f'''# R2 与可执行柔性：最终实验报告

## 1. W0 是否完全冻结？

**YES。** 选定配套谱系为 seed 20240829，W0 epoch 80 → W1 epoch 80。新实验中 W0 来源参数可训练数为 0。W0/W1 是一个模型的历史训练阶段，执行时冻结最终 W1 全快照，而非把两个网络重复串联。单独 W0 的最低验证分数来自 seed 20240813，但与本次 W1 不混接。原 W0 诊断 FAIL 保留，未被改写为 PASS。

## 2. W1 是否完全冻结？

**YES。** 最终世界模型 59,156,961 参数全部冻结，可训练数为 0，不进入 R2 优化器。W1 在共同 2024 验证的最差跨度评分为 0.568950799，三个已有兼容谱系中最低；没有用 2025 选 W0/W1。

## 3. 是否需要 action adapter？

**NO，针对本次实际部署接口。** 直接将持续队列输入原 W1 队列头的诊断为 SEVERE：条件单步队列 RMSE 从 0.069 升至 2.50–3.90。但 R2 使用已发布参考情景表征，实时执行队列和新调度动作不进入其市场分支；精确队列、容量与截止时间由显式执行层推进。该隔离不证明原 W1 能正确预测实时持续队列。若改变为实时队列反馈世界预测，应另行适配。旧动作对照和新 R2 执行的最终分布已补入 OOD CSV，未据此调参。

## 4. R2 相比旧 RASC 的预测如何？

相同 35,040 个小时—分区观测、日前公布后 DART：Pinball 5.922785 → 5.680674，改善 4.09%；MAE 16.944290 → 16.708286，改善 1.39%；90% 覆盖 87.73% → 88.48%；尾部样本 90% 覆盖 45.34% → 60.13%。中央 50% 覆盖 48.17% → 44.26%，存在欠覆盖。无世界 R2 的 Pinball 5.620475 更低，但尾部 90% 覆盖为 50.04%。因此，完整表征的优势是条件性的，不是全部指标占优。全指标见 results/R2_forecast_metrics.csv 与 R2_calibration_metrics.csv。

## 5. R2 + EF 比 old RASC + EF 增加多少全年价值？

12 小时点预测：US${rp.value_usd:,.2f}，增加 US${rp.value_usd-op.value_usd:,.2f}（{100*(rp.value_usd/op.value_usd-1):.2f}%）。12 小时区间：US${ri.value_usd:,.2f}，增加 US${ri.value_usd-oi.value_usd:,.2f}（{100*(ri.value_usd/oi.value_usd-1):.2f}%），服务调整后日 CVaR 下降 {100*(1-ri.mtm_daily_loss_cvar95_usd/oi.mtm_daily_loss_cvar95_usd):.2f}%。24 小时点预测增加 US$3,990.11，区间增加 US$6,611.59。完整窗口和配对区间见 results/paired_policy_differences.csv。

## 6. 增量分别来自什么？

同一 12 小时合同下：旧 R 为 US${nr.value_usd:,.2f}；可执行日历＋已公布 DA 为 US${da.value_usd:,.2f}；替换为旧 RASC 点预测变动 US${op.value_usd-da.value_usd:,.2f}；替换为 R2 点预测变动 US${rp.value_usd-op.value_usd:,.2f}；增加区间筛选变动 US${ri.value_usd-rp.value_usd:,.2f}，同时减少 {100*(1-ri.shifted_IT_mwh/rp.shifted_IT_mwh):.2f}% 调节电量与 {100*(1-ri.mtm_daily_loss_cvar95_usd/rp.mtm_daily_loss_cvar95_usd):.2f}% 日 CVaR。

第一步同时改变执行器及可用价格评分，不能称为严格隔离的纯执行因果效应。后三步固定日历，能够给出条件增量。区间门控减少部分收益以集中价值与控制风险，不能虚构正收入贡献。各增量依赖比较顺序，不给出误导性的固定贡献百分比。世界表征单独比较：12 小时点策略 full−no-world 为 +US$909.94，区间为 −US$422.41。

## 7. 12 小时哪个策略价值最高？

预先声明的主要可行对照中为 **{best['name']}，US${best.value_usd:,.2f}**。所有固定校准敏感性中，外部校准 R2 点预测为 US$40,771.73；它未被用来替换已由 2024 选定的主模型。

## 8. 12 小时哪个策略单位调节价值最高？

主要可行对照中为 **{eff['name']}，US${eff.value_per_shifted_mwh:.2f}/MWh-IT**。完整 R2 区间为 US${ri.value_per_shifted_mwh:.2f}/MWh-IT，VCR={ri.VCR:.3f}，保留 {100*ri.VRR:.2f}% 主对照最高价值，使用 {100*ri.RF:.2f}% 调节量。原始未校准 R2 区间的敏感性结果具有更高单位值，但总价值较低，未按测试结果重选。

## 9. 哪个策略 CVaR 最低？

12 小时主要可行对照中为 **{risk['name']}，服务调整后日 CVaR95=US${risk.mtm_daily_loss_cvar95_usd:,.2f}**。单看低风险可能偏好几乎不响应的策略，必须同时阅读收益和调节量。R2 区间 US${ri.mtm_daily_loss_cvar95_usd:,.2f}，低于旧 RASC 区间 US${oi.mtm_daily_loss_cvar95_usd:,.2f}；无世界消融为 US${nw.mtm_daily_loss_cvar95_usd:,.2f}，也完整保留。全年、各窗口、现金风险与服务风险均在政策 CSV 中。

## 10. 是否保持 deadline feasibility？

完整 R2 区间在 7、12、24 小时均为零逾期，R2 点策略在 12、24 小时为零逾期，7 小时逾期 9.821 MWh。无世界区间 7 小时逾期 0.431 MWh。所有义务跨日继承，年末未完成工作保留并估值。不存在逾期删除、午夜清空或收费替代履约。

## 11. 混合截止时间下重排是否消除逾期？

**YES，在所有已测试的混合合同策略中。** 完整 R2 区间从 {fixed.late_IT_mwh:.6f} MWh 逾期降至零，价值仅变化 US${rec.value_usd-fixed.value_usd:.6f}，重排后价值 US${rec.value_usd:,.2f}。这是全年观测结果；未来任务是否可行由已释放工作的截止时间前缀条件判断，不自动保证未知到达。

## 12. 是否支持“可执行柔性比名义柔性更重要”？

**YES。** 低平均利用率仍对应 6.5 小时物理服务下限；相同价格预测接入预约机制后形成不同执行规模；混合截止时间需要优先级保护。证据支持以工作义务和真实容量解释价值，而非用装机容量或单个预测误差推断收益。

## 13. R2 是否提供既有对照之外的实质增量？

**YES，但最有力的证据位于 24 小时点策略。** 它比旧 RASC 多 US$3,990.11（七日配对区块 95% 区间 US$1,831.80–6,296.97），比已公布 DA 多 US$4,626.53（US$2,026.47–7,303.40）。12 小时观测差额为正，但配对区间跨零；区间策略的绝对增量也不能仅凭单年点估计宣称普遍显著。所有区间均为同一年度内变化的描述。

## 14. 最终是否推荐 W0→W1→R2→EF？

**YES，作为有数据支持、职责清晰的候选系统；不是所有模块均已证明不可替代的唯一最优架构。** W0/W1 应画为一个冻结主干的训练谱系，R2 负责市场分布和风险信息，EF 负责精确任务与容量。完整表征在 7 小时区间履约、尾部覆盖和 12 小时点策略有价值，而无世界消融在部分价格和区间财务指标更好。因此论文中心是市场信息连接可执行服务，而不是网络规模本身。

## 完成记录

39 组显式执行情景，每组 35,040 连续时段；3 组配套旧 R 连续轨迹全部完成。旧 R 回放耗时 {neural['seconds']/60:.1f} 分钟，R2/EF 年度实验 5.2 分钟。英文正文 {counts['main_words_excluding_legends_tables']} 词，摘要 {counts['abstract_words']} 词，5 张主图、1 张主表、19 张补充表。英文正文含 Methods；英文补充单独提供；中文阅读稿合并正文、方法、参考文献和补充材料。全部使用纵向页面。

训练、选择、校准在 2025 之前完成。2025 年此前已用于机制开发，因此本研究是冻结参数的回顾性评估，不宣称第一次盲测。未修改 W0/W1 或旧 R 检查点，未按 2025 年结果重新挑模型。没有不必要的独立时隙排序头训练。
'''
    (ROOT/'FINAL_R2_REPORT.md').write_text(text,encoding='utf-8')
    article=(M/'NatureEnergy_ExecutableFlexibility_R2_EN.md').read_text(encoding='utf-8')
    body=article.split('## Methods')[0];methods=article.split('## Methods',1)[1].split('## Data availability')[0]
    patch='# Manuscript R2 revision\n\n## Title options\n\n1. Executable service commitments unlock the market value of computing flexibility\n2. Market intelligence acquires value through executable computing flexibility\n3. Service deadlines shape the value of grid-responsive computing\n\n## Revised abstract\n\n'+(M/'abstract_en.md').read_text(encoding='utf-8')
    patch+='\n\n## Architecture paragraph\n\n'+body.split('Here we connect')[1].split('\n\n')[0].join(['Here we connect',''])
    patch+='\n\n## Methods: R2\n\n'+methods.split('### R2 distributional model, training and calibration')[1].split('### Forecast baselines')[0]
    patch+='\n\n## Results: forecasting\n\n'+body.split('## Price distributions improve forward market information')[1].split('## Execution determines')[0]
    patch+='\n\n## Results: attribution and risk\n\n'+article.split('## Execution determines scale and prediction adds value')[1].split('## Deadline priority')[0]
    patch+='\n\n## Discussion replacement\n\n'+body.split('## Discussion')[1]
    patch+='\n\n## Figure captions\n\n'+'\n\n'.join(x for x in body.split('\n\n') if x.startswith('**Fig.'))
    (ROOT/'MANUSCRIPT_R2_PATCH.md').write_text(patch,encoding='utf-8')
    print('Report and manuscript patch written')
if __name__=='__main__':main()
