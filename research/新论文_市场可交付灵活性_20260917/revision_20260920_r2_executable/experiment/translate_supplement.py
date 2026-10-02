"""Parallel Chinese notes with identical, source-generated table values."""
from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[1];M=ROOT/'manuscript'
sections=json.loads((M/'supplement_table_sections.json').read_text(encoding='utf-8'))
titles=['冻结检查点的配套选择','日前公布后的概率预测精度','全年连续策略比较','预测与世界表征的条件贡献','混合截止时间与优先重排','模型与开发配置','条件单步队列迁移诊断','物理调节与年末义务','日前公布前的预测','12 小时校准与价值集中度敏感性','历史神经动作对照','RASC 服务窗口扩展','容量和物理边界细化','声明的工作负载利用率情景','传输约束下的迁移情景','有界主动日前仓位','独立辅助服务价格上限','激活恢复敏感性','附加实施成本交叉点']
descriptions=[
'分数越低越好，选择 seed 20240829 配套谱系。单独 W0 的最低分属于 seed 20240813，但不与另一 W1 混接。使用既有“仅验证评分”选择规则，原生 W0 资格标记仍为 FAIL，选定 W1 为 PASS；数值选优不覆盖原诊断标记。',
'损失和误差单位为美元/MWh，覆盖率为百分比。所有模型采用相同的分区训练期尾部阈值，不拟合任何 2025 年预测参数。',
'电量单位为 MWh-IT，CVaR 为服务调整后的日损失，单位为美元。价值包含交易摩擦与年末未完成工作估值。',
'差异为前者减后者，区间来自 2,000 次配对循环七日区块重采样，描述年度内变动，并非新增独立运行年份。',
'25%、50%、25% 工作分别采用 4、12、24 小时窗口，独立尽早截止时间参考为零逾期。重排仅改变尚未执行的预约。',
'完整与无世界特征版本使用相同训练配置，消融屏蔽全部世界来源通道并单独训练权重。',
'17 个 2024 年时点对应每种策略 68 个站点预测，误差单位为模型工作单位。该条件压力诊断表明直接迁移队列头存在严重误差，不是全年预测评分。实际 R2 市场分支排除实时执行队列。',
'恢复量为相对尽早服务的正 IT 服务偏差，电网调节量为购电绝对配对偏差。年末工作若截止时间位于下一年，是需要估值的义务，不自动等于违约。',
'本门控下 DA 尚未知。覆盖率与宽度描述联合构造的边际分布；恒等式检验针对成对样本和一致均值。',
'完整 R2 的内置校准由 2024 年经济表现选定。原始、外部校准及世界消融均保留，不按 2025 年结果重选。比值分母取最高价值的可行主对照，所以消融结果可以超过一。',
'此表为 seed 20240813、1 小时时钟的历史结果，与本次配套 seed 20240829、相同服务窗口的神经对照分开。价值已考虑年末工作，缺电独立报告。',
'这些扩展使用冻结的原 RASC。逾期是物理工作量，不是罚款。年末工作保留并估值，不强行清空。',
'在独立计算工作负载—容量物理下限后，追加边界点，不修改预测或准入参数；这些边界比较属于回顾性机制分析。',
'使用原冻结 RASC，设备容量保持不变。利用率与可调比例是恒定到达、可分割工作情景，不代表实测商业部署。',
'pair 表示 WEST 与 N.Y.C. 之间两个有向通道，all 表示站点之间各有向链路。只有历史观测而没有 2025 年证据时，严格可用性门控不开启任何链路。该路径不计入 R2 能量移峰总价值。',
'在一半接入容量的共同日前基准上，按表列比例扰动数量。这是有界数量敞口实验，并非已经出清的报价；采用原冻结 RASC。',
'仅计入可调计算工作和正购电，表内未含激活、资格和出清成本。展示 0.1 MW 门槛，0 与 1 MW 敏感性保留在源 CSV。两种负荷下全年始终可用的下限均为零，不等于季节容量认证结果。各产品不能相加为收入栈。',
'固定请求时间为每月 15 日当地 17:00–18:00，每站点最多 0.1 MW。全部推迟工作均恢复。时间从首个降载时段计，分辨率为十五分钟，不能据此认定快速响应资格或原任务截止时间全部满足。',
'成本是每调节 MWh-IT 的假设综合实施预算，不是逾期任务费用。负交叉点意味着在附加成本为零时，某策略已经以更少调节提供更多价值。该表使用原 RASC。']
mapping={'Window h':'窗口 h','Policy':'策略','Value US$':'价值 US$','Shifted IT':'调节 IT','CVaR US$':'CVaR US$','Debt CVaR US$':'服务 CVaR US$','Late IT':'逾期 IT','Closing IT':'年末 IT','Model':'模型','Pinball':'分位损失','50% cover':'50%覆盖','90% cover':'90%覆盖','Tail 90%':'尾部90%','First':'前者','Second':'后者','Δ value US$':'价值差 US$','Lower 95%':'95%下限','Upper 95%':'95%上限','Recourse':'重排','Yes':'是','No':'否','Item':'项目','Specification':'配置','Context':'上下文','Sites':'样本数','RMSE ratio':'RMSE倍数','Recovery IT':'恢复 IT','Peak MW':'峰值 MW','Grid MWh':'电网 MWh','Target':'目标','90% width':'90%宽度','Seed':'随机种子','W0 epoch':'W0轮次','W1 epoch':'W1轮次','W0 score':'W0评分','W1 score':'W1评分','Shortage MWh':'缺电 MWh','Capacity':'容量比例','Utilization':'利用率','Eligible':'可调比例','Routes Gbit/s':'路由 Gbit/s','Moved IT':'迁移 IT','Fraction':'比例','Cash CVaR US$':'现金 CVaR US$','Drawdown US$':'回撤 US$','Profile':'负荷情景','Product':'产品','Gross ceiling US$':'毛上限 US$','MW hours':'MW·小时','Site hours':'站点小时','Deferred IT':'推迟 IT','Recovered IT':'恢复 IT','Energy cost US$':'电量成本 US$','Interval versus':'区间策略对比','Crossing US$/MWh':'交叉点 US$/MWh','Deadline feasible':'截止可行','Not both':'非双方可行','Both':'双方可行','Original':'原始','10 min reserve':'10分钟备用','30 min reserve':'30分钟备用','Regulation capacity':'调频容量','60% utilization':'60%利用率','Archived issue':'原归档发布时间','Closed links':'关闭链路','2025 evidence absent 0':'2025证据缺失：0','Point':'点预测','Interval':'区间','Direct':'直接','Price lag':'价格滞后','Published DA':'已公布DA','R2 no world':'R2无世界','R2 external':'R2外部校准','R2 raw':'R2原始','Old R':'旧R','original_validation_actions':'原验证动作','Frozen world parameters':'冻结世界参数','R2 price-core parameters':'R2价格核心参数','Native calibration parameters':'内置校准参数','Inactive ranking head':'未启用排序头','World width / blocks':'世界宽度／块数','R2 width / expert width':'R2宽度／专家宽度','Training prices':'训练价格','Early stopping':'早停','Optimisation':'优化','Batch / dropout / patience':'批量／dropout／耐心','Calibration':'校准','Economic choice':'经济选择','2025 changes':'2025更新','Distribution identity':'分布恒等式','Live queue feedback to R2':'实时队列反馈R2','8,321; disabled':'8,321；未启用','2021–2023; 102,144 rows':'2021–2023；102,144行','2024 selection; selected epoch 1 for both variants':'2024选择；两个版本均为第1轮','AdamW; lr 0.002; weight decay 0.0001':'AdamW；学习率0.002；权重衰减0.0001','1 April–29 June 2024':'2024年4月1日至6月29日','21 August–31 December 2024 UTC':'UTC 2024年8月21日至12月31日','No fitting or threshold changes':'无拟合或阈值调整','Paired RT = DA − DART; complementary quantiles after DA':'成对RT=DA−DART；DA公布后互补分位','None; issued reference-context market branch':'无；采用已发布参考情景市场分支'}
notes='''## 补充材料

### 补充说明 1｜服务承诺可行性证明

恒定容量、相同释放后截止时间条件下，最早截止顺序等于释放顺序。时段 t 新批次到达后，直到该批次的剩余工作前缀为 Q(t)+a(t)，需要 ceil((Q(t)+a(t))/K) 个时段完成。更早批次具有更早截止时间，因此若尽早截止调度不可行，替换为更晚执行不能使它可行。反之，每个前缀均按时完成即得到可行性。取所需时段数的最大值即式（1）。到达时段本身可服务，因此等待 6.25 小时可能需要 6.5 小时服务窗口。该边界适用于指定容量、接入不约束的可分割工作模型，不可直接推广至任意不可抢占异构机器任务。

重排时，已知剩余任务都已释放，合法时隙构成由截止时间界定的嵌套前缀。式（5）的必要性来自较早到期的工作不能使用更晚容量。充分性可用归纳证明：按截止时间顺序分配，较早任务在每个后续前缀中占用的容量不超过其总工作量，因此前缀不等式保证下一批任务仍有足够合法位置，与内部价格排序无关。证明针对当前已知任务；不可预见的新到达需要在发生后重新测试。溢出工作保留并记逾期，不删除。

### 补充说明 2｜财务与物理解释

电力价值相对相同需求的尽早执行参考计算。工作需求、完成工作与商业计算收入是不同量，本研究测量前两项及增量用电价值。逾期与缺电是物理输出，不赋予假设罚款。源表同时保留现金风险和服务调整后风险，后者在未完成工作价值变化时即确认义务。CVaR 精确使用最差 5% 日期概率质量，回撤使用全部十五分钟增量。

瀑布图是有序比较。从旧 R 到已公布 DA 预约，既改变执行动作规则，也改变评分规则；之后预测替换固定执行层。区间门控选择更少响应，其负向总价值增量伴随较少干预和下行风险。完整与无世界 R2 是单独训练的配对表征比较，不能推导每个世界特征都必然获益，也不推导 Shapley 或与替换顺序无关的因果贡献比例。

### 补充说明 3｜不同市场路径

迁移采用声明的链路吞吐量，因为历史路由观测不能建立 2025 年路由可用性。主动日前数量在物理账户之外增加价差敞口。辅助服务按产品、按分区独立计价，不能据此证明资格、出清或同时交付，也不叠加进入能量移峰总价值。工作负载利用率情景保持设备不变，明确改变需求和可调比例。

### 补充说明 4｜扩展路径实现

迁移作用于新释放工作，至少经过一个完整时段后才在目的地服务。每个方向的容量为 $B10^{9}(900)/(8\\,2^{30}m)\\times0.00177935943$ 工作单位/时段，B 为 Gbit/s 链路速度，m=700 GiB/请求节点。载荷代理来自严格训练子集，链路速度是声明情景。测试 WEST 与 N.Y.C. 双向各 100 或 400 Gbit/s，以及各站点对 100 Gbit/s。目的地计算容量限制接收量。历史证据不覆盖 2025 年，故严格历史可用性门控在该年不开启迁移。

主动日前数量以接入容量一半为基准，增量限制为接入容量的 10%、25%、50%。增量结算为 −ΔtΣDΔq 减交易摩擦。正常日前截止时间为前一天 05:00；该情景假设相同预测在 04:55 生成，所有特征截止均更早。原归档 05:00 时戳单独处理，不能证明更早投标。方向采用价差中位数、50% 或 90% 区间，不使用目标日实际价格。物理和金融组合仅叠加固定控制的不同敞口。

辅助服务使用 2025 年全部十二个月官方日前价格。合格降载量受正购电和正在执行的可调计算量约束，排除不可调部分。小时内最小能力分别按旋转备用、30 分钟备用和调频容量价格估值，站点门槛为 0、0.1、1 MW。这些是产品毛价格上限，实际参与受资格、性能与需求侧项目向 DER 聚合转换规则约束。固定激活敏感性在每月 15 日 17:00–18:00 请求每站点最多 0.1 MW，然后利用剩余容量恢复工作；它检验十五分钟级恢复，不认证亚分钟调频或激活期间原任务截止时间。容量认证、里程收入和未签约接入补贴均不计入主结算。

### 补充说明 5｜服务可行价值前沿

对已测试策略集合，在给定服务窗口、容量和附加实施成本 c 下，取零逾期策略中“全年价值减 c×调节 IT 电量”的最大值。尽早服务只有满足相同截止时间时，才能以零增量价值进入。该前沿属于有限策略、固定执行的敏感性分析，不是全局最优调度。恒定到达情景采用 40%、60%、80% 可变计算容量，主要可调比例为 50%；60% 利用率另测 25% 和 100% 可调。不可调工作先占据每时段所需容量，其余容量分配给可调工作。容量折减对响应与参考同时使用原计算容量的 75% 或 50%。

'''
out=[]
for i,section in enumerate(sections):
    rows=[line for line in section.splitlines() if line.startswith('|')]
    translated=[]
    for line in rows:
        cells=line.split('|');new=[]
        for cell in cells:
            content=cell.strip()
            if content in mapping:content=mapping[content]
            else:
                for a,b in [('Published DA','已公布DA'),('R2 no world','R2无世界'),('R2 external','R2外部'),('R2 raw','R2原始'),('Old R','旧R'),(' interval',' 区间'),(' point',' 点预测'),(' action',' 动作')]:content=content.replace(a,b)
            new.append(content)
        translated.append('| '+' | '.join(new[1:-1])+' |')
    out.append(f'### 补充表 {i+1}｜{titles[i]}\n\n{descriptions[i]}\n\n'+'\n'.join(translated)+'\n')
(M/'supplement_cn.md').write_text(notes+'\n'.join(out),encoding='utf-8')
print('Chinese supplementary tables',len(out))
