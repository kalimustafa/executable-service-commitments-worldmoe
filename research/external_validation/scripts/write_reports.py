"""Final source-backed answers, provenance, visual contract and run instructions."""
import json,shutil
from datetime import datetime,timezone
import pandas as pd,numpy as np
from paths import *

def table(df):
    heads=list(df.columns);lines=['| '+' | '.join(heads)+' |','| '+' | '.join(['---']*len(heads))+' |']
    lines+=['| '+' | '.join(str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)]
    return '\n'.join(lines)
def main():
    s=pd.read_csv(RESULTS/'external_validity_summary.csv').set_index(['trace','market_year'])
    runs=pd.read_csv(RESULTS/'all_mechanism_runs.csv');fl=pd.read_csv(RESULTS/'trace_service_floor_all_proxies.csv')
    mixed=pd.read_csv(RESULTS/'all_recourse_runs.csv');chars=pd.read_csv(RESULTS/'frontier_trace_characteristics.csv')
    bad=runs[runs.policy.eq('persistent_da')&runs.proxy.ne('field7_sensitivity')&runs.persistence_increment_usd.lt(0)].copy()
    bad.to_csv(RESULTS/'non_positive_persistence_cases.csv',index=False)
    r24=s.loc[('Frontier',2024)];r25=s.loc[('Frontier',2025)];e24=s.loc[('Eagle',2024)]
    aud=json.loads((RESULTS/'replication_completed.json').read_text())
    report=f'''# 外部有效性与第二市场年份复现结果

## QUESTION 1 换成 Frontier 后物理服务下限是否仍存在

是，但不是统一的 6.5 小时。主节点请求代理下，Frontier-2024 为 **4 小时**，Frontier-2025 为 **3 小时**；Eagle-2024 为 5.75 小时，Eagle-2025 为 6.5 小时。Frontier 两个年份都预先归一化为 **2,981.607392515 MWh-IT**。75%/50% 容量下，Frontier-2024 下限为 6/9.75 小时，Frontier-2025 为 4/7 小时。Frontier-2025 的最大前缀发生在轨迹零时刻；全部 1,000 项零时刻提交保留，没有删除起始突发。

## QUESTION 2 Frontier 上 Published DA 调度是否仍降低成本

是。12 小时持续预约在 Frontier-2024 节省 **US${r24.persistent_da_saving:,.2f}（{r24.persistent_da_saving_pct:.2f}%）**，在 Frontier-2025 节省 **US${r25.persistent_da_saving:,.2f}（{r25.persistent_da_saving_pct:.2f}%）**。对应尽早执行的服务调整后成本为 US${r24.earliest_cost:,.2f} 和 US${r25.earliest_cost:,.2f}。短视与持续预约在所测试主 7/12/24 小时窗口中均有正节省；服务可行性须另读逾期列。

## QUESTION 3 相同 Published DA 信息下持续预约是否仍优于短视执行

**在预先指定的 12 小时匹配比较中是，在所有窗口上不是。** Frontier-2024 的增量为 US${r24.persistence_increment:,.2f}，占持续策略节省的 {r24.persistence_share_pct:.2f}%；Frontier-2025 为 US${r25.persistence_increment:,.2f}，占 {r25.persistence_share_pct:.2f}%。四组主 12 小时配对的增量全部为正，且均无逾期。2025 年 24 小时的反例保留在 QUESTION 6。

## QUESTION 4 混合截止时间下重排是否仍保护服务

Frontier-2025 的固定预约逾期由 **{r25.fixed_mixed_late:.6f} MWh 降至 0**，全年价值变化为 US${r25.recourse_value_change:,.2f}。Frontier-2024 为 **0 → 0**，只能说明保持可行，不能说消除了原本不存在的违约。两个 Eagle 年份的逾期分别由 0.114107 和 2.547769 MWh 降至零。原资源字段敏感性也保留：Frontier-2024 的 8.396614 MWh 和 Frontier-2025 的 0.327832 MWh 逾期都被重排消除。各次重排只处理尚未执行预约，导出的历史执行记录与最终日历一致。

## QUESTION 5 换到 2024 年机制是否保持

12 小时下保持同方向。Eagle-2024 持续预约节省 **US${e24.persistent_da_saving:,.2f}（{e24.persistent_da_saving_pct:.2f}%）**，相对短视执行增加 **US${e24.persistence_increment:,.2f}**。混合合同逾期由 {e24.fixed_mixed_late:.6f} MWh 降至零，价值变化为 US${e24.recourse_value_change:,.2f}。2024 年完整覆盖 **35,136 个十五分钟时段、366 个当地日期**，含闰日及 92/100 时段的两个 DST 切换日。采用当年的官方价格、带日期的 NASA 派生光伏和冻结物理模型；没有复制 2025 光伏。2024 仅为执行机制年份复现，绝不是 R2 第二独立测试年。

## QUESTION 6 哪些结果没有复现

- **持续预约普遍占优没有复现。** 2025 年 24 小时下，Eagle 的持续减短视增量为 **−US$1,827.61**，Frontier 为 **−US$2,306.56**；两者均无逾期。持续保留既有时隙会限制后续重新排序，表明其贡献取决于合同与市场。
- **物理下限不等于每种在线价格策略都可行。** Eagle-2025 的物理下限是 6.5 小时，但 7 小时下带保护短视和持续 DA 仍分别有约 12.437、12.211 MWh 逾期；尽早执行可行。
- **Frontier-2024 主节点代理未出现固定混合预约失败。** 0→0 不能被包装为主动修复的外部证据。
- **具体下限和收益对资源代理敏感。** 使用原第 7 字段后，Frontier-2024 下限变为 8.25 小时、Frontier-2025 为 4.5 小时，12 小时持续预约增量仍为正。主代理依据日志头的明确节点属性选定，选择发生在新经济回放之前，未按收益择优。
- 不同年份与轨迹没有重复同一个 7.32%。2024 Eagle 保留正式映射，年工作量为 3,006.506796 MWh，比共同 Frontier 归一化目标高 0.835%；因此 2024 跨轨迹比较并非完全等工作量实验。

## QUESTION 7 新证据支持怎样的结论

可将证据范围从一个 2025 Eagle 回放扩展到 **一个额外系统的两个公开轨迹年份，以及第二个 NYISO 市场年份的 Published-DA 执行机制复现**。四组匹配 12 小时案例支持价格响应节省和正持续预约增量；所有实际发生的主混合合同违约均可通过原截止时间优先重排消除。结论应写成有服务合同条件的机制证据，保留 24 小时的逆向增量和字段敏感性。

## QUESTION 8 仍不能声称什么

- 不能把 2024 称为 R2 独立测试，或声称新实验验证了 R2/W1 架构泛化。
- 不能把 Frontier 与 NYISO 配对写成实际共址、同步商业运行或现场部署。
- 不能将提交需求代理视为实测设施用电，或将声明的截止时间视为观测商业 SLA。
- 不能声称验证了不可抢占任务、检查点开销、真实服务收入、商业利润或全电网收益。
- 不能声称持续预约在所有合同、年份和代理下都优于滚动重排。

## 12 小时主结果

'''
    tab=s.reset_index()[['trace','market_year','physical_floor_h','earliest_cost','myopic_da_saving','persistent_da_saving','persistent_da_saving_pct','persistence_increment','fixed_mixed_late','recourse_mixed_late']].copy()
    for col in ['earliest_cost','myopic_da_saving','persistent_da_saving','persistence_increment']:tab[col]=tab[col].map(lambda x:f'{x:,.2f}')
    for col in ['persistent_da_saving_pct','physical_floor_h']:tab[col]=tab[col].map(lambda x:f'{x:.2f}')
    for col in ['fixed_mixed_late','recourse_mixed_late']:tab[col]=tab[col].map(lambda x:f'{x:.6f}')
    report+=table(tab)+f'''

## 完成范围与验证

主均质回放 36 行、原资源字段敏感性 6 行、无保护诊断 4 行，以及主混合合同 4 对和资源字段混合敏感性 2 对全部保留。2024 的 12/24 小时点策略年末队列均为空，追加的 48 小时实测价格尾段对增量价值影响为零。工作守恒、无提前执行、物理容量与接入边界、历史执行不变及绝对成本闭合均已通过；冻结 2025 输入、结果和母稿哈希未改变。主 2025 三条定位行的最大成本/价值闭合误差小于 10⁻⁹ 美元。

英文主稿的新增结果段与补充表 31–38保留上述成功与非成功案例；Figures 2–6 使用统一配色，Figure 1 的图像字节、尺寸和 caption 保持原样。全部新图提供 SVG/PDF/600 dpi PNG 和 source CSV。最终排版记录见 document_build_manifest.json 与交付验收文件。

主要证据：01_results/external_validity_summary.csv、trace_service_floor.csv、all_mechanism_runs.csv、all_recourse_runs.csv、frontier_resource_field_sensitivity.csv、non_positive_persistence_cases.csv 与 replication_completed.json。来源与解释规则见 DATA_PROVENANCE.md。
'''
    # Values of the field-7 failures come directly from the saved rows.
    for year,literal in [(2024,'8.396614'),(2025,'0.327832')]:
        val=mixed[mixed.trace.eq('Frontier')&mixed.market_year.eq(year)&mixed.proxy.eq('field7_sensitivity')].fixed_late_mwh.iloc[0]
        report=report.replace(literal,f'{val:.6f}')
    (REPORTS/'EXTERNAL_VALIDATION_REPORT.md').write_text(report,encoding='utf-8')
    manifests=[json.loads((DATA/f'frontier_{y}/download_manifest.json').read_text()) for y in [2024,2025]]
    provenance='''# Data provenance and interpretation

## Official independent workload source

The JSSPP source page is https://jsspp.org/workload/index.php?page=frontier25. It supplies the two original ZIPs directly; no third-party mirror was used. The reference is Maheshwari, K., Klusáček, D., Suter, F. and Brewer, W., Workload traces from the Frontier exascale supercomputer, in Job Scheduling Strategies for Parallel Processing, LNCS 16980, Springer (2026). The archive supplies this publication metadata but no DOI; no DOI is invented.

'''
    for m in manifests:provenance+=f"- {m['year']}: {m['url']}\n  SHA-256: `{m['sha256']}`; bytes: {m['bytes']}; retrieved UTC: {m['downloaded_utc']}.\n"
    provenance+='''
The exact ZIP, extracted SWF, LICENSE.txt, header.txt, FIELDS.md, parsed job subset, mapped arrivals and parse-quality JSON are retained in external_validation/data/frontier_2024 and frontier_2025. The compact delivery retains the original ZIPs (containing the full SWF), licences, headers and mapped inputs.

Primary mapping uses field 20's explicit node-count property, because field 7 is described as nodes or cores and its ratio to property nodes varies from record to record. All properties match frontier:excl:Nx112; none are guessed or dropped. Field 7 is evaluated separately with its own fixed annual scalar. Runtimes are characterised but never multiplied into the primary work measure. The header's status-constant statement conflicts with the rows; every status is retained. Job IDs are unique, all submission offsets lie within the calendar, and all requested-node counts are positive.

The year origin in each header is UTC Jan 1. Elapsed seconds are translated to New York local Jan 1 midnight by a constant 5 h UTC offset, preserving complete event order and duration without losing boundary submissions or deleting leap day. This is a counterfactual phase pairing, not a claim of geographic co-location. 338 and 1,000 epoch-zero jobs remain in 2024 and 2025. Original maxima and zero-arrival intervals are retained.

## Normalisation and exposure

The target comes from the frozen Eagle-2025 forcing and existing IT conversion, not a fitted result: 2,981.607392515 MWh-IT. A single annual scalar preserves the within-year requested-node shape. There is no per-day adjustment, clipping, winsorisation or outcome-based scaling. Both Frontier years have exactly that annual demand. Eagle-2024 retains the original deterministic circular source-date map and frozen quantile bridge; its annual volume is 3,006.506796152 MWh-IT, 0.835% higher. The complete coefficients and statistics are in frontier_trace_characteristics.csv and PROTOCOL.json.

## Market and physical sources

The 2024 price panel is the existing official-source gate_hourly_panel.parquet, restricted to the complete New York local year. Twelve original official DA archives and twelve integrated-RT archives are identified by URL and SHA-256 in forcing_source_manifest.json. Their governing indices are https://mis.nyiso.com/public/P-2Alist.htm and https://mis.nyiso.com/public/P-4Alist.htm. The UTC timestamps include all 35,136 quarter-hours: local 10 March has 92, 3 November has 100, and 29 February has 96. Hourly prices are held over each four physical quarter-hours.

PV uses the existing dated NASA POWER/pvlib-derived 2024 series in nyiso_zone_pv_15m.parquet, with the required UTC boundary on 1 January 2025. PUE is recomputed through the frozen physical twin using the same seasonal same-weekday template rule, restricted to strictly pre-2024 issue-legal templates. Physical forcing is common between workload sources within each year, isolating the changed arrival proxy. No 2025 PV series is substituted into the 2024 year. Exact source hashes, template assignments and chronology are in forcing_source_manifest.json, physical_day_provenance_2024.csv and calendar_and_physical_audit_2024.json.

The unchanged Eagle work source map is target UTC date offset from 1 January 2022 modulo the ordered 1,086-date historical source list. No bridge refit or 2024 scale selection occurs. The shared 2024 DA schedule uses the original issue-load formula from training_data.py: clipped forecast facility load minus issue-time PV, based on frozen empirical pools. Every policy and workload trace in that year receives the same schedule. Its common annual DA-minus-RT contribution is US$8,256.690196458. The archived 2025 contribution remains US$2,409.784214306. Both are included in absolute costs and cancel in paired dollar savings.

## Information and interpretation boundaries

Published DA becomes available at 11:00 New York on the preceding day; unavailable target prices use pre-2024 RT medians. No future arrivals are visible to the reused kernels. 2024 previously informed predictive-model development, so this study is a second market-year mechanism replay, not an independent R2 evaluation. No W0/W1/R2 forward pass, training, stopping, calibration or selection is performed for external traces.

## Licence and acknowledgement

Dataset Frontier-2024-2025 arose under UT-Battelle Contract DE-AC05-00OR22725 with the US Department of Energy. The supplied licence requests academic attribution and acknowledgement of ORNL staff and e-INFRA CZ project ID 90254. The exact licence text is carried beside each ZIP. The manuscript acknowledges those contributors as dataset providers and does not imply they funded the present study.
'''
    (REPORTS/'DATA_PROVENANCE.md').write_text(provenance,encoding='utf-8')
    style='''# Visual style guide

Figure 1 is protected: original raster bytes, dimensions, caption and internal elements are retained. No Figure 1 drawing or export is performed.

Figures 2–6 use Arial, a white background (#FFFFFF), primary text #222222, secondary text #5B616A, major grid #E5E7EB and neutral uncertainty #BCC2CA. R2 is indigo #4F63B8; RASC and fixed reservations are copper #C47A44; persistent Published DA is slate #778493; myopic DA is light slate #A6B2BF; earliest service is charcoal #444A50; repaired recourse is #4359A5. PV green #719A77 is reserved but not used when no PV curve is needed.

Exports are 175 mm wide, SVG/PDF vector and PNG at 600 dpi. Tick text is 8 pt, axis labels 8.7 pt, annotations approximately 8 pt and panel labels 10 pt bold. Main lines are 1.3 pt, supporting lines 0.8–1.0 pt, intervals 1 pt, and markers 5.5 pt. Top/right spines are removed. Price-model identity is encoded by consistent colour and direct labels, with diamonds for interval policies. Marker area no longer encodes CVaR in Figure 4. Figure 6 uses circles for Eagle, squares for Frontier-2024 and triangles for Frontier-2025; 2024 markers are open and 2025 markers filled. Line styles distinguish physical-capacity fractions. These redundant encodings preserve identity without relying on colour alone. No gradients, shadows, 3D or rainbow scales are used.

Submission manuscripts retain the original title and single-column Letter page, with 2.3 cm margins, Times New Roman 11 pt body, 1.18 line spacing and 5 pt after-paragraph spacing. Chinese counterparts use SimSun for Chinese body text. Headings are black 12 pt, title black 18 pt. Captions are 9 pt with bold figure labels; image/caption pairs remain together. Table 1 retains absolute cost, adjusted cost, saving, saving percentage and CVaR; its header is #F2F3F5, body white, and borders are light horizontal rules. Native Word equations remain editable.

Source data are in 02_figures/source_data/fig2_source.csv through fig6_source.csv. A panel column separates the different tables underlying multipanel figures. Inapplicable columns in the stacked panel CSVs are structurally empty, not missing observations.
'''
    (REPORTS/'VISUAL_STYLE_GUIDE.md').write_text(style,encoding='utf-8')
    changes='''# Manuscript changes

- Mother manuscript: the delivered NatureEnergy_ExecutableFlexibility_20260922_EN.docx; title retained.
- The frozen 2025 experiments, forecasts, economic values, equations and Table 1 numerical entries remain unchanged. Figure 1's original image SHA-256 is 2c83c9f7ad1cdf188d6d5c1a66b7a6aaaf43868595c533a90f3420754eac2cca; dimensions and caption also remain unchanged.
- Abstract adds one qualified sentence about matched 12-hour persistence across the external traces and secondary market year. It contains 144 words.
- A new Results section reports four trace/year cases, trace-specific physical floors, positive 12 h increments, recourse outcomes and the negative 24 h persistence cases. Main narrative is 2,845 words before Methods.
- Discussion replaces the one-year/one-trace limitation with the primary locked 2025 predictive test plus secondary 2024 execution replay and independent Frontier records. It retains counterfactual, divisible-work, non-commercial-SLA, non-field-deployment and network-boundary qualifications.
- Methods adds independent Frontier replication and secondary 2024 replay, including field semantics, fixed normalisation, complete leap-year/DST handling, original Eagle mapping, annual financial positions and the no-R2-test boundary.
- Frontier reference and dataset acknowledgement are added without inventing a DOI.
- Figures 2–5 are redrawn from unchanged measured source data; Figure 6 presents external validation. Figure 4 removes marker-area risk encoding. Figure 5's magnified dollar scale is explicit.
- Supplementary Information adds Notes 19–23 and Tables 31–38, including all prescribed windows, resource-field sensitivities, unhedged failures and the empty-obligation 2024 tail.
- English and Chinese main texts and supplements are updated together. The equations remain native editable Word objects. No successful mechanism claim was written before the experiment outputs existed.
'''
    (REPORTS/'MANUSCRIPT_CHANGES.md').write_text(changes,encoding='utf-8')
    commands='''# Run commands

Run from C:/Users/Kali/Desktop/电力市场 in PowerShell. Scientific scripts reuse the registered project environment; document authoring uses the bundled runtime. No package installation or model retraining is required.

```powershell
& '.\\power-dc-vpp\\.venv\\Scripts\\python.exe' -X utf8 'external_validation\\scripts\\fetch_frontier.py'
& '.\\power-dc-vpp\\.venv\\Scripts\\python.exe' -X utf8 'external_validation\\scripts\\parse_frontier.py'
& '.\\power-dc-vpp\\.venv\\Scripts\\python.exe' -X utf8 'external_validation\\scripts\\prepare_years.py'
& '.\\power-dc-vpp\\.venv\\Scripts\\python.exe' -X utf8 'external_validation\\scripts\\run_replication.py'
& '.\\power-dc-vpp\\.venv\\Scripts\\python.exe' -X utf8 'external_validation\\scripts\\tail_2024.py'
& '.\\power-dc-vpp\\.venv\\Scripts\\python.exe' -X utf8 'external_validation\\scripts\\build_figures.py'
& 'C:\\Users\\Kali\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe' -X utf8 'external_validation\\scripts\\revise_documents.py'
& 'external_validation\\scripts\\render_final.ps1'
& 'C:\\Users\\Kali\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe' -X utf8 'external_validation\\scripts\\raster_qa.py'
& '.\\power-dc-vpp\\.venv\\Scripts\\python.exe' -X utf8 'external_validation\\scripts\\write_reports.py'
```

Original inputs and selected frozen physical files remain dependencies in power-dc-vpp and 新论文_市场可交付灵活性_20260917. They are identified by absolute path and hash in the source manifests; the compact delivery is not a redistributed neural-checkpoint bundle. The original dataset ZIPs and exact licences are included. Full generated cohort assignments and annual trajectory arrays remain under external_validation/runs in the working project.

All calculations are CPU execution/settlement replications. No W0/W1/R2 head or backbone is trained or inferred. run_replication.py completed its main and sensitivity set in approximately 291 seconds in the registered environment. The original year-end 2025 observed-price analysis remains part of the mother manuscript; tail_2024.py is the new year-boundary diagnostic.

The bundled DOCX renderer lacks a LibreOffice executable on this Windows host, so the verified native Word export route is reused. Intermediate PDFs and PNGs are internal QA files under external_validation/qa and are excluded from the delivered manuscript package.
'''
    (REPORTS/'RUN_COMMANDS.md').write_text(commands,encoding='utf-8')
    print('Five requested reports written from completed results')
if __name__=='__main__':main()
