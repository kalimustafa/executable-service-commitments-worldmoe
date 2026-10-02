"""Measured evidence and paragraph replacements for the revised paper."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parent
PAPER=WORK/'新论文_市场可交付灵活性_20260917'
SOURCE=Path(r'C:\Users\Kali\Downloads\NatureEnergy_ExecutableFlexibility_FINAL_REFINED_20260923.docx')
OLD_SI=WORK/'EXTERNAL_VALIDATION_FINAL_20260922/05_supplementary/Supplementary_ExecutableFlexibility_20260922_EXTVALID_FINAL.docx'
OUT=PAPER/'交付稿_20260924_R2_APIC_完整修订'
def csv(name):return pd.read_csv(ROOT/'results'/name)
def record(name):return json.loads((ROOT/'results/paper_runs'/f'{name}.json').read_text())
def primary(w=24,p='interval',seed=20260920):
    return record(f'Eagle_2025_registered_seed{seed}_{w}h_{p}_particle_trailing4_normal')
def benchmark(model,w=24,p='point'):return record(f'baseline_{model}_2025_{w}h_{p}')
def mix(recourse=True,p='interval',shares='25_50_25'):
    return record(f'Eagle_2025_R2_seed20260920_mix{shares}_{p}_recourse{int(recourse)}')
M=primary();P=primary(p='point');M12=primary(12);P12=primary(12,'point')
summary=json.loads((ROOT/'results/PAPER_PRIMARY_RESULT.json').read_text())
fixed=mix(False);repaired=mix()
retained=100*M['value_usd']/P['value_usd']
energy_ratio=100*M['shifted_IT_mwh']/P['shifted_IT_mwh']
risk_reduction=100*(1-M['mtm_daily_loss_cvar95_usd']/P['mtm_daily_loss_cvar95_usd'])
delta_recourse=repaired['net_operating_cost_usd']-fixed['net_operating_cost_usd']
main=csv('multiseed_economics_2025.csv')
external=csv('external_r2_replications.csv')
components=csv('final_component_ablations.csv')
sensitivity=csv('original_parameter_sensitivities.csv')
grid=csv('grid_operational_metrics_2025.csv')
g=grid[grid.name.eq(M['name'])].iloc[0]
paired=pd.read_csv(ROOT/'figures/source_data/Fig3_cost_and_information_c_paired.csv')
ci24=paired[paired.window_h.eq(24)].iloc[0]
ci12=paired[paired.window_h.eq(12)].iloc[0]
forecast=csv('forecast_metrics_2025.csv')
f=forecast[(forecast.model=='R2_normal')&(forecast.publication=='post_DA')&(forecast.calibration=='native')]
price_only=components[(components.physical_branch=='price_only')&(components.window_h==24)&(components.policy=='interval')].iloc[0]
no_residual=components[(components.ablation=='force_gate_0')&(components.window_h==24)&(components.policy=='interval')].iloc[0]
fallback=csv('missing_context_replacement.csv')
fallback_change=max(abs(r.value_usd-primary(int(r.window_h),r.policy)['value_usd']) for r in fallback.itertuples())
tail=csv('settlement_tail_2026.csv')
TERMS=json.loads((ROOT/'configs/terminology_contract.json').read_text())
CAPTIONS=json.loads((ROOT/'configs/concise_figure_captions.json').read_text())

PATCH={}
PATCH[2]=(
    'Computing flexibility creates market value when delayed work remains executable. We combine anchored price-identity coupling (APIC), a reservation-state cost head and deadline-indexed execution in R2. '
    'The cost head converts publication-aware price distributions and causal photovoltaic updates into incremental site-cost scores. '
    f'In a four-zone 2025 replay, the frozen 24 h interval policy reduces service-adjusted annual electricity cost by {summary["mean_savings_pct"]:.2f}% on average across three seeds, with zero late work and zero closing work. '
    f'The primary seed saves USD {M["value_usd"]:,.0f} and retains {retained:.1f}% of point-policy saving using {energy_ratio:.1f}% of its shifted IT energy, while reducing daily-loss CVaR95 by {risk_reduction:.1f}%. '
    f'Priority recourse removes {fixed["late_IT_mwh"]:.3f} MWh of late work from the registered mixed contract for a USD {delta_recourse:.2f} cost increase. '
    'Frontier workload replications, hourly contract sweeps and observed-price boundary settlement support an operational framework linking consistent forecasts, capacity claims and service obligations.')
PATCH[6]=(
    'We connect the workload-dependent physical service floor to R2, the price-and-cost module in R-ECMoE-EF (Fig. 1). '
    'APIC couples ordered price and DA–RT spread distributions across publication stages. A reservation-state head ranks incremental electricity costs using booked capacity, issued PV scenarios and completed PV observations. '
    'The unchanged capacity calendar executes these rankings, with deadline-priority recourse for mixed contracts. Three frozen initializations are evaluated over the complete 2025 New York market calendar. '
    'The 24 h interval policy is the registered primary configuration; 12 h controls, hourly window sweeps and external workload replications resolve the contributing mechanisms.')
PATCH[14]=(
    'The execution layer carries released work and reservations across all 35,040 quarter-hour intervals. '
    f'At 7 h, the primary R2 (interval) policy saves USD {primary(7)["value_usd"]:,.2f} with zero late work; R2 (point) records {primary(7,"point")["late_IT_mwh"]:.3f} MWh late. '
    'The hourly sweep from 6 to 24 h separates the physical floor from the additional slack required by each ranking rule (Fig. 2c). Both R2 policies meet all deadlines at 12 and 24 h across all three seeds.')
sr=sensitivity[sensitivity.model.eq('R2')]
PATCH[15]=(
    f'The parameter sensitivity matrix contains {len(sr)} R2 replays across workload scales, service-capacity scales and 2–5% capacity overheads. '
    f'All {int((sr.late_IT_mwh<1e-8).sum())} R2 replays have zero late work. Each case uses its own demand- and capacity-matched earliest-service reference. '
    'Connection-limit exceedances are reported in the capacity-expansion scenarios, keeping physical service capacity and grid admissibility explicit (Supplementary Table 12).')
PATCH[18]='Persistent reservations add value under matched price information'
PATCH[19]=(
    f'Earliest service provides a conventional annual electricity-cost baseline of USD {M["reference_cost_usd"]:,.2f}. '
    'It includes the shared DA position, idle demand, cooling, PV curtailment charges and transaction friction. Service-adjusted policy cost additionally includes the liability for closing work. '
    'Table 1 reports the primary 24 h comparison; Fig. 3a follows the same account across 19 service windows.')
PATCH[20]=(
    'Matched Published DA controls isolate the effect of maintaining capacity claims. At 12 h, Published DA (myopic) saves USD 35,849.61 and Published DA (persistent) saves USD 39,794.66, an increment of USD 3,945.05 under identical price information. '
    'The expanded window sweep also identifies the longer-horizon reversal in this increment (Fig. 6c). These controls establish the execution contribution separately from the additional market and physical information used by R2.')
PATCH[21]=(
    'Table 1 | Primary 24 h electricity-cost comparison. Adjusted costs include shared-DA settlement, PV curtailment, friction and closing-work liability. '
    'Saving percentages use the same earliest-service reference. CVaR95 uses service-adjusted daily losses. R2 and Previous R2 use seed 20260920; complete seed results are in Supplementary Table 4.')
PATCH[22]='Reservation-state cost signals increase value under fixed execution'
PATCH[23]=(
    'R2 uses the 2021–2023 price panel for training and separate 2024 windows for stopping, calibration and economic selection. Its guarded residual branch starts from a fitted Base price model. '
    'APIC derives pre-publication RT prices from paired DA and spread draws and uses the published DA anchor after publication. '
    'The cost head uses only already released work, current reservations and completed PV observations; allocation follows the original kernel.')
PATCH[24]=(
    f'Across three seeds, post-publication spread pinball loss averages {f.dart_pinball.mean():.3f} US$/MWh. Mean 90% interval coverage is {100*f.coverage90.mean():.2f}%, and tail-event coverage is {100*f.tail_coverage.mean():.2f}%. '
    'Weekly trajectories and their differences from RASC expose when the forecast gains occur (Fig. 3b). All target rows, including the three dates without a native world packet, remain in the score (Supplementary Table 3).')
PATCH[25]=(
    f'Using the R2 price signal alone gives a 24 h interval cost of USD {price_only.net_operating_cost_usd:,.2f}; the complete cost head reduces this to USD {M["net_operating_cost_usd"]:,.2f}. '
    f'Removing the residual correction while retaining the cost head raises annual cost by USD {no_residual.net_operating_cost_usd-M["net_operating_cost_usd"]:,.2f}. '
    'The separately trained Base price model provides a second matched control. These interventions distinguish the residual forecasting branch from its operating-cost interpretation under the same physical account (Supplementary Table 7).')
PATCH[26]=(
    f'The primary R2 (interval) policy reduces 24 h annual cost to USD {M["net_operating_cost_usd"]:,.2f}, saving USD {M["value_usd"]:,.2f} ({M["saving_pct"]:.2f}%). '
    f'It shifts {M["shifted_IT_mwh"]:,.3f} MWh of IT service and closes with zero work outstanding. R2 (point) saves USD {P["value_usd"]:,.2f}, with {P["final_queue_IT_mwh"]:.3f} MWh retained in the closing-work account. '
    f'At 12 h, R2 (point) and R2 (interval) save USD {P12["value_usd"]:,.2f} and USD {M12["value_usd"]:,.2f}, respectively.')
PATCH[27]=(
    f'Against RASC (point), the primary R2 (point) saving increment is USD {ci12.difference_usd:,.2f} at 12 h '
    f'(paired seven-day-block 95% interval USD {ci12.lower95:,.0f}–{ci12.upper95:,.0f}) and USD {ci24.difference_usd:,.2f} at 24 h '
    f'(USD {ci24.lower95:,.0f}–{ci24.upper95:,.0f}). The complete hourly contract sweep shows how this increment develops with service slack (Fig. 3c).')
PATCH[28]=(
    f'The three 24 h interval costs span USD {summary["cost_range_usd"][0]:,.2f}–{summary["cost_range_usd"][1]:,.2f}, with a mean saving of USD {summary["mean_savings_usd"]:,.2f} ({summary["mean_savings_pct"]:.2f}%). '
    'All three seeds have zero late work and zero closing work. The architecture, calibration family and execution policy are retained as a set; the primary seed remains 20260920. '
    'The full seed-by-window matrix and causal RT-price-lag comparison accompany the source data.')
PATCH[29]=(
    f'Replacing the three missing-context dates with the frozen Base price model changes annual saving by at most USD {fallback_change:.2f}. '
    'Every replacement replay preserves the full physical calendar and meets the tested deadlines. A separate paired aggregation excludes those dates without truncating trajectories or annualising the subtotal (Supplementary Tables 13 and 14).')
PATCH[32]='Interval guidance balances saving, energy shifting and daily risk'
PATCH[33]=(
    f'At 24 h, R2 (interval) retains {retained:.2f}% of R2 (point) saving while using {energy_ratio:.2f}% of its shifted IT energy. '
    f'Service-adjusted daily-loss CVaR95 falls from USD {P["mtm_daily_loss_cvar95_usd"]:.2f} to USD {M["mtm_daily_loss_cvar95_usd"]:.2f}, a {risk_reduction:.2f}% reduction. '
    'Figure 4 follows this relationship over 19 service windows and reports the three-seed range.')
PATCH[36]=(
    f'Here the reference is the highest-saving deadline-feasible primary policy under the same contract. At 24 h, the interval policy has VCR = {retained/energy_ratio:.3f} and earns '
    f'USD {M["value_usd"]/M["shifted_IT_mwh"]:.2f} per shifted MWh of IT service, compared with USD {P["value_usd"]/P["shifted_IT_mwh"]:.2f} for point guidance. '
    'The cost and service accounts identify the physical intervention associated with each saving.')
PATCH[37]=(
    'Known reservations determine the incremental load created by a new cohort in each slot. Issued PV paths determine its expected locally supplied fraction, and completed observations update those paths causally. '
    'Mapping price quantiles through this incremental balance changes which admissible slots are attractive, even when the price forecast is fixed (Supplementary Table 7).')
PATCH[41]=(
    'The registered mixed contract assigns 25%, 50% and 25% of arrivals to 4, 12 and 24 h deadlines. '
    f'Under R2 (interval), fixed reservations leave {fixed["late_IT_mwh"]:.3f} MWh late. Priority recourse reallocates known, unexecuted work in deadline order while preserving each cohort’s original causal ranking.')
PATCH[42]=(
    f'Recourse removes all {fixed["late_IT_mwh"]:.3f} MWh of late work in this contract. Annual saving changes from USD {fixed["value_usd"]:,.2f} to USD {repaired["value_usd"]:,.2f}, '
    f'a USD {delta_recourse:.2f} electricity-cost increase ({100*delta_recourse/fixed["value_usd"]:.4f}% of the fixed-reservation saving). '
    'The eight-share sweep retains both successful repairs and small residual violations at alternative mixes (Fig. 5a–c).')
PATCH[43]=(
    f'For common deadlines, primary interval-policy saving rises from USD {primary(7)["value_usd"]:,.0f} at 7 h to USD {M12["value_usd"]:,.0f} at 12 h and USD {M["value_usd"]:,.0f} at 24 h. '
    'The three-seed hourly sweep provides 19 measured settings for each policy. Feasibility and value are assessed on the same continuous trajectories (Fig. 5d).')
PATCH[44]=(
    f'The primary 24 h interval policy reduces grid imports by {-g.import_change_top10pct_rt_mwh:.2f} MWh during the top 10% of realised RT-price intervals. '
    f'It reduces annual PV curtailment by {-g.pv_curtailment_change_vs_reference_mwh:.2f} MWh relative to earliest service. Maximum fleet recovery is {g.max_fleet_recovery_mw:.2f} MW. '
    'Grid, ramp and monthly profiles specify the timing and scale of the participant response (Supplementary Table 15).')
PATCH[48]=(
    'We repeat the original execution controls on Eagle and official Frontier workload traces in both 2024 and 2025, retaining their registered calendar mappings and resource-count sensitivity. '
    'Frozen R2 is additionally replayed on both Frontier years and the full 2024 Eagle calendar. Frontier demand is normalised to the original Eagle-2025 annual IT-service total. '
    'Each trace/year uses its own earliest-service reference; the 2024 R2 results are development-period sensitivities.')
PATCH[51]=(
    'The full-capacity service floors are 5.75 and 6.5 h for Eagle in 2024 and 2025, and 4 and 3 h for the corresponding Frontier traces. '
    'Eleven capacity fractions trace these workload-dependent floors (Fig. 6a). Matched 12 h Published DA controls reproduce positive persistence increments in all four trace/year pairs; the 19-window sweep identifies their subsequent evolution (Fig. 6b,c).')
PATCH[52]=(
    'Priority recourse is repeated across eight deadline compositions for the original external controls and the three registered compositions for new-R2 external replays. '
    'The heatmaps retain zero, small residual and stress-case violations explicitly (Fig. 6d). The alternate Frontier resource-count field repeats the registered mechanism comparison. '
    'Complete outcomes appear in Supplementary Tables 8–11 and 17–18.')
PATCH[53]=(
    'Longer service windows increase the value of refreshing unexecuted decisions. At 24 h in 2025, Published DA (myopic) exceeds Published DA (persistent) saving by USD 1,827.61 on Eagle and USD 2,306.56 on Frontier. '
    'The sign change identifies the contract-dependent trade-off between reserving capacity and retaining the option to re-rank work.')
PATCH[55]=(
    f'R2 turns service-constrained computing flexibility into measurable electricity-cost reductions. The frozen 24 h interval policy lowers annual cost by {summary["mean_savings_pct"]:.2f}% on average across three seeds, '
    'while completing every cohort on time and closing with no work outstanding. APIC supplies consistent price distributions, and the reservation-state cost head translates them into the marginal electricity consequence of each admissible placement.')
PATCH[56]=(
    f'The main gain comes from representing the operating decision directly. The full site-cost mapping lowers cost by USD {price_only.net_operating_cost_usd-M["net_operating_cost_usd"]:,.2f} relative to using the same R2 price signal alone. '
    'Booked load, expected PV absorption and completed observations connect the forecast to the quantity the executor changes, extending the prediction-to-decision principle.^9^ The ablations establish this contribution with a common workload, calendar and settlement account.')
PATCH[57]=(
    f'Interval guidance provides a clear trade-off: {retained:.1f}% of point-policy saving, {energy_ratio:.1f}% of its shifted IT energy and {risk_reduction:.1f}% lower daily-loss CVaR95 in the primary 24 h replay. '
    f'Under the registered mixed contract, priority recourse clears {fixed["late_IT_mwh"]:.3f} MWh of late work for USD {delta_recourse:.2f}. '
    'The dense service-window and deadline-share sweeps locate the conditions under which these mechanisms deliver their strongest gains.')
PATCH[58]=(
    'Eagle and Frontier produce different service floors and saving profiles under the same market calendar. The 12 h persistence gain recurs across four trace/year pairs, while the longer-window reversal exposes the value of preserving re-ranking options. '
    'Service contracts therefore define both the available energy response and the execution mechanism suited to capture it.')
PATCH[59]=(
    f'The primary interval policy absorbs an additional {-g.pv_curtailment_change_vs_reference_mwh:.1f} MWh of available PV and lowers imports in expensive intervals. '
    'Time-resolved recovery and ramp records make the resulting electrical obligations visible. Together with the work ledger, they provide concrete inputs for designing and assessing computing-flexibility commitments.')
PATCH[60]=(
    f'Three-seed evaluation, missing-context treatments and boundary settlement support the same result. Settling remaining R2 work over the observed-price 48 h tail changes saving by at most USD {tail.difference_usd.abs().max():.2f}; '
    'every tail obligation meets its original deadline. The selected interval policy has no closing-work liability to replace.')
PATCH[61]=(
    'A useful computing-flexibility offer specifies when work may move, which capacity remains available and how the resulting electricity exposure is valued. '
    'R2 and APIC place these quantities in one executable account, providing an operational basis for reducing electricity expenditure while maintaining explicit service commitments.')
PATCH[64]=(
    'Price models use 2021–2023 training data. Separate 2024 windows determine stopping, calibration and economic selection; the post-calibration development block selects the 24 h interval configuration. '
    'Three seeds, calibrated weights and policy are frozen before the new 2025 runs. The complete 2025 calendar supplies retrospective annual evaluation. Main comparisons retain the same demand, capacities, publication gates, transaction friction and closing-work valuation.')
PATCH[78]=(
    'Policies within each market year share the issue-time DA schedule and physical forcing. The common DA settlement contribution is included in absolute costs and cancels in paired savings. '
    'Published DA uses the preceding-day 11:00 availability gate and a training-median RT fallback. The original myopic, persistent and deadline-priority allocation rules are reused. '
    'R2 additionally receives each trace’s causally released work and actual reservation calendar. Full-calendar 2024 R2 results are development-period sensitivities; fitted parameters remain unchanged.')
PATCH[82]=(
    'WorldMoE supplies issued reference-context representations. Live reservations and newly released work enter the cost head and capacity calendar. The actual cohort ledger advances service state. '
    'For common contracts, the cost adapter mirrors the unchanged deterministic calendar exactly. Mixed contracts supply the same cost function with the actual calendar at admission; recourse retains each cohort’s original ranking.')
PATCH[83]='R2 residual forecasting and APIC calibration'
PATCH[84]=(
    'The Base numerical encoder has width 128. Separate adapters project the frozen states from interaction blocks 3, 6 and 10, site summaries, expert states, global context and tail context into width-128 tokens. '
    'Market-conditioned attention combines these tokens with a width-64 daily-profile query. Before publication, the profile’s DA values are masked. A gated residual corrects median and increment logits before monotone decoding. '
    'Positive softplus increments preserve seven ordered DA and spread quantiles. A zero final residual projection preserves the fitted Base forecast at initialization. The total head contains 969,395 parameters; the frozen world contains 59,156,961.')
PATCH[85]=(
    'The price panel contains 102,144 training rows, 8,444 selection rows, 8,640 calibration rows, 17,668 later-2024 rows and 384 embargo rows. Seeds are 20260920, 20260921 and 20260922. '
    'During ten warm-up epochs, the Base encoder and heads are frozen and Base dropout is disabled; new modules use AdamW at 0.001. Joint training restores the best warm-up weights and optimizer moments, '
    'with learning rates 0.0003 for new modules, 0.00002 for the Base encoder and 0.00005 for Base heads. Batch size is 4,096, weight decay 0.0001, maximum duration 40 epochs and joint patience 12. '
    'The unchanged proper-score loss^8^ weights DA, spread and RT pinball terms by 0.5, 1 and 0.5, adds 0.05 regime cross-entropy, and regularises the gate and residual. '
    'Training uses BF16; selection uses FP32. Selected epochs are 11, 0 and 3. Epoch 0 is the fitted Base-equivalent checkpoint retained by the same selection rule.')
PATCH[86]=(
    'Native affine calibration fits 24 location and width parameters on 1 April–29 June 2024, using 250 Adam iterations at learning rate 0.02. Positive width multipliers are bounded between exp(−2) and exp(2). '
    'Calibration and final price predictions use FP32. All calibration parameters are fixed before final-year evaluation. APIC preserves the known-DA anchor and price identity after calibration.')
PATCH[87]=(
    'Decision configuration is selected over 21 August–31 December 2024 UTC, comprising 133 UTC days and 134 local dates. Selection first prioritises zero late work, '
    'then maximises saving minus 0.1 times the number of local dates times service-adjusted daily-loss CVaR95. The chosen residual forecasting branch, four-hour PV update and 24 h interval policy remain fixed across the remaining seeds and final-year runs.')
PATCH[88]='APIC and publication-aware information'
PATCH[89]=(
    'Let D = DA − RT denote the DA–RT spread. Anchored price-identity coupling (APIC) links its distribution to the two market prices. Before DA publication, ordered DA and spread inverse CDFs are sampled jointly '
    'using a learned Gaussian copula and 64 fixed scrambled Sobol normal pairs per seed. The dependence coefficient is 0.95 tanh of the dependence-head output. '
    'Piecewise-linear inverse CDFs interpolate the seven quantiles and continue the outer segments. Every paired draw obeys RT = DA − D, and reported means share the identity. After publication, DA = A is fixed and')
PATCH[91]=(
    'The complementary quantile index follows from subtraction by a fixed anchor. For cost prediction, released work and remaining capacity determine each slot’s incremental load. '
    'Expected additional PV absorption is averaged over issued scenario paths, whose weights use at most the previous four completed hours of observations. Gaussian discrepancy scales are fitted on 722 training dates in 2022–2023. '
    'Let φ denote the expected PV-covered fraction of incremental load. The price-equivalent cost score is (forecast PUE / training-reference PUE) times [(1 − φ) × RT price quantile − φ], '
    'where the second term uses the fixed US$1/MWh curtailment charge. These ordered, quantile-indexed cost scores feed point or interval ranking.')
PATCH[97]=(
    'The calendar assigns released work to remaining capacity between release and deadline. R2 (point) ranks predicted incremental-cost medians and adds the unchanged friction allowance 2f to deferred slots, with f = US$1/MWh. '
    'R2 (interval) prefers deferred slots whose upper cost score falls below the current lower score by more than 2f:')
PATCH[99]=(
    'Here c denotes the price-equivalent incremental-cost score, with decision time and site held common. Future slots use the upper score; the current slot retains its median score. '
    'All admissible slots remain available for unavoidable work through the original stable capacity-allocation rule. Scores are emitted in FP32 so equal PV-covered slots preserve deterministic tie order.')
PATCH[107]=(
    'Frozen-forecast sensitivities scale workload and nominal capacity by 0.75× and 1.25×, and remove 2% or 5% of usable capacity without changing nominal electrical coefficients. '
    'Each case uses a matched earliest-service reference. Interconnection excess is reported explicitly in capacity-expansion scenarios. Eight mixed shares extend the original 25/50/25, 50/25/25 and 25/25/50% contracts. '
    'Grid diagnostics retain the original definitions: ex post realised-price and reference-load quantiles, positive paired import recovery, 15-minute ramp differences, and annual PV use and curtailment.')
PATCH[113]=(
    f'Absolute electricity cost includes two-settlement energy purchases, curtailment charges and friction. Service-adjusted cost additionally values closing work. The common DA settlement contribution is USD {M["shared_da_component_usd"]:,.2f}; '
    f'including it in every policy yields the USD {M["reference_cost_usd"]:,.2f} earliest-service baseline. This account is used consistently for absolute costs, saving percentages and matched comparisons.')
PATCH[118]=(
    'The primary risk series subtracts each interval’s change in outstanding-work valuation from incremental cash saving. Its annual sum equals service-adjusted saving. CVaR95 averages exactly the largest 5% probability mass of daily losses, '
    'including a fractional boundary day.^15^ Maximum drawdown uses quarter-hour increments. Paired differences use 2,000 circular seven-day block resamples with seed 20260920 and describe within-calendar financial variability. '
    'Seed ranges describe initialization variability. Feasibility is assessed on the original continuous cohort trajectory.')
PATCH[120]=(
    'The three seeds share architecture, data splits, optimizer settings, calibration form and execution policy. Initialization, minibatch order, dropout and each model’s fixed Sobol quadrature realization follow its registered seed. '
    'All three calibrated checkpoints are retained without final-year winner selection. Primary mechanism figures and Table 1 use seed 20260920; means and ranges explicitly identify three-seed aggregation.')
PATCH[121]=(
    'Missing-context sensitivity uses 27–28 May and 12 November 2025. One comparison excludes these dates only from paired daily aggregation while retaining physical trajectories. '
    'The second substitutes the frozen Base price model’s distributions on those dates and replays the complete year. The Base control retains its development-fitted native calibration. Neither treatment updates model parameters or physical inputs.')
PATCH[124]=(
    'Experiments use Python 3.13, PyTorch 2.9.0 and CUDA 12.6 on the registered RTX 3080 system. Frozen-world inference uses BF16 and context storage FP16; R2 selection, calibration, prediction and cost-score output use FP32. '
    'Contexts are cached once and reused across seeds. Execution advances every 15 minutes using the original allocation and accounting functions. '
    'The local experiment archive retains checkpoints, lineage, cohort assignments, daily outcomes and figure observations. Archived scripts generate the measured tables and figures from these records. OpenAI Codex assisted scripting and manuscript preparation.')

PATCH[72]=(
    'Under common release-relative deadlines, earliest-deadline order equals release order. Equation (1) follows from the work prefix present at each release under earliest service. '
    'For divisible work, constant service capacity and the specified nonbinding connection, it gives the physical minimum feasible common window. '
    'Eleven capacity fractions use the identical arrival sequence. The Supplementary Information gives the proof and Supplementary Table 19 reports the capacity sweep.')
PATCH[75]=(
    'Elapsed seconds from each header’s UnixStartTime are mapped to the corresponding NYISO local-year opening. This constant five-hour UTC phase translation preserves event order and interarrival durations over the complete 366- or 365-day calendar. '
    'One scalar maps annual submitted-node demand to the frozen Eagle-2025 service total, 2,981.607392515 MWh. The coefficients are 9.29566755×10⁻⁵ and 1.16001300×10⁻⁴ total model work units per submitted node for 2024 and 2025, respectively; work is split equally across four equal-capacity sites. '
    'The physical floor is recomputed at eleven capacity fractions from 50% to 100%. PV, PUE, prices, fallback prices, commitment schedules and service capacities are held common between workload sources within a market year. '
    'No Frontier workload is used for model fitting. Frozen price forecasts are shared; live arrivals enter the cost head and executor at release.')
PATCH[93]='Forecast comparators'
PATCH[94]=(
    'RASC uses LightGBM quantile regressors^13^ trained on the 2021–2023 panel and a 2023 subset, with 300 trees, learning rate 0.04, 15 leaves, minimum leaf size 100, L2 penalty 10, 127 bins and seed 20260917. '
    'Selection over 2024 chooses zero weight on the recent-only component. Its residual calibration uses the same April–June dates, zone/volatility/six-hour groups and shrinkage 84. '
    'The seven quantiles are 0.05, 0.10, 0.25, 0.50, 0.75, 0.90 and 0.95. Published DA supplies a post-publication RT point comparator, equivalent to zero predicted DA–RT spread.')
PATCH[95]=(
    'Previous R2 denotes the earlier frozen price-only module using the same W1 lineage, original data splits and development-fitted native calibration. '
    'Its archived predictions are replayed through the common point and interval reservation rules without selecting a new checkpoint. '
    'The Base price model supplies the matched control for the new residual branch. Table 1 and the hourly source matrix distinguish these comparators from the proposed R2 cost head.')
PATCH[101]=(
    'Matched Published DA (myopic) and Published DA (persistent) controls are replayed at each integer service window from 6 to 24 h. '
    'The myopic executor replans released but unexecuted work every interval and executes only the current slot. Its feasibility hedge forces current service whenever outstanding work would otherwise exceed the remaining post-current-slot capacity of a deadline prefix. '
    'Both controls use the same causal price ranks. The retained earliest-slot control establishes their neutral-ranking equivalence, and the 12 h unhedged variant measures the consequence of removing this service protection (Supplementary Table 21).')
PATCH[115]=(
    'The physical tail spans 1–2 January 2026 local time, adding 192 quarter-hours without new releases or forecast updates. All 2025 reservations retain their original deadlines and execution slots. '
    'Official NYISO integrated hourly RT prices from the January 2026 archive are held over their four quarter-hours.^10^ NASA POWER inputs drive the archived PV model; PUE comes from the frozen physical twin with the held 31 December template multiplier. '
    'Tail electricity use remains simulated. Each policy’s original closing-work liability is replaced with its incremental tail settlement and friction relative to the corresponding no-new-work reference. '
    'Shared DA terms cancel in this paired comparison. Supplementary Table 16 reports all twelve R2 seed, window and policy combinations.')
