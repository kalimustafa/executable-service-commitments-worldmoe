"""Formal supplementary tables from archived, completed numerical outputs."""
from pathlib import Path
import json,re
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];M=ROOT/'manuscript';PAPER=ROOT.parent
O=ROOT/'results'
NAMES={'Published_DA':'Published DA','Old_RASC':'RASC','R2_full_native':'R2','R2_no_world_native':'R2 no world','R2_full_raw':'R2 raw','R2_full_external':'R2 external','Old_R_action':'Old R','Seasonal_quantiles':'Seasonal','Ridge_residual_quantiles':'Ridge','RASC':'RASC','Published_DA_point':'DA point'}
def fmt(v):
    if pd.isna(v):return '—'
    if isinstance(v,(float,np.floating)):return f'{v:,.3f}'.rstrip('0').rstrip('.') if abs(v)<100 else f'{v:,.2f}'
    return str(v)
def table(frame):
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))+'\n'
def policies(f):
    return f.model.map(NAMES).fillna(f.model)+' '+f.policy.map({'point':'point','robust':'interval','native':'action'}).fillna(f.policy)
def main():
    chunks=[]
    def add(n,title,description,frame):chunks.append(f'### Supplementary Table {n} | {title}\n\n{description}\n\n'+table(frame))
    selected=pd.read_csv(ROOT/'w0_w1_validation_ranking.csv')
    # Build explicitly from the recorded phase selection records.
    if 'phase' in selected:
        wide=selected.pivot(index='seed',columns='phase',values=['selected_epoch','primary_score']).reset_index()
        rows=[{'Seed':int(r[('seed','')]),'W0 epoch':int(r[('selected_epoch','w0')]),'W0 score':f"{r[('primary_score','w0')]:.6f}",'W1 epoch':int(r[('selected_epoch','w1')]),'W1 score':f"{r[('primary_score','w1')]:.6f}"} for _,r in wide.iterrows()]
    else:
        rows=[{'Seed':20240813,'W0 epoch':128,'W0 score':.5377914086,'W1 epoch':56,'W1 score':.5707426920},
              {'Seed':20240829,'W0 epoch':80,'W0 score':.5412616907,'W1 epoch':80,'W1 score':.5689507990},
              {'Seed':20240917,'W0 epoch':96,'W0 score':.5391671332,'W1 epoch':64,'W1 score':.6226699415}]
    add(1,'Compatible frozen lineage selection','Lower scores are better. The selected pair is seed 20240829. The standalone best W0 score belongs to seed 20240813; it is not spliced into a different W1. Selection used the recorded validation-score-only policy. The native W0 eligibility flag remains FAIL for the selected parent, while the selected W1 native flag is PASS. Recording a numerical winner does not overwrite these original diagnostic flags.',pd.DataFrame(rows))
    fm=pd.read_csv(O/'R2_forecast_metrics.csv');q=fm[fm.target.eq('dart')&fm.gate.eq('post_DA')].copy()
    t=pd.DataFrame({'Model':q.model.map(NAMES),'Pinball':q.pinball,'MAE':q.mae,'50% cover':100*q.coverage50,'90% cover':100*q.coverage90,'Tail 90%':100*q.tail_coverage90})
    add(2,'Post-publication distributional accuracy','Scores are US$/MWh and coverage is percent. Tail events use the same zone-specific training threshold for every model. No 2025 forecast parameter is fitted.',t)
    f=pd.read_csv(O/'policy_comparison_2025.csv');g=f[f.primary_comparison].sort_values(['window_h','model','policy'])
    add(3,'Continuous annual policy comparison','Energy quantities are MWh-IT. CVaR is the service-adjusted daily-loss measure in US$. These values include transaction friction and annual closing-work valuation.',pd.DataFrame({'Window h':g.window_h,'Policy':policies(g),'Value US$':g.value_usd,'Shifted IT':g.shifted_IT_mwh,'CVaR US$':g.mtm_daily_loss_cvar95_usd,'Late IT':g.late_IT_mwh}))
    differences=pd.read_csv(O/'paired_policy_differences.csv');difference=differences[differences.role.isin(['forecast','representation','versus_published_DA'])].copy()
    def short(s):
        s=re.sub(r'_w\d+_', ' ',s)
        for key in sorted(NAMES,key=len,reverse=True):s=s.replace(key,NAMES[key])
        return s.replace('robust','interval').replace('_',' ')
    add(4,'Conditional forecast and representation contributions','Differences are first minus second. Intervals use 2,000 paired circular seven-day block resamples. They describe within-year variability and do not create independent operating years.',pd.DataFrame({'Window h':difference.window_h,'First':difference['first'].map(short),'Second':difference.second.map(short),'Δ value US$':difference.difference_usd,'Lower 95%':difference.lower95,'Upper 95%':difference.upper95}))
    mix=pd.read_csv(O/'heterogeneous_deadline_R2_results.csv')
    add(5,'Mixed deadlines and recourse','The common contract assigns 25%, 50% and 25% of work to 4, 12 and 24 h. The independent earliest-deadline reference has zero late work. Recourse changes only unexecuted reservations.',pd.DataFrame({'Policy':policies(mix),'Recourse':mix.recourse.map({True:'Yes',False:'No'}),'Value US$':mix.value_usd,'Late IT':mix.late_IT_mwh,'CVaR US$':mix.mtm_daily_loss_cvar95_usd,'Closing IT':mix.final_queue_IT_mwh}))
    params=[('Frozen world parameters','59,156,961'),('R2 price-core parameters','248,402'),('Native calibration parameters','24'),('Inactive ranking head','8,321; disabled'),('World width / blocks','256 / 10'),('R2 width / expert width','128 / 64'),('Training prices','2021–2023; 102,144 rows'),('Early stopping','2024 selection; selected epoch 1 for both variants'),('Optimisation','AdamW; lr 0.002; weight decay 0.0001'),('Batch / dropout / patience','4,096 / 0.05 / 12'),('Calibration','1 April–29 June 2024'),('Economic choice','21 August–31 December 2024 UTC'),('2025 changes','No fitting or threshold changes'),('Distribution identity','Paired RT = DA − DART; complementary quantiles after DA'),('Live queue feedback to R2','None; issued reference-context market branch')]
    add(6,'Model and development specification','The two full/no-world training runs share the declared settings. The ablation masks all world-derived channels; its weights are trained separately.',pd.DataFrame(params,columns=['Item','Specification']))
    od=pd.read_csv(O/'w1_predictive_ood_metrics.csv')
    add(7,'Conditional one-step queue transfer diagnostic','Seventeen 2024 origins yield 68 site predictions per policy. Queue errors are in model work units. This conditional stress diagnostic identifies severe direct queue-head transfer error and is not an annual forecast score. The deployed R2 market branch excludes live execution queues.',od[['policy','n','queue_mae','queue_rmse','rmse_ratio_vs_original_actions']].rename(columns={'policy':'Context','n':'Sites','queue_mae':'MAE','queue_rmse':'RMSE','rmse_ratio_vs_original_actions':'RMSE ratio'}))
    add(8,'Physical response and closing obligations','Recovery is positive IT-service deviation from earliest service. Grid displacement is absolute paired grid deviation. Closing work whose deadline lies after the annual endpoint remains a valued obligation, rather than a missed deadline.',pd.DataFrame({'Window h':g.window_h,'Policy':policies(g),'Recovery IT':g.recovery_energy_mwh,'Peak MW':g.peak_recovery_rate_mw,'Grid MWh':g.grid_displacement_mwh,'Closing IT':g.final_queue_IT_mwh}))
    q=fm[fm.gate.eq('pre_DA')&fm.model.isin(['R2_full_native','R2_no_world_native'])]
    add(9,'Pre-publication forecasts','The DA target is unknown in this gate. Coverage and widths describe the joint-construction marginal distributions; identity checks apply to paired samples and consistent means.',pd.DataFrame({'Model':q.model.map(NAMES),'Target':q.target.str.upper(),'Pinball':q.pinball,'RMSE':q.rmse,'90% cover':100*q.coverage90,'90% width':q.width90}))
    q=f[f.window_h.eq(12)&~f.model.eq('Old_R_action')]
    add(10,'Calibration and value-concentration sensitivity at 12 h','Native full R2 is selected on 2024 economics. Raw/external variants and the no-world ablation are retained without choosing a new model on 2025. Ratio denominators use the highest-value feasible primary comparator; an ablation can therefore exceed one.',pd.DataFrame({'Policy':policies(q),'Value US$':q.value_usd,'CVaR US$':q.mtm_daily_loss_cvar95_usd,'VRR':q.VRR,'RF':q.RF,'VCR':q.VCR}))
    previous=(PAPER/'revision_20260920_market_expansion/manuscript/Supplementary_ExecutableFlexibility_2025_EN.md').read_text(encoding='utf-8')
    sections=re.split(r'(?=^### Table S\d+ )',previous,flags=re.M)
    retain=[(2,11,'Historical learned-action context'),(3,12,'RASC service-window extension'),(4,13,'Capacity and physical-boundary refinement'),(7,14,'Declared workload utilisation scenarios'),(8,15,'Transport-constrained migration scenarios'),(9,16,'Bounded active day-ahead positions'),(10,17,'Separate ancillary-service price ceilings'),(11,18,'Activation-recovery sensitivity'),(12,19,'Implementation-cost crossings')]
    for old,new,title in retain:
        section=next(s for s in sections if s.startswith(f'### Table S{old} '));section=section.split('## Supplementary Figures')[0].strip()
        body=section.split('\n',1)[1].strip()
        if old==2:body='These historical seed-20240813 results use the 1 h timeliness clock and are not the contract-matched seed-20240829 neural comparator. Monetary results include closing-work valuation.\n\n'+body
        if old>=7:body='This pathway uses the previously frozen RASC controller and is separate from the R2 energy-shifting total.\n\n'+body
        chunks.append(f'### Supplementary Table {new} | {title}\n\n'+body+'\n')
    notes='''# Supplementary Information

## Executable service commitments unlock the market value of computing flexibility

### Supplementary Note 1 | Feasibility of service commitments

For constant capacity and a common release-relative deadline, earliest-deadline order equals release order. At release t, the remaining prefix through the new cohort is Q(t)+a(t), requiring ceil((Q(t)+a(t))/K) intervals. Earlier cohorts have earlier deadlines; replacing earliest service cannot rescue an infeasible earliest-deadline schedule. Conversely, completing every prefix before its deadline establishes feasibility. Maximising the required interval count yields equation (1). The arrival interval counts as service, so a 6.25 h waiting delay can require a 6.5 h window. The bound applies to the divisible-work model with the specified capacity and nonbinding connection, not to arbitrary non-preemptible heterogeneous machines.

At a recourse decision, all known remaining jobs are already released. Their allowable slot sets are nested prefixes ending at their deadlines. Necessity of equation (5) follows because work due by a deadline cannot use later capacity. For sufficiency, assign jobs in deadline order. Earlier assignments consume no more than their total work inside each later prefix, so its inequality leaves enough room for the next job, regardless of admissible price order. This induction concerns currently known work; unforeseen arrivals are subsequently introduced and tested. Overflow is retained and counted as late rather than erased.

### Supplementary Note 2 | Financial and physical interpretation

Electricity value is incremental to a demand-matched earliest-service reference. Workload demand, completed work and commercial computing revenue are different quantities; this study measures the first two and incremental electricity value. Lateness and electrical shortfall are physical outputs without hypothetical fines. Cash risk and service-adjusted risk are both retained in the source tables. Service-adjusted risk recognises changes in outstanding-work value as they arise. CVaR uses exactly 5% daily probability mass; drawdown uses all quarter-hour increments.

The waterfall is an ordered comparison. The transition from old R to published-DA reservations changes both the executable action rule and the scoring rule. Forecast substitutions then hold that executor fixed. The interval gate selects less response; its negative gross-value increment accompanies reduced intervention and downside. Full versus no-world R2 isolates a separately trained representation contrast, not a guaranteed gain from every world feature. No Shapley or order-independent causal percentages are inferred.

### Supplementary Note 3 | Distinct market pathways

Migration assumes declared network throughput because archived route observations do not establish 2025 route availability. Active DA quantities add a spread exposure to the common physical account. Ancillary-service tables price eligible response separately for each product and zone; they do not establish qualification, clearing or simultaneous delivery. Product ceilings are not stacked or included in the energy-shifting total. Workload-utilisation scenarios hold installed equipment fixed while explicitly changing demand and eligible fractions.

'''
    old_methods=(PAPER/'revision_20260920_formal_bilingual/manuscript/methods_en.md').read_text(encoding='utf-8')
    pathways=old_methods.split('### Additional market pathways')[1].split('### Computational assistance')[0].strip()
    pathways=re.sub(r'<sup>(.*?)</sup>',r'^\1^',pathways).replace('equation (5)','the main energy-shifting account')
    notes+='### Supplementary Note 4 | Extended market-pathway implementation\n\n'+pathways+'\n\n'
    notes+='''### Supplementary Note 5 | Service-feasible value frontier

For an evaluated policy set P, define the frontier at service window w, capacity K and additional implementation cost c as the maximum of V minus c times shifted IT energy among policies with zero deadline-missed work. Earliest service enters with zero incremental value only when it meets the same deadline contract. This is a finite-policy, fixed-execution sensitivity rather than a globally optimal schedule. Constant-arrival scenarios use 40%, 60% and 80% of variable computing capacity. The principal eligible share is 50%; 25% and 100% shares are also evaluated at 60% utilisation. Ineligible constant work reserves its capacity in every interval before flexible work is admitted. Capacity derating applies 75% or 50% of the original compute capacity to both the response and reference.

'''
    (M/'supplement_en.md').write_text(notes+'\n'.join(chunks),encoding='utf-8')
    (M/'supplement_table_sections.json').write_text(json.dumps(chunks,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Supplementary tables',len(chunks),'neural',bool(f.model.eq('Old_R_action').any()))
if __name__=='__main__':main()
