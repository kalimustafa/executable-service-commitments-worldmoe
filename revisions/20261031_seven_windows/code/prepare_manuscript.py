"""Revise the existing manuscript around completed seven-window evidence."""
from pathlib import Path
from fractions import Fraction as F
import json,re,shutil
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT.parent/'EnergyEconomics_Revision_20261031'
OUT=ROOT/'manuscript';OUT.mkdir(exist_ok=True)
source=(OLD/'deliverables/main.md').read_text(encoding='utf-8')
R=json.loads((ROOT/'analysis/results.json').read_text());C=json.loads((ROOT/'certificates/results/seven_window_summary.json').read_text())
M=json.loads((ROOT/'certificates/results/key_economic_comparisons.json').read_text())
s=source
def para(start,new):
    global s
    pattern=re.compile(r'(?m)^'+re.escape(start)+r'[^\n]*')
    matches=list(pattern.finditer(s));assert len(matches)==1,(start,len(matches))
    m=matches[0];s=s[:m.start()]+new+s[m.end():]
def block(start,end,new):
    global s
    a=s.index(start);b=s.index(end,a);s=s[:a]+new+s[b:]

para('Contractual scheduling rights determine',r'''Contractual scheduling rights determine which electricity-saving schedules are available while completing the same admitted computing service. We identify a minimum value of broader rights by comparing an exactly feasible expanded-rights witness with a global narrower-rights ceiling under a fixed financial account. Exact service repair and original-cost dual evaluation turn the comparison into verifiable monetary bounds. For a recovered four-site 2025 New York workload, the 24 h perfect-information opportunity is US\$132,376.82–132,598.65, and its premium over 12 h is US\$34,580.41–34,959.31. Separately, WorldMoE saves US\$105,271.06, with a historical same-checkpoint 12→24 h increase of US\$33,050.09. New frozen-checkpoint evaluations and certificates extend the comparison to seven deadlines from 8 to 48 h. The certified incremental opportunity per extra deadline hour rises from the 24–36 h interval to the 36–48 h interval, rejecting a globally diminishing-return interpretation of the sampled curve. Policy capture, contract-created opportunity and information-adapted prospective value remain distinct. These bounds identify the economic room for improved control or broader rights without interpreting hindsight opportunity as a negotiated price or guaranteed future profit.''')

lit=(ROOT/'literature/critical_review.md').read_text(encoding='utf-8')
review,refs=lit.split('\n\n[12]',1);refs='[12]'+refs
first_review=review.split('\n\n')[0]
s=s.replace('We identify this value by combining',first_review+'\n\nWe identify this value by combining',1)
para('The same construction supplies an upper endpoint',r'''The same construction supplies an upper endpoint for the rights premium and bounds the additional value available to improved control. Its identifying comparison holds service and settlement fixed: a sufficiently valuable expanded-rights witness rules out every narrower-rights controller. This is a different capability from explaining an equilibrium flexibility premium or the surplus effects of procurement channels. The fixed-price scope does not identify market-wide welfare or a negotiated contract price; it supplies a verifiable contract-specific interval for a procurement analysis. The distance from an operating policy to a ceiling includes attainable improvement and relaxation slack. An information-gap identity further explains why even an expected hindsight rights premium need not bound prospective value in either direction. Tenant costs and participation determine which part of the available surplus can support a transaction.''')
para('WorldMoE supplies a concrete operating policy',r'''WorldMoE supplies a concrete operating policy with which to measure captured value. Its price-distribution forecasts, physical import valuation and treatment of outstanding service connect the policy to the model's cost and capacity terms. The rights bounds apply independently of this controller. Recovering the original admissions and complete execution books permits both matched settlement and exact service verification. On that same annual path, the 24 h opportunity is US\$132,376.82–132,598.65, compared with WorldMoE's US\$105,271.06. The seven-window study separates a controller's response to changed deadlines from the certified expansion of the feasible opportunity set. The earlier admission-compatible uniform guarantee is retained as a separate robustness result.''')

# Restore understandable visual explanation at the points where symbols are introduced.
concept=r'''Figure 1 summarises the identifying comparison. The horizontal separation between the expanded-rights feasible floor and the narrower-rights ceiling is a guaranteed opportunity increment; a difference between operating-policy points has no such interpretation without the corresponding endpoints.

![](figures/figure_contract_value_identification.png){width=6.5in}

**Fig. 1. Contract rights and the value objects they identify.** The same admitted work, realised prices, physical limits, reference and terminal account define the nested sets. Panel (a) illustrates the positive exclusion margin $H_B-U_A$ when the expanded-rights witness exceeds the narrower-rights global ceiling. Positions are schematic. Panel (b) distinguishes realised policy saving, perfect-information opportunity and information-adapted expected value; the latter also requires prospective information and tenant-cost evidence before supporting a contract quote.

'''
s=s.replace('# 3 Empirical design and the operating policy',concept+'# 3 Empirical design and the operating policy',1)
s=s.replace('![](figures/figure1_worldmoe_architecture.png){width=10in}','![](figures/figure1_worldmoe_main_readable.png){width=6.5in}')
para('**Fig. 1. WorldMoE architecture',r'''**Fig. 2. WorldMoE from decision inputs to settled computing service.** The revised layout follows the information and decision sequence. Market representation provides price distributions; physical and queue valuation translates them into work-execution and future-capacity costs. The contract executor commits only the current 15 min action, updates outstanding service and settles the common account. The authorised window changes service permissions, while the fitted model remains fixed. Detailed neural layers remain in Supplementary S28. The diagram describes the permitted information pathway; historical first-release evidence has the scope stated in Section 2.2 and S23.''')
s=s.replace('The empirical design requires a concrete policy to measure the contractual opportunity captured in execution.','The empirical design requires a concrete policy to measure the contractual opportunity captured in execution. Figure 2 shows the operating pathway.',1)

methods=r'''## 3.4 Seven-window extension with a fixed neural horizon

The expanded study evaluates 8, 12, 16, 20, 24, 36 and 48 h deadlines on the same recovered admissions and account. The 12/16/24 h policy rows retain their historical origin. Fifteen new annual evaluations comprise 8/20/36/48 h for each of the three selected checkpoints and three 24 h anchors. All three anchors reproduce the archived complete books and settlement trajectories bit for bit. Parameters and preprocessing remain fixed; no new training or fitting is performed.

The neural input and output horizon remains 96 quarter-hour slots. The original lead coordinates and 4/8/12/24 h deadline bins remain unchanged, while total backlog includes all known obligations. The contract queue and deterministic planning calculation extend to 144 or 192 slots for 36 or 48 h. Beyond slot 95, continuation uses the frozen training-period zonal median RT price, zero PV, maximum declared PUE, zero learned opportunity adjustment and unchanged service capacity. The entire queue enters deadline-prefix checks. The median price is a declared continuation assumption, not a conservative price bound or a new 48 h neural forecast. Longer-duration policy results are therefore sensitivities of this specified controller.

All windows retain the archived policy's voluntary early-completion target at year end. Independent service checks use the original exclusive deadlines rather than treating that target as the contract. Raw books are preserved, and separate exact numerical repairs establish release, deadline, capacity, import and closing-service feasibility with explicit monetary allowances. The perfect-information certificates instead permit legitimate closing work with the original terminal valuation; their outer sets do not force year-end completion. S31 gives seven new actual-admission certificates, and S32 gives the policy extension, source versions, anchors and exact service evidence.

'''
s=s.replace('# 4 Results',methods+'# 4 Results',1)

waterfall=r'''Figure 3 makes the recovery account explicit: avoided high-cost imports are a gross benefit, and later imports must be paid before operating value is calculated. This accounting step is also why an apparent reduction during selected expensive intervals is not itself a contract-value estimate.

![](figures/figure_cash_flow_reconciliation.png){width=6.5in}

**Fig. 3. From avoided imports to net operating saving.** The four-site three-checkpoint mean is reconstructed from the recovered 24 h books. Recovery-import cost is deducted from gross avoided-import value, followed by the import-deviation charge and PV-curtailment benefit. The final bar equals the common-account US\$105,271.06 in Table 1; displayed components are rounded after aggregation. Tenant compensation and new implementation costs are outside this operating account.

'''
s=s.replace('## 4.2 Same-checkpoint gains from additional rights',waterfall+'## 4.2 Same-checkpoint gains from additional rights',1)
s=s.replace('**Fig. 2. Historical operating value','**Fig. 4. Historical operating value')

para('**Table 4. Newly verified conservative bounds',r'''**Table 4. Uniform admission-compatible bounds retained as a robustness result.** Annual four-site US dollars. Both implementations use the common outward admission relaxation. Ceilings round upward and floors downward after exact calculation. The statement holds for every admission sequence compatible with the reference. The tighter bounds for the recovered actual admissions are reported separately in Table 5 and S31.''')
para('The 24 h opportunity lies between US\\$130,931.31',r'''These uniform bounds enclose the opportunity of every compatible 24 h admission stream by US\$130,931.31 and US\$134,437.85. They certify at least US\$30,305.84 of additional value over 12 h and US\$15,196.08 over 16 h, with a remaining uniform bound-to-witness gap at most US\$3,506.54. Restoring the actual admissions permits a tighter, separately computed enclosure without changing the underlying workload or realised policy saving. S31 and Table 5 report that actual-admission calculation across seven windows; S27–S29 retain the stronger quantifier scope of the uniform statement.''')

block('## 4.5 Remaining opportunity within the existing 24 h contract','# 5 Economic interpretation',r'''## 4.5 Remaining opportunity under the recovered 24 h contract

For the recovered actual admissions, the new exact original-cost certificate and repaired witness give $H_{24}=\mathrm{US\$}\,132{,}376.82$ and $U_{24}=\mathrm{US\$}\,132{,}598.65$ after outward cent rounding. The unrounded endpoints are used in every subsequent subtraction or ratio. The remaining witness-to-ceiling gap is at most US\$221.82. This refinement follows from recovered input information and new certificates; it is not an increase in the operating policy's realised saving.

Let $\overline S_{24}$ be the exact three-checkpoint mean of the feasible numerical policy witnesses, which preserves US\$105,271.06 at cent precision. On the same service and financial account,

$$
H_{24}-\overline S_{24}\le V_{24}^{PI}-\overline S_{24}\le U_{24}-\overline S_{24},
\qquad
\frac{\overline S_{24}}{U_{24}}\le\frac{\overline S_{24}}{V_{24}^{PI}}\le\frac{\overline S_{24}}{H_{24}}.
\qquad\text{(10d)}
$$

The unrounded certificates give **US\$27,105.76–27,327.58** of remaining annual opportunity and a capture enclosure of **79.39%–79.53%**, with both displays rounded outward. The fraction divides mean policy saving by the common annual optimum; it is not a confidence interval. Figure 5 separates the large remaining operating opportunity from the much smaller uncertainty in the optimum's enclosure.

![](figures/figure_actual_24h_remaining_opportunity.png){width=6.5in}

**Fig. 5. Captured value and remaining opportunity with recovered admissions.** The policy mean and the new actual-admission perfect-information enclosure use identical service obligations, physical inputs and financial accounts. The displayed gap and capture intervals use full-precision certificate and policy values before outward rounding. They replace the wider admission-envelope comparison for this known workload; the uniform guarantee remains in Table 4.

The corresponding optimal 12→24 h rights increment is US\$34,580.41–34,959.31. Its lower endpoint already exceeds the historical policy-pair increment of US\$33,050.09. The difference between these two objects is a change in unexploited opportunity, as Eq. (9) predicts, rather than an inconsistency between estimates of one payoff. Controller improvement and rights procurement therefore require separate comparisons.

## 4.6 Operating response and certified value across seven deadlines

The new certificates use the same recovered actual admissions at every deadline, including 8 h. A lower bound on exact feasible saving and an upper bound on every admissible schedule are computed for each of the 28 site–deadline combinations. Table 5 and Figure 6 compare their annual sums with the matched three-checkpoint operating results. The original uniform admission-envelope proof is not silently extended to 8 h.

__TABLE5__

![](figures/figure_seven_window_contract_value.png){width=6.5in}

**Fig. 6. Seven deadlines separate operating response from contract-created opportunity.** Panel (a) uses actual unequal deadline spacing and distinguishes recovered historical policy points, new frozen-checkpoint evaluations and the replicated 24 h anchor. Policy ranges describe three refinements on one calendar; PI intervals are deterministic certificates. Panel (b) divides adjacent matched-policy and certified opportunity increments by the number of added deadline hours. The unit is annual US dollars per additional hour of contractual discretion, not an electricity price or a continuous derivative. The 36/48 h policy points use the declared continuation rule in Section 3.4.

__POLICY_FINDINGS__

The certified opportunity increases from US\$78,751.60–78,882.41 at 8 h to US\$152,798.28–153,070.23 at 48 h. However, nesting implies nondecreasing optimal value, not diminishing marginal value. For equal twelve-hour extensions, the average incremental opportunity is US\$783.16–823.30 per added deadline hour from 24 to 36 h and US\$878.48–922.81 from 36 to 48 h. The latter lower endpoint exceeds the former upper endpoint, certifying a local increase in the sampled slope. A smooth concave response or a saturation point cannot be imposed on these observations. Procurement must evaluate the particular rights interval and its implementation and tenant costs.

''')
s=s.replace('__TABLE5__',(ROOT/'analysis/main_table5.md').read_text(encoding='utf-8').strip())
rows={r['window_h']:r for r in R['windows']}
pol=r'''For the specified controller, mean annual savings are US\$__P8__, US\$__P20__, US\$__P36__ and US\$__P48__ at the newly evaluated 8, 20, 36 and 48 h deadlines. These points are new evaluations of the original frozen checkpoints, with complete execution and settlement records. The seven-point comparison therefore measures both the economic scope of wider rights and how effectively the declared controller uses that scope.'''
for w in [8,20,36,48]:pol=pol.replace(f'__P{w}__',f"{rows[w]['policy_mean_usd']:,.2f}")
s=s.replace('__POLICY_FINDINGS__',pol)
s=s.replace('The latter lower endpoint exceeds the former upper endpoint, certifying a local increase in the sampled slope.',
    'The latter lower endpoint exceeds the former upper endpoint, certifying a local increase in the sampled slope. Over those same intervals, the policy\'s annual gain per added deadline hour falls from US\\$748.34 to US\\$737.51. Thus the controller\'s response suggests diminishing returns where the certified rights opportunity exhibits a local increase; the operating curve alone would mischaracterise this part of the contract-value curve.')
s=s.replace('The seven-point comparison therefore measures both the economic scope of wider rights and how effectively the declared controller uses that scope.',
    'At 8 h, however, each checkpoint misses original service deadlines: mean late work is approximately 3.01 IT MWh despite zero closing backlog. Its positive cash account is therefore excluded from feasible-value, capture and policy-increment calculations. The independently feasible 8 h hindsight witness establishes that this failure belongs to the evaluated controller on this path, rather than proving that the 8 h contract itself is infeasible. The other windows measure how effectively the stated controller captures their certified opportunity.')
s=s.replace('The certified incremental opportunity per extra deadline hour rises',
    'The 8 h controller incurs substantive service violations despite feasible hindsight schedules. The certified incremental opportunity per extra deadline hour rises',1)
s=s.replace('Raw books are preserved, and separate exact numerical repairs establish release, deadline, capacity, import and closing-service feasibility with explicit monetary allowances.',
    'Raw books are preserved. Only roundoff-scale residuals receive separate exact numerical repairs, with explicit monetary allowances; substantive deadline failures remain reported as failures and are excluded from feasible-value calculations.')
para('On the historical perfect-information criterion,',r'''On the recovered-path perfect-information criterion, the tighter 12→24 h rights interval leaves at least US\$14,580.41 after an assumed US\$20,000 of additional operator payments and costs. The historical policy-pair margin at that assumed cost remains US\$13,050.09. They concern distinct value objects; subtracting a fee does not convert either into an information-adapted expected profit.''')
para('The annual verification encloses 24 h perfect-information value',r'''With recovered actual admissions, the annual verification encloses 24 h perfect-information value by US\$132,376.82 and US\$132,598.65, and bounds its 12→24 h rights premium by US\$34,580.41 and US\$34,959.31. A separate uniform guarantee remains valid across all reference-compatible admissions. The historical policy books and three bitwise-identical 24 h rerun anchors reproduce US\$105,271.06 of mean operating saving. Its capture enclosure is 79.39%–79.53%, leaving US\$27,105.76–27,327.58 inside the existing 24 h contract. Fifteen new policy evaluations and 28 site–deadline certificates extend the study to seven windows. The certified rise in incremental value between the two longest tested intervals shows why neither a uniformly diminishing-return curve nor a preferred contract duration follows from flexibility alone. The economic decision depends on the specific rights increment, the controller and information available to exploit it, and the costs and participation conditions of both parties.''')
para('The public research archive is',r'''The public research archive is https://github.com/kalimustafa/executable-service-commitments-worldmoe. The initial released-array evidence uses commit 323c2084e7ee668ba512fb1e33dd56b8394b32d4; the recovered-admission revision is retained at commit 822aac606a47c6be2639371f90ef2444c8320267. The completed extension is supplied under `revisions/20261031_seven_windows/`, including new source versions and frozen-runtime assets, fifteen new policy outputs, complete service books, exact numerical repairs, twenty-eight actual-admission certificates, figure sources and S31–S32. The previous verification package and original ablation images remain in the earlier revision directory. Hashes and run manifests distinguish recovered historical files, new evaluations and numerical witnesses. No model was retrained or refitted for the extension.''')
s=s.rstrip()+'\n\n'+refs.strip()+'\n'
s=s.replace('figure_cash_flow_reconciliation','figure_24h_cashflow_reconciliation').replace('figure_actual_24h_remaining_opportunity','figure3_remaining_opportunity_actual_admissions')
s=s.replace(r'give $H_{24}=\mathrm{US\$}\,132{,}376.82$ and $U_{24}=\mathrm{US\$}\,132{,}598.65$ after outward cent rounding.',r'give endpoints $H_{24}$ and $U_{24}$ that display as US\$132,376.82 and US\$132,598.65 after outward cent rounding.')

# Reorder numbered references by first appearance, keeping all citations linked.
body,bibliography=s.split('# References',1)
entries={int(m.group(1)):m.group(2).strip() for m in re.finditer(r'(?ms)^\[(\d+)\]\s+(.*?)(?=^\[\d+\]\s+|\Z)',bibliography)}
order=[]
for m in re.finditer(r'\[(\d+)\]',body):
    num=int(m.group(1))
    if num in entries and num not in order:order.append(num)
order += [n for n in entries if n not in order]
mapping={old:new+1 for new,old in enumerate(order)}
body=re.sub(r'\[(\d+)\]',lambda m:'['+str(mapping.get(int(m.group(1)),int(m.group(1))))+']',body)
s=body+'# References\n\n'+'\n\n'.join(f'[{mapping[n]}] {entries[n]}' for n in order)+'\n'
# The user requested short captions for every figure; detailed qualifications
# remain in the nearby methods, results and supplement.
captions={
1:r'**Fig. 1. Contract rights and three value objects.** Nested feasible sets yield the certified margin $H_B-U_A$; positions are schematic.',
2:r'**Fig. 2. WorldMoE decision and settlement pathway.** Forecasts and service state guide execution; completed work updates the queue and common financial account.',
3:r'**Fig. 3. Annual cash-flow reconciliation.** Recovery costs and account charges reduce gross avoided-import value to US\$105,271.06.',
4:r'**Fig. 4. Historical policy value and fee sensitivity.** Bands show three-refinement ranges; additional fees are analytical assumptions.',
5:r'**Fig. 5. Captured and remaining 24 h opportunity.** The same workload and account are used; interval endpoints are rounded outward.',
6:r'**Fig. 6. Seven-window value and incremental gains.** Policy ranges span three refinements; PI bands are certified bounds. Pink crosses mark the 8 h service failure.'}
for n,caption in captions.items():
    s=re.sub(r'(?m)^\*\*Fig\. '+str(n)+r'\.[^\n]*',lambda m:caption,s)
(OUT/'main.md').write_text(s,encoding='utf-8')
(OUT/'reference_number_map.json').write_text(json.dumps(mapping,indent=2),encoding='utf-8')
figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
needed=['figure_contract_value_identification','figure1_worldmoe_main_readable','figure_24h_cashflow_reconciliation','figure2_historical_economic_value','figure3_remaining_opportunity_actual_admissions','figure_seven_window_contract_value']
for name in needed:
    for ext in ['png','svg','pdf']:
        p=ROOT/'figures'/f'{name}.{ext}'
        if not p.exists():p=OLD/'figures'/f'{name}.{ext}'
        assert p.exists(),p
        shutil.copy2(p,figdir/p.name)
print('Prepared complete main source',len(s.split()),'words; references',len(entries))
