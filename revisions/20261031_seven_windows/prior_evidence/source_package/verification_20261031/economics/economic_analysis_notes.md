# Economic-value audit and mathematical support — 31 October 2026

## 1. What the new calculation establishes

The repository contains nine 2025 annual WorldMoE observations: three frozen checkpoints, each evaluated under 12, 16 and 24 h rights. Within each seed the checkpoint hash is identical across rights, and every observation uses the same US$555,168.2050556969 reference account. The same-seed differences are therefore the appropriate descriptive controller comparisons. The comparison retains the trained mapping and changes contractual scheduling inputs and their resulting queue state; it is not a model retraining experiment.

The three mean upgrade values are **US$13,817.584760550363** (12→16 h), **US$19,232.507466893387** (16→24 h), and **US$33,050.092227443750** (12→24 h). The same-seed range for 12→24 h is US$32,903.53879017835–33,204.01976118209. The exact differences of published decimal strings are recorded in `paired_upgrades.csv`; displayed extra digits audit arithmetic and do not assert equivalent empirical precision.

The values are operating-account differences. RT costs, PV curtailment value, the US$1/MWh absolute import-deviation charge and the archived terminal convention already enter the reported saving. Additional tenant compensation, implementation, contracting and verification costs do not have observed values in the release. Thus these results identify break-even budgets for retrospective sensitivity analysis, not actual contract net profits.

All 24 h observations were independently re-settled from the full released arrays. The maximum discrepancy from the published seed summary is US$5.88×10⁻¹². The recomputed mean before terminal-cost difference is US$105,271.06184163212695. The source reports zero nominal terminal differences; admissions and complete recovery records are unavailable to independently validate that statement. The 12/16 h full trajectories are not in the inspected release; their results remain summary-supported. No new training or full contract replay was performed.

## 2. Three estimands and the role of information

Use a common price-taking physical/accounting model, fixed admissions and a common reference. Let Ω carry the uncertain path and let ℱ be the information filtration. A contract W determines the pathwise feasible execution set X_W(ω) and the nonanticipative policy class Π_W^ℱ. All policies must meet the original service/electrical/terminal constraints almost surely. If the work/admissions or information set changes between contracts, the stated nesting argument must be checked again.

For a realised path ω, S_W^π(ω)=C₀(ω)−C(π,ω) is a particular policy's settled saving. It is defined for supplied actions independently of whether historical information admissibility can be verified. Calling π nonanticipative requires the actual receipt-time data proof; an output ledger alone cannot supply it.

The perfect-information value is V_W^PI(ω)=sup_{x∈X_W(ω)}[C₀(ω)−C(x,ω)]. This optimiser may inspect the full future path. A feasible hindsight schedule is a lower bound on this optimum and a cost-relaxation certificate provides an upper bound. Neither is a future-year expectation.

The ex-ante value is V_W^ℱ=sup_{π∈Π_W^ℱ} E[S_W^π]. Expectations may throughout be conditional on information at contract signing. The probability law, common information set and terminal account must be fixed before the comparison. Assume the relevant value functions are measurable, finite and integrable. For every admissible π, S_W^π(ω)≤V_W^PI(ω) almost surely; taking expectations and then the supremum proves V_W^ℱ≤E[V_W^PI]. No interchange of expectation and supremum is asserted.

Information adaptation is necessary for implementing a policy. It does not supply the distribution, repeated independent outcomes, or statistically valid expectation bounds. A retrospective annual policy saving can be a valid observation of an implementable system without being a guaranteed or estimated ex-ante value.

## 3. Why a perfect-information contract premium is not an ex-ante premium

Set I_W=E[V_W^PI]−V_W^ℱ≥0. With nested rights and a common account, V_B^ℱ≥V_A^ℱ and V_B^PI(ω)≥V_A^PI(ω). Algebra gives

ΔV^ℱ=E[ΔV^PI]−(I_B−I_A).

The change in information gap can have either sign. In particular, the marginal perfect-information gain does not generally upper-bound the marginal adapted value, even though each perfect-information level upper-bounds its adapted counterpart.

Two exact equal-probability, two-state decision examples verify both possible directions. Actions must be selected before the state is observed. Payoffs are illustrative utility units, not simulation outputs:

1. A offers payoff (0,0). B additionally offers (1,−1) and (−1,1). The expected PI upgrade is 1, while the best adapted expected upgrade is 0. Here I_A=0 and I_B=1.
2. A offers (2,0) and (0,2). B additionally offers (2,2). The expected PI upgrade is 0, while the best adapted expected upgrade is 1. Here I_A=1 and I_B=0.

The supplied program evaluates these examples exactly using rational arithmetic. They prove the logical distinction without importing a fictitious market distribution.

There is an analogous policy decomposition on a fixed path. For a feasible selected policy define r_W^π=V_W^PI−S_W^π≥0. Then

S_B^{π_B}−S_A^{π_A}=ΔV^PI−(r_B^{π_B}−r_A^{π_A}).

The US$33,050.09 paired controller increment therefore need not equal, lower-bound or upper-bound the unknown optimal rights increment. It measures the operating change of the selected system, combining a change in contractual opportunity with a change in that controller's unused opportunity. This explains why it can exceed the US$19,197.82 lower certificate margin without any contradiction.

## 4. Correct incremental certificate interval

Assume exact feasible witnesses H_A≤V_A^PI and H_B≤V_B^PI, valid upper certificates V_A^PI≤U_A and V_B^PI≤U_B, and nested feasible sets. Then

max(0,H_B−U_A)≤ΔV^PI≤U_B−H_A.

The proof is direct: V_B^PI≥H_B and V_A^PI≤U_A give the first difference lower bound; nesting gives the additional zero bound. The opposite pair gives the upper bound. Independently valid marginal bounds can be combined in this way; merely subtracting U_B−U_A or H_B−H_A gives no such bound.

Example: let A's valid interval be [50,100] and B's [80,120]. Nested true values A=99,B=100 yield upgrade 1, below the difference 30 between the two witnesses. Nested true values A=50,B=120 yield upgrade 70, above the difference 20 between the two ceilings. The correct generic increment interval is [0,70].

Use the tighter of the two valid A ceilings in the combined interval. If the intended statement is that the witness exceeds each independently evaluated ceiling, compare it with the larger ceiling. The latter is a stronger verification claim and yields the smaller numerical margin. Do not call the two independently assembled representations unrelated mathematical relaxations.

With manuscript-reported scalars and the primary fixed seed 20260920 as A's nominal witness, the nominal intervals are:

| Upgrade | Combined nominal PI lower endpoint | Both-methods nominal lower endpoint | Combined nominal PI upper endpoint |
|:---|---:|---:|---:|
| 12→16 h | 0 | 0 | 40,943.5651408533 |
| 16→24 h | 3,695.0279185550 | 3,689.8179185550 | 46,341.3353577700 |
| 12→24 h | 19,203.8179185550 | 19,197.8179185550 | 60,237.0251408533 |

These are **conditional arithmetic on reported values**, not newly certified annual intervals. Replace the B witness by H_B^raw−δ_B and the A witness by H_A^raw−δ_A. The corrected lower endpoint is max(0,H_B^raw−δ_B−U_A), and the upper endpoint is U_B−H_A^raw+δ_A. Any extra ceiling uncertainty must also be added in the adverse direction. The 24 h hindsight array and annual multipliers/facets are still absent, and even the primary policy requires exact service/account verification before it is H_A. The distinct role of A's correction is essential for a valid upper increment bound.

The 12→16 h zero lower endpoint does not show that the optimal gain is zero. It shows that these particular witnesses and ceilings do not certify a strictly positive optimal premium. Likewise a negative net lower endpoint is inconclusive; only a negative valid net upper endpoint excludes positive value.

## 5. Constructive ex-ante comparison

For an admissible fixed B policy π_B, let h_B^ℱ be a proved lower bound on E[S_B^{π_B}]. Let u_A^ℱ upper-bound V_A^ℱ. Then

V_B^ℱ−V_A^ℱ≥h_B^ℱ−u_A^ℱ.

An integrable pathwise ceiling gives V_A^ℱ≤E[U_A], so a proved upper bound on E[U_A] can serve as u_A^ℱ, although it may be loose. The B lower bound must use an implementable policy evaluated under the same probability law; the expectation of a PI witness cannot replace it. For a finite scenario law, nonanticipativity must couple decisions with the same available history. Solving an independent hindsight schedule in each scenario is not a policy witness.

If h and u are estimated, their asserted confidence coverage must be joint: two separate nominal 95% intervals do not automatically provide 95% coverage for their difference. No such distribution or inferential interval is supplied here. Three model-refinement seeds share one 2025 calendar and cannot be treated as three independent economic years. The available block intervals comparing different models at 24 h do not become upgrade intervals for 12/16/24 h.

## 6. Net value, fee budgets and transfers

Keep all monetary terms incremental relative to retaining A. If F is an additional tenant transfer and K_o operator implementation/contracting/verification cost, the operator's ex-ante incremental value is ΔV^ℱ−F−K_o for deterministic, dispatch-independent charges. For a fixed historical policy pair, replace ΔV^ℱ by its observed paired increment to obtain a retrospective cost sensitivity. For a retrospective PI analysis, replace it by ΔV^PI. These substitutions define different estimands and must be labelled throughout.

If g∈[g_L,g_U], F∈[F_L,F_U] and K_o∈[K_L,K_U] under the same criterion, an enclosure is [g_L−F_U−K_U,g_U−F_L−K_L]. Correlation between uncertain costs may make this rectangular enclosure loose; it remains safe when all bounds hold together. A positive lower endpoint ensures a positive margin under that specified criterion. A negative upper endpoint excludes one. Zero-inclusive intervals do not decide strict profitability.

For fixed-calendar monetary amounts or common risk-neutral expectations, an unknown tenant delay/disruption cost D and tenant-side implementation K_t leave the tenant F−D−K_t. The bilateral weak-participation interval is

D+K_t≤F≤g−K_o.

For strict improvement on both sides, use strict inequalities, and the interval is nonempty exactly when g>D+K_o+K_t. The transfer F cancels from joint surplus g−D−K_o−K_t. Calling g−F−K_o social welfare would double-count the distribution of surplus. Participation conditions at a known type do not establish IC for a private-information menu; cross-type incentive constraints must be checked separately.

The historical 12→24 h mean permits the conditional upper transfer endpoint 33,050.09222744375−K_o dollars, or 32,903.53879017835−K_o if positivity is requested in each of the three observed paired refinements. These are neither reservation prices nor future-year guarantees. The lower endpoint D+K_t is unobserved. The same formulas apply to the other two paired upgrade means in `upgrade_summary.csv`.

The delivered fee grid uses assumed F+K_o values 0, 5,000, 10,000, 15,000, 20,000, 30,000 and 35,000 dollars, explicitly labelled analytical scenarios. For example, at US$20,000 the mean historical net increments are −6,182.415239449637, −767.492533106613 and +13,050.092227443750 dollars for 12→16, 16→24 and 12→24 h. At US$35,000, 12→24 h gives −1,949.907772556250 dollars. The existing import-deviation fee is already charged and must not be subtracted again. If a proposed fee varies with delay, execution, or market conditions, include it in the original objective and re-optimise/re-certify; subtracting a constant at the end is then generally wrong.

An optional size normalization divides by 2,981.6073925153343935 MWh of common annual reference variable-IT execution calculated from the NPZ. It gives 4.63427, 6.45038 and 11.08466 dollars per reference IT MWh, respectively. This is a transparent denominator conversion, not a tariff, price per shifted MWh, price per unit of curtailed service, or capacity-market rate. Keeping the main result in four-site annual dollars avoids those ambiguities.

## 7. Risk accounting correction

The current manuscript's E[L]+ρ CVaR_α(L) is monotone and preserves optimal-value ordering under nested policy sets, but is not cash-normalized. Adding deterministic fee F inside L changes that objective by (1+ρ)F, not F. There are three coherent choices:

1. Keep risk adjustment as a separately defined monetary premium on operating uncertainty, with deterministic fees charged outside that premium, and say so.
2. Use the cash-additive convex combination (1−γ)E[L]+γ CVaR_α(L), 0≤γ≤1, and place all fees inside L.
3. Retain E[L]+ρ CVaR_α(L), but correctly scale deterministic additions and avoid interpreting a single unscaled dollar subtraction as the same risk objective.

Also CVaR(L_B−L_A) generally differs from CVaR(L_B)−CVaR(L_A). The archived daily CVaRs cannot identify an incremental annual contract risk premium merely by subtraction. Fees and tenant disruption costs are empirical inputs, not values that can be inferred from the controller's daily risk statistic.

## 8. Concrete edits to the 30 October manuscript

The body indices below count direct document-body elements in the final 30 October DOCX, including tables, and provide anchors rather than permanent Word paragraph numbers.

- Block 8 (abstract): add the three-estimand separation. Report the historical paired US$33,050.09 budget distinctly from the conditional hindsight lower margin. Remove any implication that the paper numerically estimates ex-ante net procurement value.
- Blocks 17 and 19 (contributions): adaptation establishes implementability, while expected valuation additionally needs a law and matched evaluation. Replace the unqualified 'executable net contracting rule' with a conditional criterion and empirical break-even sensitivity.
- Blocks 42–50 (Eqs. 3b–4a): retain level weak duality, explicitly distinguish V^PI(ω) and V^ℱ, and insert the information-gap identity. Avoid suppressing ω without restating the retrospective target in subsequent economic claims.
- Blocks 72–74 (Proposition 1 and implication): keep the pathwise theorem, then state the separate ex-ante test. Replace any suggestion that one observed adapted witness proves a positive expected premium.
- Blocks 138–140 (operating results) and the 12/16/24 results discussion: add the same-checkpoint paired increments and clearly label seed variation. Do not convert those increments into a universal optimal-rights lower bound.
- Blocks 186–191 (Eq. 13, interval and Eq. 14): specify whether g is PI, historical-policy, or expected policy value in each occurrence; add max(0,H_B−U_A); retain A witness correction on the upper endpoint; quantify the retrospective fee budget. Use the appropriate ex-ante lower bound in any procurement objective, not the hindsight H_W.
- Block 192: distinguish a historical operating-value threshold from an actual contract payment, retaining the existing exclusion of unsupported capacity/ancillary receipts.
- Block 197 (risk objective): fix cash normalization or explicitly put deterministic fees outside its risk term.
- Blocks 216–217 (conclusion): conclude with observed paired budgets and unmeasured ex-ante/tenant quantities, without erasing the existing conditional numerical-certificate evidence.

## 9. Reproduction and evidence boundaries

Run:

```sh
python economic_value_analysis.py --repo /path/to/executable-service-commitments-worldmoe --output-dir /path/to/results
```

The input `reported_certificate_scalars.json` records which values come only from the manuscript and flags annual verification as false. All source SHA-256 values are recorded in `economic_value_results.json`. The program checks the complete seed/right pairing, identical checkpoint hashes, common reference, settlement identities, aggregate means and the independently re-settled 24 h values. It also computes exact two-state information-gap examples. These tests do not claim deadline feasibility, first-release replay, missing 12/16 h trajectories, annual PI repair, a new scenario distribution, or observed compensation data.

Relevant source locations:

- `source_data/current/fig3_service_windows.csv`: rows 2–4 (12 h), 5–7 (16 h), 8–10 (24 h); paired checkpoint hash and all annual scalar fields.
- `source_data/current/0927_adapted_window_contrasts.csv`: rows 2–4; aggregate means, with paired model-comparison interval availability not equivalent to deadline-upgrade inference.
- `source_data/current/current_provenance.json`: checkpoint list and `shared_initialization.controller`; three refinements share a pretrained backbone and evaluation calendar.
- `research/ExecutableServiceCommitments_Revision_20260928/engine/scripts/run_service_window.py`: lines 21, 33–44, 67–78 define rights and frozen selected configuration; line 197 uses the year-end tightening heuristic; lines 263–281 dispatch and recover work; lines 283–285 call the common measurement and record the reported information rule.
- `source_data/grid_metrics_inputs_2025.npz`: full 24 h observed-year arrays; no 12/16 h execution, original arrivals or full recovery assignment.
- `source_data/current/extensions/certified_value_bounds.csv`: original Lagrangian scalar summaries; the second-formulation and hindsight values remain separately tagged manuscript reports.

No source repository files, DOCX manuscripts or original ablation results were edited by this task.
