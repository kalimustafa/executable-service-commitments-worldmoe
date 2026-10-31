# S25 Information and net contract value

## S25.1 A common account and three value objects

Fix admissions, physical equipment, site assignments, the reference execution and the original electricity/service account. The cost $C(x,\omega)$ includes energy, the existing import-deviation and curtailment charges, and the declared terminal treatment. The realised path $\omega$ contains prices, physical forcing and other uncertain inputs. It has a probability law $\mathbb P$ and decision-time filtration $\mathcal F=(\mathcal F_t)_t$. The expectation below can be conditional on information available at contract signing. It must use the same law and reference for every contract.

For right $W$, let $\mathcal X_W(\omega)$ be the complete pathwise service-feasible set and $\Pi_W^{\mathcal F}$ the class of policies whose committed actions are adapted to $\mathcal F$ and satisfy those constraints almost surely. Each policy must preserve the original deadline, electrical and terminal definitions. For a selected policy and a realised path, the historical operating saving is

$$S_W^\pi(\omega)=C_0(\omega)-C(\pi,\omega).\qquad\text{(S25.1)}$$

The perfect-information opportunity and the optimal nonanticipative expected value are different objects:

$$\begin{aligned}V_W^{\mathrm{PI}}(\omega)&=\sup_{x\in\mathcal X_W(\omega)}\big(C_0(\omega)-C(x,\omega)\big),\\ V_W^{\mathcal F}&=\sup_{\pi\in\Pi_W^{\mathcal F}}\mathbb E[S_W^\pi].\end{aligned}\qquad\text{(S25.2)}$$

The first optimiser may inspect the entire realised path. The second selects a policy whose actions use only available information. Assume the relevant value functions are measurable, finite and integrable. Every feasible adapted policy produces a pathwise feasible schedule, giving

$$\mathbb E[S_W^\pi]\le V_W^{\mathcal F}\le\mathbb E[V_W^{\mathrm{PI}}].\qquad\text{(S25.3)}$$

The proof first applies $S_W^\pi(\omega)\le V_W^{\mathrm{PI}}(\omega)$ almost surely, takes expectations, and then takes the supremum over admissible policies. It does not interchange a supremum and an expectation. A single historical value of Eq. (S25.1) does not identify either expectation. Re-settlement establishes an account for supplied actions; the actual input vintages establish their information admissibility. A nonanticipative rule does not by itself supply a probability law or an expected-value guarantee.

## S25.2 Information gaps and nested-rights increments

Suppose the expansion $A\subseteq B$ preserves all common inputs and constraints and allows every $A$ policy under $B$. Then both the pathwise perfect-information optimum and the optimal adapted expected value are nondecreasing. Define the information gap

$$I_W=\mathbb E[V_W^{\mathrm{PI}}]-V_W^{\mathcal F}\ge0.\qquad\text{(S25.4)}$$

Subtracting the two definitions gives the exact decomposition

$$V_B^{\mathcal F}-V_A^{\mathcal F}=\mathbb E[V_B^{\mathrm{PI}}-V_A^{\mathrm{PI}}]-(I_B-I_A).\qquad\text{(S25.5)}$$

Although each information gap is nonnegative, their difference need not have a fixed sign. Therefore an expected perfect-information upgrade is not generally an upper or lower bound on the adapted expected upgrade. Table 38 supplies two exact counterexamples. Each has two equiprobable states and an action selected before the state is observed. The actions have abstract payoff pairs; they are not claimed to arise from the paper's deadline-only expansion.

Supplementary Table 38. Abstract nested-action examples separating the two incremental values. Values are illustrative utility units, not market observations. In each row, B retains every A action.

| Example | A action payoffs | Additional B action payoffs | Expected PI increment | Adapted expected increment |
|:---|:---|:---|---:|---:|
| Information becomes more useful | (0, 0) | (1, −1), (−1, 1) | 1 | 0 |
| New action removes information dependence | (2, 0), (0, 2) | (2, 2) | 0 | 1 |

In the first row, every available action has expected payoff zero, but hindsight selects the positive-payoff action in either state. Thus $I_A=0$ and $I_B=1$. In the second row, A's best ex-ante action yields one and its statewise maximum yields two; B's new action yields two in both states. Thus $I_A=1$ and $I_B=0$. These calculations verify Eq. (S25.5) and both possible orderings.

There is a corresponding historical-policy decomposition. For an exactly feasible policy, define $r_W^\pi(\omega)=V_W^{\mathrm{PI}}(\omega)-S_W^\pi(\omega)\ge0$. Then

$$S_B^{\pi_B}-S_A^{\pi_A}=V_B^{\mathrm{PI}}-V_A^{\mathrm{PI}}-(r_B^{\pi_B}-r_A^{\pi_A}).\qquad\text{(S25.6)}$$

A same-controller historical upgrade therefore combines the change in optimal opportunity with the change in that controller's unused opportunity. The paired values in S26 are operating comparisons; they do not independently identify the optimal rights premium.

## S25.3 Incremental bounds under a stated criterion

Let $H_A,H_B$ be exactly feasible savings, including their monetary correction, and let $U_A,U_B$ be valid savings ceilings for the same realised path and account. The corrected incremental interval is

$$\max(0,H_B-U_A)\le V_B^{\mathrm{PI}}-V_A^{\mathrm{PI}}\le U_B-H_A.\qquad\text{(S25.7)}$$

The lower difference follows from $V_B^{\mathrm{PI}}\ge H_B$ and $V_A^{\mathrm{PI}}\le U_A$; nesting adds zero as a lower bound. The opposite pair gives the upper difference. Subtracting two ceilings or two witnesses does not establish this interval. For example, A and B value intervals [50, 100] and [80, 120] permit nested true values (99, 100) or (50, 120). The corresponding increments are 1 and 70, while the difference of witness endpoints is 30 and the difference of ceiling endpoints is 20. The valid general increment interval is [0, 70].

Using $H_B=H_B^{\mathrm{raw}}-\delta_B$ and $H_A=H_A^{\mathrm{raw}}-\delta_A$ makes the lower endpoint $\max(0,H_B^{\mathrm{raw}}-\delta_B-U_A)$ and the upper endpoint $U_B-H_A^{\mathrm{raw}}+\delta_A$. A's repair allowance must therefore be retained when reporting an upper incremental bound. Any additional ceiling allowance enters in its adverse direction. Independent implementations combine through the smaller valid savings ceiling; exceeding both separately requires comparison with the larger ceiling. These statements apply to any original or newly constructed certificate once its domain, objective dominance and verification conditions have been established.

For the ex-ante criterion, a constructive test uses an admissible B policy $\pi_B$ and a validated lower bound $h_B^{\mathcal F}$ on its expected saving. If $u_A^{\mathcal F}$ upper-bounds $V_A^{\mathcal F}$, then

$$V_B^{\mathcal F}-V_A^{\mathcal F}\ge h_B^{\mathcal F}-u_A^{\mathcal F}.\qquad\text{(S25.8)}$$

An integrable pathwise ceiling gives $V_A^{\mathcal F}\le\mathbb E[U_A]$, so a valid upper bound on that expectation can supply $u_A^{\mathcal F}$. The B lower bound must come from an implementable policy under the same law; a hindsight witness cannot replace it. Scenario-specific hindsight choices are not a nonanticipative policy unless their decisions agree whenever the available histories agree. If either endpoint is statistically estimated, the claimed coverage of their difference must hold jointly. The original annual hindsight scalar and one observed policy year do not establish Eq. (S25.8) numerically.

## S25.4 Net operator value and bilateral participation

Let $g$ denote an explicitly identified incremental operating-value criterion: a historical policy contrast, a pathwise PI difference, or the adapted expected difference. Let $F$ be an incremental tenant transfer and $K_o$ the operator's incremental implementation, contracting and verification cost, all relative to retaining A. If these charges are fixed and independent of dispatch, the operator net increment is

$$N_o=g-F-K_o.\qquad\text{(S25.9)}$$

The ex-ante version uses $g=V_B^{\mathcal F}-V_A^{\mathcal F}$; the historical sensitivity in S26 uses the selected policy's realised difference. These substitutions must not be mixed. If the chosen criterion has a valid interval $g\in[g_L,g_U]$ and the additional charges satisfy $F\in[F_L,F_U]$ and $K_o\in[K_L,K_U]$, a safe enclosure is

$$g_L-F_U-K_U\le N_o\le g_U-F_L-K_L.\qquad\text{(S25.10)}$$

A positive lower endpoint proves a positive margin under that criterion, and a negative upper endpoint rules one out. An interval containing zero does not identify strict profitability. Correlation among uncertain costs may make the enclosure loose; simultaneous bounds remain sufficient. Costs already charged in the original operating ledger are not deducted again. Dispatch-dependent compensation must enter the optimisation and the corresponding certificate directly.

For fixed-calendar monetary amounts or common risk-neutral expectations, let $D$ be the tenant's incremental delay/disruption cost and $K_t$ its incremental implementation cost. Its net benefit is $F-D-K_t$. Binary bilateral weak participation requires

$$D+K_t\le F\le g-K_o,\qquad N_{\mathrm{joint}}=g-D-K_o-K_t.\qquad\text{(S25.11)}$$

The transfer cancels from joint surplus. Strict improvement on both sides requires strict inequalities and $g>D+K_o+K_t$. This participation interval does not establish incentive compatibility for a menu of private types. The original IC conditions require their own type-dependent costs and incentive comparisons; no such empirical distribution is inferred here.

Risk adjustments also require a consistent monetary unit. The objective $\mathbb E[L]+\rho\operatorname{CVaR}_\alpha(L)$ changes by $(1+\rho)F$ when a deterministic fee $F$ is added inside the loss, because both components are cash-additive. Deducting only $F$ from that objective would be inconsistent. One cash-normalized alternative is $(1-\gamma)\mathbb E[L]+\gamma\operatorname{CVaR}_\alpha(L)$ with $0\le\gamma\le1$. Another convention places deterministic fees outside an explicitly defined risk premium on operating uncertainty. In either convention, the risk functional and its parameter must be declared before comparing contracts. The difference between two CVaRs does not generally equal the CVaR of their paired loss difference.

# S26 Matched contract increments and fee sensitivity

## S26.1 Retained paired annual evidence

The released file source_data/current/fig3_service_windows.csv contains nine observations: each of the three frozen WorldMoE checkpoints appears under 12, 16 and 24 h rights. Within a seed, the full checkpoint SHA-256 is unchanged across rights. All nine rows use the common reference cost US$555,168.2050556969. The annual saving fields satisfy reference cost minus adjusted cost to within US$4×10⁻¹¹ of their printed decimal representations. The aggregate means independently match source_data/current/0927_adapted_window_contrasts.csv.

The archived controller verifies the selected configuration and checkpoint before the 2025 run, constructs the contract-specific queue and uses the common measurement function. Its year-end dispatch tightening is retained as the implemented heuristic described in S23. The pairing holds the trained model fixed while changing the service right and its resulting queue state. Table 39 presents every paired observation and their arithmetic mean. The values retain the original operating charges and terminal convention; no tenant payment has been added.

Supplementary Table 39. Full historical same-checkpoint contract pairing, annual four-site US dollars. Differences are calculated from unrounded source values before display rounding. The mean and refinement spread describe one calendar.

| Seed | 12 h saving | 16 h saving | 24 h saving | 12→16 increment | 16→24 increment | 12→24 increment |
|:---|---:|---:|---:|---:|---:|---:|
| 20260920 | 72,361.65 | 86,257.34 | 105,565.67 | 13,895.69 | 19,308.33 | 33,204.02 |
| 20260921 | 72,099.98 | 85,833.93 | 105,003.52 | 13,733.95 | 19,169.59 | 32,903.54 |
| 20260922 | 72,201.28 | 86,024.40 | 105,244.00 | 13,823.11 | 19,219.61 | 33,042.72 |
| Mean | 72,220.97 | 86,038.55 | 105,271.06 | 13,817.58 | 19,232.51 | 33,050.09 |

The paired ranges are US$13,733.95–13,895.69, US$19,169.59–19,308.33 and US$32,903.54–33,204.02, respectively. They describe variation across three refinements with a shared pretrained backbone and the same market calendar. They are not confidence intervals over future years. Existing paired intervals for different models at 24 h do not become deadline-upgrade intervals by relabelling them.

All 24 h observations are independently re-settled from the released grid_metrics_inputs_2025.npz, which contains 35,040 intervals at four sites. Reconstructing grid imports and the original energy, curtailment and import-deviation ledger gives mean saving US$105,271.06184163212695 before terminal-cost difference. The largest discrepancy from a published seed saving is US$5.88×10⁻¹²; the largest exported-grid reconstruction difference is below 3.24×10⁻¹⁶ MW. The source reports zero nominal terminal differences. The 12 and 16 h full trajectories, original admissions and complete recovery assignments are not in the inspected release, so those components remain summary-supported. This calculation does not turn their reported numerical service residuals into exact feasible witnesses.

## S26.2 Historical fee sensitivity and break-even budgets

For the fixed historical policy pair, Eq. (S25.9) gives a one-dollar reduction in operator net value for each additional dollar of dispatch-independent compensation or implementation expense. Table 40 applies illustrative annual budgets $q=F+K_o$ to the observed mean paired differences. The grid is an analytical sensitivity choice; the repository contains no observed contract fees or tenant reservation-price sample. In particular, $q=0$ is a stated scenario, not an estimate that additional contracting costs vanish.

Supplementary Table 40. Mean historical net increments under assumed additional annual four-site costs. All values are US dollars. The first column is an analytical input; the remaining columns are paired operating increments minus that input.

| Assumed additional $F+K_o$ | 12→16 h net | 16→24 h net | 12→24 h net |
|---:|---:|---:|---:|
| 0 | 13,817.58 | 19,232.51 | 33,050.09 |
| 5,000 | 8,817.58 | 14,232.51 | 28,050.09 |
| 10,000 | 3,817.58 | 9,232.51 | 23,050.09 |
| 15,000 | −1,182.42 | 4,232.51 | 18,050.09 |
| 20,000 | −6,182.42 | −767.49 | 13,050.09 |
| 30,000 | −16,182.42 | −10,767.49 | 3,050.09 |
| 35,000 | −21,182.42 | −15,767.49 | −1,949.91 |

The exact mean historical break-even budgets from the printed source decimals are US$13,817.584760550363, US$19,232.507466893387 and US$33,050.092227443750. Positive mean net value requires a strictly smaller added budget. Positivity for every one of the three observed refinements instead requires $q$ below US$13,733.95372379268, US$19,169.58506638567 and US$32,903.53879017835. Neither condition guarantees future-year profitability. Extra printed digits document arithmetic and do not claim equivalent physical precision.

The reference arrays imply 2,981.6073925153343935 MWh of annual variable-IT execution. Dividing the three mean budgets by that same denominator gives 4.63427, 6.45038 and 11.08466 US dollars per reference IT MWh. This optional normalization is neither a price per shifted MWh nor a capacity-market rate; the principal result remains the annual four-site budget. No incremental cost already present in the original ledger is deducted again.

## S26.3 Conditional transfer intervals

Table 41 quantifies the operator-side endpoint of Eq. (S25.11) for three illustrative implementation budgets. The tenant-side endpoint remains $D+K_t$ because those costs are unobserved. A positive operator endpoint alone does not establish a mutually acceptable transaction.

Supplementary Table 41. Mean historical upper transfer endpoints $g-K_o$ under assumed operator implementation expenses. The lower endpoint in every cell is the unobserved $D+K_t$. These are conditional retrospective participation intervals, not observed payments or ex-ante procurement prices.

| Assumed $K_o$, US$ | 12→16 h upper $F$, US$ | 16→24 h upper $F$, US$ | 12→24 h upper $F$, US$ |
|---:|---:|---:|---:|
| 0 | 13,817.58 | 19,232.51 | 33,050.09 |
| 5,000 | 8,817.58 | 14,232.51 | 28,050.09 |
| 10,000 | 3,817.58 | 9,232.51 | 23,050.09 |

For example, the mean historical 12→24 h pairing with $K_o=\mathrm{US\$}\,5,000$ gives the interval $D+K_t\le F\le\mathrm{US\$}\,28,050.09222744375$. It exists only if the unknown tenant-side cost does not exceed that endpoint. For an actual ex-ante agreement, replace the historical $g$ with a validated expected or consistently risk-adjusted operating increment and value the tenant's disruption under the same criterion. The source data do not identify those replacements.

## S26.4 Reproduction records and provenance

The supplied economic_value_analysis.py reads the nine annual rows, pairs by seed, checks identical checkpoint hashes and the common reference, recomputes each mean and difference, independently re-settles the available 24 h arrays, and generates the fee and transfer tables. It also evaluates Table 38 with exact rational arithmetic. The outputs are paired_upgrades.csv, upgrade_summary.csv, historical_fee_scenarios.csv, conditional_transfer_frontiers.csv and economic_value_results.json. Published decimal strings are interpreted exactly for descriptive arithmetic; the independent physical re-settlement uses extended numerical precision.

The source commit is 323c2084e7ee668ba512fb1e33dd56b8394b32d4. The NPZ SHA-256 is 5be123c40f01b111cf6503b07882e703a4c2cc29a03dde3cfb7aa4321ce2cf72. Complete hashes for the paired CSV, published aggregate means, checkpoint provenance, original certificate summaries and separate manuscript-reported scalars are recorded in economic_value_results.json. The original same-seed checkpoint hashes are checked against source_data/current/current_provenance.json rather than inferred from display epoch labels.

The original annual witness and second-formulation files remain distinct from these historical account checks. Any independently constructed replacement certificate must record its admissions domain, constraints, terminal account and relationship to the original contract before its bound is used in Eq. (S25.7). The economic definitions and interval proofs apply to a valid replacement without implying recovery of an absent original file. None of the new calculations changes the WorldMoE ablations, the original benchmark table or the workload evidence.
