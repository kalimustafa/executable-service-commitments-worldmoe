Contract rights and the economic value of data centre flexibility

Xiang Yi¹,*, Yuyang Chen²,†, Shengye Guan²,†

¹ Northwest Minzu University, Lanzhou, Gansu, China

² School of Management, Lanzhou University, 222 Tianshui South Road, Lanzhou 730000, Gansu, China

† Yuyang Chen and Shengye Guan contributed equally as joint second authors.

* Correspondence: Xiang Yi (mohamwdkali@gmail.com).

Author emails: Yuyang Chen (chenych2024@lzu.edu.cn); Shengye Guan (guanshy2024@lzu.edu.cn).

# Abstract

Contractual scheduling rights determine which electricity-saving schedules are available while completing the same admitted computing service. We identify a minimum value of broader rights by comparing an exactly feasible expanded-rights witness with a global narrower-rights ceiling under a fixed financial account. Exact service repair and original-cost dual evaluation turn the comparison into verifiable monetary bounds. For a recovered four-site 2025 New York workload, the 24 h perfect-information opportunity is US\$132,376.82–132,598.65, and its premium over 12 h is US\$34,580.41–34,959.31. Separately, WorldMoE saves US\$105,271.06, with a historical same-checkpoint 12→24 h increase of US\$33,050.09. New frozen-checkpoint evaluations and certificates extend the comparison to seven deadlines from 8 to 48 h. The 8 h controller incurs substantive service violations despite feasible hindsight schedules. The certified incremental opportunity per extra deadline hour rises from the 24–36 h interval to the 36–48 h interval, rejecting a globally diminishing-return interpretation of the sampled curve. Policy capture, contract-created opportunity and information-adapted prospective value remain distinct. These bounds identify the economic room for improved control or broader rights without interpreting hindsight opportunity as a negotiated price or guaranteed future profit.

**Keywords:** data centre flexibility; contractual scheduling rights; value bounds; nonanticipativity; tenant compensation; electricity settlement.

# 1 Introduction

The economic value of a longer computing deadline is the increase in electricity-cost saving attainable while completing the same admitted service. Its valuation starts with a contract: which task may run in which interval, under which capacity and completion obligations? Extending a deadline creates additional authorised schedules, but their value depends on electricity prices, physical import requirements and competition for future computing capacity. The procurement question is how much an operator can justify paying for that discretion. This complements research on remunerating workload shifts and tenant flexibility [1] [2], while requiring a comparison of entire contractual opportunity sets under one financial account.

Recent economic studies make clear that flexibility value depends on contractual and market institutions. Van de Sande Araujo et al. [3] show how complete bilateral flexibility contracting can recover look-ahead benefits in sequential market clearing. Biswas et al. [4] explain persistent day-ahead premia through disruption risk, limited arbitrage and generators' strategic commitments. Fei et al. [5] study direct and intermediated demand-response procurement, showing how channel competition can reduce double marginalisation. These studies identify mechanisms governing market coordination, implicit remuneration and procurement surplus. Their economic objects differ from the value of relaxing a particular service obligation on a fixed settlement path: an equilibrium premium or procurement gain cannot be inserted directly as that contract's monetary opportunity increment.

We identify this value by combining a feasible schedule under expanded rights with a global saving ceiling under narrower rights. Let $A$ and $B$ denote contracts with nested feasible sets, $U_A$ a valid upper bound on A’s attainable saving, and $H_B$ a fully settled, feasible B schedule. If $H_B>U_A$, at least $H_B-U_A$ of B’s saving is unavailable under every A-feasible schedule. The result covers all narrower-rights controllers, including schedules chosen with hindsight on the recorded calendar. It attributes this minimum increase in attainable value to the change in contractual rights within the stated model. Admissions, equipment, exogenous prices, the reference schedule and the monetary treatment of closing obligations remain fixed.

The same construction supplies an upper endpoint for the rights premium and bounds the additional value available to improved control. Its identifying comparison holds service and settlement fixed: a sufficiently valuable expanded-rights witness rules out every narrower-rights controller. This is a different capability from explaining an equilibrium flexibility premium or the surplus effects of procurement channels. The fixed-price scope does not identify market-wide welfare or a negotiated contract price; it supplies a verifiable contract-specific interval for a procurement analysis. The distance from an operating policy to a ceiling includes attainable improvement and relaxation slack. An information-gap identity further explains why even an expected hindsight rights premium need not bound prospective value in either direction. Tenant costs and participation determine which part of the available surplus can support a transaction.

These monetary comparisons require exact contractual feasibility. A small service residual is not itself a dollar-valued guarantee. We retain each cohort’s exclusive deadline, construct a capacity-compatible repair for homogeneous contracts, and deduct a price-weighted upper bound on its settlement effect. Finite-box dual evaluation accounts for multiplier signs, residuals and monetary rounding. Two formulations of a common convex relaxation provide a way to cross-check the cost lower bound and resulting saving ceiling. Complete constructions and verification records are organised by their evidential scope in the Supplementary Information.

WorldMoE supplies a concrete operating policy with which to measure captured value. Its price-distribution forecasts, physical import valuation and treatment of outstanding service connect the policy to the model's cost and capacity terms. The rights bounds apply independently of this controller. Recovering the original admissions and complete execution books permits both matched settlement and exact service verification. On that same annual path, the 24 h opportunity is US\$132,376.82–132,598.65, compared with WorldMoE's US\$105,271.06. The seven-window study separates a controller's response to changed deadlines from the certified expansion of the feasible opportunity set. The earlier admission-compatible uniform guarantee is retained as a separate robustness result.

# 2 Contract rights and value identification

## 2.1 Authorised service and a common financial account

Time is divided into 15 min slots. Let $a_t$, $x_t$ and $q_t$ denote admitted work, executed work and outstanding work immediately before admission in slot $t$. Conservation requires

$$
q_{t+1}=q_t+a_t-x_t,\qquad 0\le x_t\le q_t+a_t.
\qquad\text{(1a)}
$$

A cohort $j$ arrives at the start of slot $r_j$. Its deadline $d_j$ is the boundary by which execution must finish. For a contract of $W_j$ hours and $\Delta t=0.25$ h, the exact eligible interval is

$$
D_j=W_j/\Delta t=4W_j,\qquad
d_j=r_j+D_j,\qquad
I_j=\{r_j,\ldots,d_j-1\}.
\qquad\text{(1b)}
$$

Thus a 24 h cohort has 96 eligible slots including arrival; slot $r_j+96$ is excluded. Let $v_{jt}$ be cohort-to-slot execution and $\bar x_t$ the service-capacity limit. The contractual constraints are

$$
v_{jt}\ge0,\qquad
\sum_{t\in I_j}v_{jt}=a_j,\qquad
x_t=\sum_{j:t\in I_j}v_{jt}\le\bar x_t.
\qquad\text{(1c)}
$$

Execution outside $I_j$ is zero. Initial cohorts retain their original deadlines, and cohorts crossing the reporting boundary retain their remaining time. Shortening a deadline to year end changes the contract; choosing earlier service is merely one policy within it. Physical completion after year end and the monetary valuation of a closing liability are checked separately.

For homogeneous duration $D$ and an initially empty ledger, define $A_t=\sum_{s\le t}a_s$ and $X_t=\sum_{s\le t}x_s$, with admissions zero outside their source horizon. Earliest-deadline allocation makes cohort feasibility equivalent to

$$
A_{t-D+1}\le X_t\le A_t,\qquad 0\le x_t\le\bar x_t.
\qquad\text{(1d)}
$$

The checks include the full completion horizon. Heterogeneous deadlines retain Eq. (1c): aggregate release and due totals alone can conceal service allocated to the wrong cohort. Supplementary S19 gives the equivalence proof and a mixed-deadline counterexample. Electrical restrictions and nonpreemptive placement, where applicable, add their own constraints to the authorised set.

All contract comparisons use the same immediate-service reference and financial account. The stored reference remains the fixed expense and import-deviation anchor; its floating execution is not redefined as the unknown admissions. The reference and policy hold an identical day-ahead import position, with changes settled at real-time prices [6]. For site $z$ and policy $\pi$,

$$
S_\pi=C_0-C_\pi
=\sum_{z,t}\lambda^{RT}_{z,t}
\big(G_{0,z,t}-G_{\pi,z,t}\big)\Delta t
-\Delta K-\Delta\Psi.
\qquad\text{(2)}
$$

Here $G$ is grid import, $\Delta K$ contains the import-deviation and PV-curtailment charge differences, and $\Delta\Psi$ is the difference in terminal liabilities. The declared annual account prices closing work using frozen zonal median RT prices, maximum specified PUE and the work-to-energy conversion. Actual recovery cash flows and terminal valuation must be reconciled so the same obligation is not charged twice. Common fixed charges cancel in savings but remain in expenditure denominators.

## 2.2 Historical saving, perfect information and prospective value

Fix a probability law $\mathbb P$ at contract signing and a common information rule. Immutable numerical vintage $d_k^{(1)}$ becomes available after receipt $\rho_k$ and processing delay $\delta_k$. Preprocessing and selected checkpoints are frozen at $\theta_0$. A decision-time filtration may be written

$$
\mathcal F_t=\sigma\!\left(
\theta_0,\{d_k^{(1)}:\rho_k+\delta_k\le t\},
\{x_s:s<t\}\right),
\qquad x_t\ \text{is }\mathcal F_t\text{-measurable}.
\qquad\text{(3a)}
$$

The released-work state and other permitted observations enter this filtration through their recorded availability. Later archival revisions do not become historical decision inputs merely because they are available for settlement reconstruction.

Let $\mathcal X_W(\omega)$ contain all schedules satisfying contract $W$ on realised path $\omega$, and let $\Pi_W^{\mathcal F}$ contain policies adapted to the permitted information that satisfy those constraints almost surely. Define

$$
V_W^{PI}(\omega)
=\sup_{x\in\mathcal X_W(\omega)}
\big[C_0(\omega)-C(x,\omega)\big],
\qquad
V_W^{\mathcal F}
=\sup_{\pi\in\Pi_W^{\mathcal F}}\mathbb E[S_W^\pi].
\qquad\text{(3b)}
$$

The first value allows the optimiser to see the whole realised path. The second selects a policy before future observations are revealed. Assume finite, measurable and integrable values. Every adapted feasible policy generates a pathwise feasible schedule, so

$$
\mathbb E[S_W^\pi]\le V_W^{\mathcal F}
\le\mathbb E[V_W^{PI}],
\qquad
\mathcal I_W=\mathbb E[V_W^{PI}]-V_W^{\mathcal F}\ge0.
\qquad\text{(3c)}
$$

For nested rights $A\subseteq B$, both optimal values are nondecreasing under the common account and information rule. Subtracting their information gaps gives the exact identity

$$
V_B^{\mathcal F}-V_A^{\mathcal F}
=\mathbb E[V_B^{PI}-V_A^{PI}]
-(\mathcal I_B-\mathcal I_A).
\qquad\text{(3d)}
$$

Each information gap is nonnegative, but its change can have either sign. Consequently, even an expected perfect-information rights premium need not bound the prospective premium in either direction. More discretion may create opportunities that require better information, or make a useful decision less dependent on future information. Supplementary S25 proves the identity and supplies exact nested-action examples; those examples establish the logical distinction rather than simulate this workload. A single realised operating year identifies neither expectation.

## 2.3 Global ceilings and monetary certificates

On a fixed calendar, a valid lower bound $L_W$ on minimum contract cost yields

$$
S_W^\pi\le V_W^{PI}\le U_W=C_0-L_W.
\qquad\text{(4a)}
$$

The bound must cover every schedule admitted by the original contract, including hindsight schedules. Relaxation is permissible when the original feasible set embeds in the relaxed model and its objective never overstates original cost. This condition includes negative electricity prices and the declared terminal account. A relaxed optimiser need not itself be executable.

Consider a linear relaxation with objective $c_0+c^\top y$, equalities $Ay=b$, inequalities $My\le h$ and finite bounds $\ell\le y\le u$. For free equality multipliers $\lambda$ and nonnegative inequality multipliers $\mu$, define

$$
r=c-A^\top\lambda+M^\top\mu.
\qquad\text{(4b)}
$$

Minimisation of its Lagrangian over the finite box gives

$$
L_W(\lambda,\mu)
=c_0+\lambda^\top b-\mu^\top h
+\sum_i\min\{\ell_i r_i,u_i r_i\}.
\qquad\text{(4c)}
$$

For every original feasible schedule, the equality term vanishes and the inequality term is nonpositive. Each box minimum is no greater than that schedule’s corresponding contribution. These facts prove the cost lower bound by weak duality [7]. The residual term prices stationarity discrepancy using actual variable limits; small residuals alone do not establish a monetary allowance. If multiplier signs are repaired, the full expression is recomputed and evaluated with exact rational arithmetic or certified intervals.

The incremental-segment formulation uses a lower convex cost envelope and queue balances; the cumulative formulation uses supporting facets, epigraph variables and release/deadline prefixes. With the same complete envelope and constraints, these are representations of a common relaxation. Each implementation requires its own assembly and original-cost checks. Verified ceilings combine as

$$
U_W^{comb}
=\min\{U_W^{Lag},U_W^{cut}\}
=C_0-\max\{L_W^{Lag},L_W^{cut}\}.
\qquad\text{(5a)}
$$

If $C_0^-\le C_0\le C_0^+$ encloses the reference account and $J_B^+$ is a repaired feasible cost upper bound, the opposing enclosure directions are

$$U_W=C_0^+-L_W,\qquad H_B=C_0^- -J_B^+.\qquad\text{(5b)}$$

The smaller ceiling gives the tighter combined bound. A claim of exceeding both implementations separately compares against the larger one. Supplementary S21–S22 give original-cost breakpoint evaluation, finite-box verification and the embedding conditions.

## 2.4 Exact feasible witnesses and cross-contract inference

A feasible schedule supplies the opposite endpoint. For homogeneous contracts, let $L_t$ and $U_t$ be exact cumulative due-work and release limits, with initial and terminal conditions explicit. Propagating these limits through slot capacities yields the feasible cumulative corridor $[\underline L_t,\overline U_t]$. If candidate increments $\widehat x_t$ already satisfy the exact nonnegative capacity box and the corridor is feasible, define

$$
\widetilde X_t
=\min\{\overline U_t,\max\{\underline L_t,\widehat X_t\}\},
\qquad
\widetilde x_t=\widetilde X_t-\widetilde X_{t-1}.
\qquad\text{(5c)}
$$

This capacity-compatible cumulative repair preserves releases, deadlines and endpoints. If $\epsilon_r,\epsilon_d$ are exact maximum prefix violations over $N$ slots, then $\sum_t|\widetilde x_t-\widehat x_t|\le N(\epsilon_r+\epsilon_d)$ per site. Supplementary S20 proves the construction. Mixed deadlines require cohort flow, and ramps or nonpreemptive jobs require their full coupled feasibility conditions.

Let $\widehat G,\widehat V$ and $\widetilde G,\widetilde V$ denote candidate and repaired imports and curtailed PV. Under the stated US\$1/MWh deviation and curtailment charges, a repair-cost allowance is

$$
B_R=\sum_{z,t}\Delta t\left[
\big(|\lambda^{RT}_{z,t}|+1\big)
|\widetilde G_{z,t}-\widehat G_{z,t}|
+|\widetilde V_{z,t}-\widehat V_{z,t}|
\right]+|\delta\Psi|.
\qquad\text{(5d)}
$$

The positive-part import and curtailment functions are $k_zPUE_{z,t}$-Lipschitz in work. Their contribution is therefore at most $\sum_{z,t}\Delta t\,k_zPUE_{z,t}(|\lambda^{RT}_{z,t}|+2)|\widetilde x_{z,t}-\widehat x_{z,t}|$. Let $\epsilon_H$ cover combined reference-and-candidate saving evaluation error. The usable witness is

$$
\Delta_B=B_R+\epsilon_H,\qquad
H_B^{cert}=\widehat H_B-\Delta_B
\le C_0-C(\widetilde x)\le V_B^{PI}.
\qquad\text{(5e)}
$$

Below, $H$ always denotes a corrected, exactly feasible endpoint when used in a guaranteed inequality. The two endpoints enclose the contract's opportunity:

$$H_W\le V_W^{PI}\le U_W.\qquad\text{(6a)}$$

A feasible operating policy's headroom satisfies

$$
\max\{0,H_W-S_W^\pi\}
\le V_W^{PI}-S_W^\pi\le U_W-S_W^\pi.
\qquad\text{(6b)}
$$

For nested contracts,

$$
\max\{0,H_B-U_A\}
\le V_B^{PI}-V_A^{PI}
\le U_B-H_A.
\qquad\text{(6c)}
$$

**Proposition 1.** Under the matched service and financial account, $H_B>U_A$ excludes every A-feasible schedule from attaining the B witness’s saving. Indeed, $S_A^\pi\le V_A^{PI}\le U_A<H_B\le V_B^{PI}$, and subtraction proves the positive lower premium. The opposite pair of endpoints gives the upper premium; nesting adds the zero lower bound. Subtracting two ceilings or two policy savings does not produce these bounds.

The identity $U_W-S_W^\pi=(V_W^{PI}-S_W^\pi)+(U_W-V_W^{PI})$ separates attainable improvement from relaxation slack. A conservative ceiling may leave substantial uncertainty without implying poor operation. For a prospective contract, the analogous sufficient test uses a validated lower bound on an adapted B policy’s expected saving and a matched expected A ceiling [8]. A hindsight B witness cannot supply that prospective lower endpoint.

Figure 1 summarises the identifying comparison. The horizontal separation between the expanded-rights feasible floor and the narrower-rights ceiling is a guaranteed opportunity increment; a difference between operating-policy points has no such interpretation without the corresponding endpoints.

![](../figures/figure_contract_value_identification.png){width=6.5in}

**Fig. 1. Contract rights and three value objects.** Nested feasible sets yield the certified margin $H_B-U_A$; positions are schematic.

# 3 Empirical design and the operating policy

## 3.1 A fixed admitted workload and electricity account

The replay covers 35,040 consecutive quarter-hour slots in the 2025 New York settlement calendar at four simulated sites: WEST, CAPITL, N.Y.C. and LONGIL [9]. The common reference executes admitted work as early as electrical feasibility permits. Each site has one work unit of service capacity per interval, corresponding to 0.409765 MWh of variable IT energy, and 0.230700 MW of idle demand. Rated PV is 1 MW, PUE spans 1.023698–1.092948 and the import cap is 2.150223 MW. Physical reconstruction and priors are documented with their NASA POWER inputs [10].

Eagle job submissions and requested-node counts are aggregated into quarter-hour release cohorts [11]. A transformation fitted to the first chronological 60% of source records uses the first and ninety-ninth percentiles, 0 and 2,248 requested nodes; multiplication by $4/2{,}248$ gives aggregate work without capping larger releases. Each site receives one quarter of the stream and retains a separate service ledger. Historical source days from November 2018 to November 2019 are mapped deterministically to the 2025 calendar, with obligations continuous across target-day boundaries. The conserved quantity is admitted divisible work, and site assignments remain fixed. Source job runtimes and CPU efficiencies describe the underlying trace; they are not a replayed utilisation gain or a substitute for deadline completion. Supplementary Table 30 retains the full job-population and CPU-efficiency discussion. This construction compares temporal discretion while holding workload and migration rights fixed.

The 12, 16 and 24 h cases share the reference cost US\$555,168.205055697. The annual account includes RT energy, US\$1/MWh import-deviation and curtailment charges, common DA expenditure and the stated terminal rule. Same-checkpoint comparisons change the deadline and resulting queue state while keeping the trained model fixed. Archived year-end dispatch tightening is retained as a policy heuristic; it does not redefine the original rights set used for a global ceiling.

## 3.2 Why WorldMoE is used

The empirical design requires a concrete policy to measure the contractual opportunity captured in execution. Figure 2 shows the operating pathway. WorldMoE follows the economic structure of this account: forecasts price electricity, physical valuation translates those prices into work-execution costs, and outstanding service limits capacity available to future obligations. Its archived checkpoints were selected using pre-2025 evaluation data. Three refinements share a pretrained parent; within each refinement, the checkpoint hash is identical across the three contract windows. This isolates the historical response of the fitted policy to changed rights.

Any controller producing a verified schedule could supply a feasible-value endpoint. WorldMoE’s role is to provide operating trajectories and a substantive comparator under the common executor; neither the upper bound nor Proposition 1 depends on its architecture. The retained common-executor comparison gives a US\$19,587 annual advantage over the strongest replacement. This is evidence about the whole fitted system. Joint suppression of its opportunity, task-cost and price–PV residual outputs reduces mean saving by only US\$277.99, leaving US\$104,993.07, or 99.74% of the full result. The original ablation therefore does not attribute the whole advantage to those outputs. The original three-panel image, Supplementary Tables 9–10 and the complete comparisons in Tables 28–29 are retained unchanged in the supplementary evidence organised under S28.

The operating rationale can be stated without implementation history. With modelled PUE and the conditional distribution held fixed, the marginal imported-energy cost of work is

$$
g_t'(x)=\Delta t\,k\,PUE_t\,
\mathbb E\!\left[
\lambda_t^{RT}\mathbf 1\{P_t<L_t(x)\}
\mid\mathcal F_t\right],
\qquad L_t(x)=PUE_t(P^{idle}+kx).
\qquad\text{(7)}
$$

This derivative applies where the conditional expectation is differentiable; the original piecewise physical ledger determines settlement. Import exposure depends on whether facility load exceeds PV. Thus the price of a work unit can differ even at identical electricity prices.

Let $\ell(x)$ be direct cost, $U$ available capacity and $F(U-x)$ the continuation cost of other obligations. Within a differentiable active set, write $m_s=\partial\ell/\partial x_s$ and $\kappa_s=-\partial F/\partial U_s'$, where $U'=U-x$. A feasible marginal move from $t$ to $s$ gives

$$
D_{t\to s}\big[\ell(x)+F(U-x)\big]
=(m_s-m_t)+(\kappa_s-\kappa_t).
\qquad\text{(8)}
$$

Delay is locally beneficial when its direct cost reduction exceeds the net continuation cost of the future capacity occupied. This includes the value of capacity released earlier. The policy’s physical and queue information serves these economic quantities. Exact forecast engineering, training and executor details remain supplementary.

![](../figures/figure1_worldmoe_main_readable.png){width=6.5in}

**Fig. 2. WorldMoE decision and settlement pathway.** Forecasts and service state guide execution; completed work updates the queue and common financial account.

## 3.3 Evidence used for each numerical claim

The original admissions and all nine historical 12/16/24 h trajectories have now been recovered from the local research archive. Reapplying the archived reference recurrence reproduces the published annual reference bit for bit; prices, PV, PUE and physical coefficients also match. Each seed retains the same checkpoint hash across deadlines. Supplementary S29 supplies the exact 24 h hindsight witness, and S30 supplies the original policy books, slot-level service checks, settlement reconstruction and exact feasibility repairs. These are recovered historical executions, not new training runs. Neither seed dispersion nor a single-year re-settlement establishes a future-year expectation.

The conservative certificates in Table 4 and the materialised witness share the recovered workload and original financial account. Earlier certificate scalars and their historical verification status remain identified in Supplementary S20–S24; the present numerical conclusions use the complete S27–S30 evidence. Receipt-time numerical vintages separately determine information admissibility, so recovering service and settlement records does not establish a prospective nonanticipative policy.

## 3.4 Seven-window extension with a fixed neural horizon

The expanded study evaluates 8, 12, 16, 20, 24, 36 and 48 h deadlines on the same recovered admissions and account. The 12/16/24 h policy rows retain their historical origin. Fifteen new annual evaluations comprise 8/20/36/48 h for each of the three selected checkpoints and three 24 h anchors. All three anchors reproduce the archived complete books and settlement trajectories bit for bit. Parameters and preprocessing remain fixed; no new training or fitting is performed.

The neural input and output horizon remains 96 quarter-hour slots. The original lead coordinates and 4/8/12/24 h deadline bins remain unchanged, while total backlog includes all known obligations. The contract queue and deterministic planning calculation extend to 144 or 192 slots for 36 or 48 h. Beyond slot 95, continuation uses the frozen training-period zonal median RT price, zero PV, maximum declared PUE, zero learned opportunity adjustment and unchanged service capacity. The entire queue enters deadline-prefix checks. The median price is a declared continuation assumption, not a conservative price bound or a new 48 h neural forecast. Longer-duration policy results are therefore sensitivities of this specified controller.

All windows retain the archived policy's voluntary early-completion target at year end. Independent service checks use the original exclusive deadlines rather than treating that target as the contract. Raw books are preserved. Only roundoff-scale residuals receive separate exact numerical repairs, with explicit monetary allowances; substantive deadline failures remain reported as failures and are excluded from feasible-value calculations. The perfect-information certificates instead permit legitimate closing work with the original terminal valuation; their outer sets do not force year-end completion. S31 gives seven new actual-admission certificates, and S32 gives the policy extension, source versions, anchors and exact service evidence.

# 4 Results

## 4.1 Operating value after recovery

Independent re-settlement of the recovered 24 h execution and physical arrays gives mean observed-year saving US\$105,271.0618416321. The three seed values are US\$105,565.66546172, US\$105,003.51737831 and US\$105,244.00268487. Their mean reproduces US\$105,271.06 and 18.96% of the reference expenditure. The recovered admission and completion books now permit full release, deadline, capacity and closing-service checks. Tiny binary64 service residuals are reported explicitly and removed by exact cumulative-corridor repair before using policy values in feasible-set inequalities; the monetary enclosures in S30 preserve all displayed cent values.

**Table 1. Mean four-site wholesale cash-flow reconciliation for the released 24 h operating trajectories.** Amounts are US dollars before additional tenant compensation or implementation costs. Components are calculated before display rounding.

| Account component | Value |
|:---|---:|
| Gross avoided import value | 172,510.11 |
| Recovery-import energy cost | −64,971.82 |
| Net energy-cost reduction | 107,538.29 |
| Import-deviation charge | −3,348.76 |
| PV-curtailment valuation benefit | 1,081.53 |
| Observed-year operating saving | 105,271.06 |

Recovery absorbs a substantial share of the apparent displacement benefit. The final saving arises from completing the same admitted service under a different import trajectory, with both expensive imports avoided and subsequent imports charged. Relative to the reference, imports fall by 1,013.96 MWh, curtailed PV falls by 1,081.53 MWh and facility electricity demand rises by 67.57 MWh. These quantities reconcile under fixed PV generation; the electricity-saving mechanism does not require lower total facility consumption.

Figure 3 makes the recovery account explicit: avoided high-cost imports are a gross benefit, and later imports must be paid before operating value is calculated. This accounting step is also why an apparent reduction during selected expensive intervals is not itself a contract-value estimate.

![](../figures/figure_24h_cashflow_reconciliation.png){width=6.5in}

**Fig. 3. Annual cash-flow reconciliation.** Recovery costs and account charges reduce gross avoided-import value to US\$105,271.06.

## 4.2 Same-checkpoint gains from additional rights

The nine recovered historical trajectories pair three frozen checkpoints across all three windows. Independently re-settled mean savings are US\$72,220.97, US\$86,038.55 and US\$105,271.06 under 12, 16 and 24 h rights. Computing each within-seed difference before averaging gives Table 2. The original annual summaries are now supported by the archived interval books and recovered admissions, rather than by summary rows alone. Supplementary S30 gives the service, closing-liability and settlement records; S26 retains the original full-precision paired arithmetic.

**Table 2. Historical paired policy results at the same frozen checkpoint.** Independently re-settled annual four-site US dollars from recovered 12/16/24 h trajectories. Ranges describe three refinements on the same calendar; they are not confidence intervals over future years. Exact repaired counterparts and their monetary enclosures are reported separately in Supplementary S30.

| Rights extension | Mean incremental saving | Paired refinement range |
|:---|---:|---:|
| 12 → 16 h | 13,817.58 | 13,733.95–13,895.69 |
| 16 → 24 h | 19,232.51 | 19,169.59–19,308.33 |
| 12 → 24 h | 33,050.09 | 32,903.54–33,204.02 |

Each extension increases the selected policy’s historical saving in all three pairs. The comparison holds the fitted model fixed; it therefore does not attribute the change to retraining a more capable predictor. It also does not by itself identify the optimal rights premium. For an exactly feasible evaluated policy, if $r_W^\pi=V_W^{PI}-S_W^\pi$ is the policy’s unused opportunity, then

$$
S_B^{\pi_B}-S_A^{\pi_A}
=V_B^{PI}-V_A^{PI}-(r_B^{\pi_B}-r_A^{\pi_A}).
\qquad\text{(9)}
$$

The observed upgrade combines a change in attainable value with a change in how effectively the policy captures it. Equation (6c) supplies the separate comparison of optimal opportunity sets. This distinction is consequential when assessing whether to improve control, purchase broader rights, or pursue both.

## 4.3 Historical break-even cost budgets

For the fixed historical policy pair, each extra dollar of dispatch-independent tenant payment or operator implementation cost reduces net value by one dollar. Table 3 subtracts assumed annual budgets $q=F+K_o$ from the unrounded paired means. The source contains no observed fee or tenant reservation-price sample; these are analytical cost sensitivities.

**Table 3. Mean historical net increments after assumed additional annual costs.** Annual four-site US dollars. Existing energy, deviation and curtailment charges remain in the original ledger and are not deducted again.

| Assumed additional $F+K_o$ | 12 → 16 h net | 16 → 24 h net | 12 → 24 h net |
|---:|---:|---:|---:|
| 0 | 13,817.58 | 19,232.51 | 33,050.09 |
| 10,000 | 3,817.58 | 9,232.51 | 23,050.09 |
| 20,000 | −6,182.42 | −767.49 | 13,050.09 |

At the same assumed US\$20,000 additional cost, the full 12→24 h extension retains a positive mean historical margin, while the two smaller extensions have negative margins. This is a comparison of alternative purchases, each charged the specified budget; their net margins are not intended to add. Positive mean historical value requires additional costs strictly below US\$13,817.58476055, US\$19,232.50746689 and US\$33,050.09222744, respectively. Positivity in every observed refinement instead uses the minimum paired increments in Table 2. Neither threshold predicts future-year profitability.



![](../figures/figure2_historical_economic_value.png){width=6.5in}

**Fig. 4. Historical policy value and fee sensitivity.** Bands show three-refinement ranges; additional fees are analytical assumptions.

## 4.4 Bounds on attainable contractual value

The annual certificate applies uniformly over the admission arrays compatible with the released reference. The original admission array has additionally been recovered and matched to that reference, so the uniform theorem and an explicit original-admission witness can now be reported together. Let $b_t$ denote stored reference execution and $R_t$ its exact prefix, $e(t)$ the next unsaturated reference interval and $\mathcal A(b)$ the arrival arrays compatible with the documented reference recurrence and binary64 rounding. Analysis of its 158 busy blocks per site gives the outward envelope

$$
A_t^-:=\max(0,R_t-E)\le A_t\le R_{e(t)}+E=:A_t^+,
\qquad E=2.412\times10^{-12},\quad A_0^-=A_0^+=0.
\qquad\text{(10a)}
$$

Replacing the original prefixes by $A^-_{\max(0,t-D+1)}\le X_t\le A_t^+$ enlarges every compatible contract's feasible set. The source reference remains the fixed expense and import-deviation anchor. Its signed closing term is $\sum_z\tau_z(R_{z,N}-X_{z,N})$. The independent terminal audit proves $0<\tau_z<21$ dollars per work unit; this term is bounded below by $-\delta_T$, where $\delta_T=21\sum_z E_z=2.02608\times10^{-10}$ dollars. The outer cost floors subtract this allowance and retain legitimate closing service obligations rather than forcing completion at year end.

One implementation proposes incremental-service multipliers and then evaluates the original nonconvex slot-cost minimum at every breakpoint. The other independently assembles cost epigraphs and evaluates Eq. (4c). Its 2,095,170 facet–breakpoint comparisons, finite bounds and multiplier signs all pass exact checks. Each method supplies 12 site-contract records; solver-free replay reproduces their saved values. Their annual ceilings differ by less than US\$0.000000138. Validity follows from containment, cost domination and exact weak duality; numerical agreement supplies an additional implementation check.

For 24 h rights, four explicit rational surrogate schedules satisfy the stricter corridor $R_{e(\max(0,t-95))}\le\overline X_t\le R_t$, with $\overline X_N=R_N$. The longest busy block clears within 36 slots, and its clearing-capacity slack exceeds the rounding envelope. Exact earliest service therefore proves nonemptiness of the true 96-slot corridor for every compatible arrival array. Applying Eq. (5c) to that actual corridor gives

$$
\sum_t|\widetilde x_{zt}(a)-\overline x_{zt}|\le2NE_z,
\qquad
\forall a\in\mathcal A(b),\quad
\widetilde x(a)\in\mathcal X_{24}(a).
\qquad\text{(10b)}
$$

All actual work finishes within the year. Original-cost re-settlement, the arrival-repair allowance of US\$0.000836657841429 and the possible signed closing-anchor difference are included conservatively. The resulting uniform value guarantee is

$$
\forall a\in\mathcal A(b),\quad
\exists\widetilde x(a)\in\mathcal X_{24}(a):\quad
C_0-C(\widetilde x(a))\ge H_{24}^{\mathrm{unif}}
\ge130931.31\ \mathrm{US\$}.
\qquad\text{(10c)}
$$

**Proposition 2. Uniform existence and a materialised original-admission witness.** Under the nominal homogeneous contract, the reference recurrence, static capacities and the common account in S27, Eq. (10c) holds for every $a\in\mathcal A(b)$. The proof combines the outward admission envelope, nonempty true cumulative corridors, the exact capacity-compatible repair and the monetary allowance in Eqs. (10a–c); S29 gives the explicit construction and quantifier argument. For the recovered original array $a^\star$, the four saved rational surrogate schedules already satisfy the true corridor exactly, so $\widetilde x(a^\star)=\overline x$ and the repair norm is zero. All work finishes within the year, and the reference-total and admission-total difference is exactly zero at every site. Exact original-account evaluation encloses its saving between US\$130,931.31968120 and US\$130,931.31968149. Thus the original-admission witness is now supplied, while the uniform US\$130,931.31 floor remains valid. The quantifiers describe admission-dependent hindsight schedules; they do not assert one schedule valid for every admission array or an adapted policy.

**Table 4. Uniform admission-compatible bounds retained as a robustness result.** Annual four-site US dollars. Both implementations use the common outward admission relaxation. Ceilings round upward and floors downward after exact calculation. The statement holds for every admission sequence compatible with the reference. The tighter bounds for the recovered actual admissions are reported separately in Table 5 and S31.

| Contract | Original-cost ceiling | Epigraph ceiling | Uniform 24 h value floor | 24 h premium above both ceilings |
|:---|---:|---:|---:|---:|
| 12 h | 100,625.48 | 100,625.48 | 130,931.31 | 30,305.84 |
| 16 h | 115,735.23 | 115,735.23 | 130,931.31 | 15,196.08 |
| 24 h | 134,437.85 | 134,437.85 | 130,931.31 | — |

These uniform bounds enclose the opportunity of every compatible 24 h admission stream by US\$130,931.31 and US\$134,437.85. They certify at least US\$30,305.84 of additional value over 12 h and US\$15,196.08 over 16 h, with a remaining uniform bound-to-witness gap at most US\$3,506.54. Restoring the actual admissions permits a tighter, separately computed enclosure without changing the underlying workload or realised policy saving. S31 and Table 5 report that actual-admission calculation across seven windows; S27–S29 retain the stronger quantifier scope of the uniform statement.

## 4.5 Remaining opportunity under the recovered 24 h contract

For the recovered actual admissions, the new exact original-cost certificate and repaired witness give endpoints $H_{24}$ and $U_{24}$ that display as US\$132,376.82 and US\$132,598.65 after outward cent rounding. The unrounded endpoints are used in every subsequent subtraction or ratio. The remaining witness-to-ceiling gap is at most US\$221.82. This refinement follows from recovered input information and new certificates; it is not an increase in the operating policy's realised saving.

Let $\overline S_{24}$ be the exact three-checkpoint mean of the feasible numerical policy witnesses, which preserves US\$105,271.06 at cent precision. On the same service and financial account,

$$
H_{24}-\overline S_{24}\le V_{24}^{PI}-\overline S_{24}\le U_{24}-\overline S_{24},
\qquad
\frac{\overline S_{24}}{U_{24}}\le\frac{\overline S_{24}}{V_{24}^{PI}}\le\frac{\overline S_{24}}{H_{24}}.
\qquad\text{(10d)}
$$

The unrounded certificates give **US\$27,105.76–27,327.58** of remaining annual opportunity and a capture enclosure of **79.39%–79.53%**, with both displays rounded outward. The fraction divides mean policy saving by the common annual optimum; it is not a confidence interval. Figure 5 separates the large remaining operating opportunity from the much smaller uncertainty in the optimum's enclosure.

![](../figures/figure3_remaining_opportunity_actual_admissions.png){width=6.5in}

**Fig. 5. Captured and remaining 24 h opportunity.** The same workload and account are used; interval endpoints are rounded outward.

The corresponding optimal 12→24 h rights increment is US\$34,580.41–34,959.31. Its lower endpoint already exceeds the historical policy-pair increment of US\$33,050.09. The difference between these two objects is a change in unexploited opportunity, as Eq. (9) predicts, rather than an inconsistency between estimates of one payoff. Controller improvement and rights procurement therefore require separate comparisons.

## 4.6 Operating response and certified value across seven deadlines

The new certificates use the same recovered actual admissions at every deadline, including 8 h. A lower bound on exact feasible saving and an upper bound on every admissible schedule are computed for each of the 28 site–deadline combinations. Table 5 and Figure 6 compare their annual sums with the matched three-checkpoint operating results. The original uniform admission-envelope proof is not silently extended to 8 h.

**Table 5. Policy accounts, service compliance and certified opportunity.** Annual four-site US dollars; three-checkpoint policy means. The 8 h cash account fails service requirements and is excluded from capture calculations. Other policy endpoints use exact feasible numerical counterparts. Historical 12/16/24 h rows and new evaluations are distinguished in Figure 6.

| Deadline | Policy cash saving | Service | Certified PI opportunity | Capture |
|---|---:|:---|---:|---:|
| 8 h | 50,735.34 | Fails (3.01 IT MWh late) | 78,751.60–78,882.41 | — |
| 12 h | 72,220.97 | Feasible | 97,639.33–97,796.41 | 73.84%–73.97% |
| 16 h | 86,038.55 | Feasible | 113,119.89–113,305.19 | 75.93%–76.06% |
| 20 h | 98,199.02 | Feasible | 124,528.17–124,733.99 | 78.72%–78.86% |
| 24 h | 105,271.06 | Feasible | 132,376.82–132,598.65 | 79.39%–79.53% |
| 36 h | 114,251.12 | Feasible | 141,996.61–142,256.41 | 80.31%–80.47% |
| 48 h | 123,101.20 | Feasible | 152,798.28–153,070.23 | 80.42%–80.57% |

![](../figures/figure_seven_window_contract_value.png){width=6.5in}

**Fig. 6. Seven-window value and incremental gains.** Policy ranges span three refinements; PI bands are certified bounds. Pink crosses mark the 8 h service failure.

For the specified controller, mean annual savings are US\$50,735.34, US\$98,199.02, US\$114,251.12 and US\$123,101.20 at the newly evaluated 8, 20, 36 and 48 h deadlines. These points are new evaluations of the original frozen checkpoints, with complete execution and settlement records. At 8 h, however, each checkpoint misses original service deadlines: mean late work is approximately 3.01 IT MWh despite zero closing backlog. Its positive cash account is therefore excluded from feasible-value, capture and policy-increment calculations. The independently feasible 8 h hindsight witness establishes that this failure belongs to the evaluated controller on this path, rather than proving that the 8 h contract itself is infeasible. The other windows measure how effectively the stated controller captures their certified opportunity.

The certified opportunity increases from US\$78,751.60–78,882.41 at 8 h to US\$152,798.28–153,070.23 at 48 h. However, nesting implies nondecreasing optimal value, not diminishing marginal value. For equal twelve-hour extensions, the average incremental opportunity is US\$783.16–823.30 per added deadline hour from 24 to 36 h and US\$878.48–922.81 from 36 to 48 h. The latter lower endpoint exceeds the former upper endpoint, certifying a local increase in the sampled slope. Over those same intervals, the policy's annual gain per added deadline hour falls from US\$748.34 to US\$737.51. Thus the controller's response suggests diminishing returns where the certified rights opportunity exhibits a local increase; the operating curve alone would mischaracterise this part of the contract-value curve. A smooth concave response or a saturation point cannot be imposed on these observations. Procurement must evaluate the particular rights interval and its implementation and tenant costs.

# 5 Economic interpretation

## 5.1 Operator value, tenant participation and joint surplus

Let $g$ denote a clearly identified operating-value increment: a historical policy difference, a realised perfect-information premium, or the adapted expected premium. For dispatch-independent incremental tenant payment $F$ and operator implementation or verification cost $K_o$, the operator’s net increment is

$$
N_o=g-F-K_o.
\qquad\text{(11a)}
$$

The criterion for $g$ must remain fixed when fees are deducted. Table 3 uses a historical policy difference. Prospective procurement instead uses $g=V_B^{\mathcal F}-V_A^{\mathcal F}$ or a rigorously declared risk-adjusted analogue. A hindsight premium does not become prospective revenue by subtracting an assumed fee.

Suppose the selected criterion has valid bounds $g_L\le g\le g_U$, with $F\in[F_L,F_U]$ and $K_o\in[K_L,K_U]$. Then

$$
g_L-F_U-K_U\le N_o\le g_U-F_L-K_L.
\qquad\text{(12)}
$$

A strictly positive lower endpoint gives a sufficient purchasing margin; a nonpositive upper endpoint excludes strictly positive profit under that account and cost range, with break-even still possible at zero. An interval spanning both negative and positive values leaves profitability unresolved. This is a decision consequence of the value bounds, rather than a price estimate inferred from the best observed controller.

Tenant compensation is also income to the tenant. For fixed-calendar monetary amounts or common risk-neutral expectations, let $D$ be incremental delay/disruption disutility and $K_t$ the tenant’s implementation expense. Tenant and joint net benefits are

$$
N_t=F-D-K_t,\qquad
N_{\mathrm{joint}}=N_o+N_t
=g-D-K_o-K_t.
\qquad\text{(11b)}
$$

The delay loss and tenant costs must correspond to the same contract–policy pair and valuation criterion as the operating increment. The transfer cancels from joint surplus. If delay loss depends on the schedule, this accounting identity does not by itself solve the joint scheduling problem. Binary bilateral participation requires

$$
D+K_t\le F\le g-K_o.
\qquad\text{(13)}
$$

Strict improvement on both sides requires strict inequalities. Compensation divides the surplus and satisfies participation; delay and implementation costs determine whether sufficient combined value exists. Participation alone does not establish incentive compatibility for a menu of private types.

On the recovered-path perfect-information criterion, the tighter 12→24 h rights interval leaves at least US\$14,580.41 after an assumed US\$20,000 of additional operator payments and costs. The historical policy-pair margin at that assumed cost remains US\$13,050.09. They concern distinct value objects; subtracting a fee does not convert either into an information-adapted expected profit.

For example, the mean historical 12→24 h contrast with assumed $K_o=\mathrm{US\$}\,5{,}000$ gives a conditional upper transfer endpoint of US\$28,050.09. The lower endpoint remains the unobserved $D+K_t$. This calculation shows exactly which additional tenant information a transaction needs. Risk-adjusted values require a consistent monetary convention before fixed fees are subtracted; Supplementary S25 gives the cash-normalisation conditions. Dispatch-dependent payments must enter the optimisation and its certificate directly.

## 5.2 What time, information and better control each change

Additional time matters because it enlarges the capacity allocations available to complete admitted work. Equation (8) explains why the cheapest electricity slot need not be the cheapest system allocation. Suppose unit cohort A may use slots 1–2 and unit cohort B slots 2–3, with unit capacity and $c_2<c_1<c_3$. Moving A to slot 2 saves $c_1-c_2$ directly but displaces B to slot 3 at cost $c_3-c_2$. Executing A earlier saves $c_3-c_1$ overall. The economic scarcity is a claim on a useful slot, even when total annual computing capacity is ample.

Better control changes how a given feasible set is used. Broader rights change the feasible set itself. Information determines which contingent choices can be implemented. Equations (3d), (6c) and (9) distinguish these channels. A large same-checkpoint upgrade can reflect both a larger opportunity set and better capture within it. A large hindsight premium can coexist with a smaller prospective premium if exploiting the added opportunities requires more information. Conversely, an expanded action set may reduce information dependence. No sign for the change in the information gap is assumed.

These distinctions give the bound a practical role even when it is conservative. If an evaluated policy exceeds a valid narrower-rights ceiling, further improvements to a narrower-rights controller cannot reach that saving. If the intervals overlap, the available evidence leaves that comparison unresolved. The appropriate response depends on which uncertainty remains: a stronger witness, a tighter ceiling, improved control, or better information evidence.

## 5.3 Settlement and service determine applicability

Customer value depends on the charges borne by the relevant account. Repricing the same NYC trajectory under the two specified SC-9 tariff cases yields US\$28,967.61 in energy-charge saving in each case, but US\$41,068.72 and US\$36,304.20 in demand-charge savings [12] [13]. Recovery raises billing peaks in some months despite positive annual demand savings. The two NYC alternatives remain separate one-site accounts, with the full charge scope and exclusions in Supplementary Tables 23–24 and 31. They are not added to the four-site wholesale saving.

Likewise, service rights must match the workload. Eagle’s divisible cohorts allow interrupted execution across eligible slots; nonpreemptive Frontier jobs need contiguous capacity for their rounded runtimes [14]. The archived 24 h and 30 h Frontier tests record 1,764 and 27 overdue jobs, whereas the 48 h test records none while retaining 519 later-due unfinished jobs at year end. Thus a 24 h result for divisible work is not a universal service guarantee. Mixed deadlines, migration rights and different electrical constraints require their corresponding feasible sets and witnesses.

The model values the operator’s temporal discretion at exogenous prices. Capacity, ancillary-service or carbon payments would introduce additional deliverability conditions and settlement terms; any dispatch-dependent objective change requires corresponding bounds. System-wide price effects and welfare require a market model beyond this fixed-price account. These boundaries follow from the economic object being valued, and preserve a direct interpretation of the results within the stated contracts.

# 6 Conclusion

Contractual flexibility has an economic value distinct from the saving achieved by one controller. A corrected feasible schedule and a global ceiling convert that distinction into a monetary interval for additional rights. The information-gap identity explains why prospective value can differ from perfect-information opportunity, while bilateral accounting separates the operator’s payment budget from joint surplus.

With recovered actual admissions, the annual verification encloses 24 h perfect-information value by US\$132,376.82 and US\$132,598.65, and bounds its 12→24 h rights premium by US\$34,580.41 and US\$34,959.31. A separate uniform guarantee remains valid across all reference-compatible admissions. The historical policy books and three bitwise-identical 24 h rerun anchors reproduce US\$105,271.06 of mean operating saving. Its capture enclosure is 79.39%–79.53%, leaving US\$27,105.76–27,327.58 inside the existing 24 h contract. Fifteen new policy evaluations and 28 site–deadline certificates extend the study to seven windows. The certified rise in incremental value between the two longest tested intervals shows why neither a uniformly diminishing-return curve nor a preferred contract duration follows from flexibility alone. The economic decision depends on the specific rights increment, the controller and information available to exploit it, and the costs and participation conditions of both parties.

# Data and code availability

The public research archive is https://github.com/kalimustafa/executable-service-commitments-worldmoe. The initial released-array evidence uses commit 323c2084e7ee668ba512fb1e33dd56b8394b32d4; the recovered-admission revision is retained at commit 822aac606a47c6be2639371f90ef2444c8320267. The completed extension is supplied under `revisions/20261031_seven_windows/`, including new source versions and frozen-runtime assets, fifteen new policy outputs, complete service books, exact numerical repairs, twenty-eight actual-admission certificates, figure sources and S31–S32. The previous verification package and original ablation images remain in the earlier revision directory. Hashes and run manifests distinguish recovered historical files, new evaluations and numerical witnesses. No model was retrained or refitted for the extension.

# Acknowledgements

We thank the Eagle, NYISO, ISO New England and NASA POWER data providers. Oak Ridge National Laboratory supplied Frontier traces, produced under UT-Battelle, LLC Prime Contract DE-AC05-00OR22725 for the US Department of Energy, with creator support from e-INFRA CZ (ID 90254). Source attributions and licences apply.

# Funding

This research received no specific grant from any funding agency in the public, commercial or not-for-profit sectors.

# Author contributions

X.Y. conducted the research, analysed the results and prepared the manuscript. Y.C. and S.G. provided financial advice.

# Competing interests

The authors declare no competing interests.

# Declaration of generative AI and AI assisted technologies

OpenAI ChatGPT and Codex assisted editorial revision, mathematical exposition, verification-code development, document preparation and supplementary-material organisation. Numerical results were checked using the reproducible procedures documented in the Supplementary Information. All numerical graphics derive from the corresponding source calculations or retained original images. The authors retain responsibility for the scientific content, computations and publication claims.

# References

[1] Zhang, W. & Zavala, V. M. Remunerating space–time, load-shifting flexibility from data centers in electricity markets. Applied Energy 326, article 119930 (2022). https://doi.org/10.1016/j.apenergy.2022.119930.

[2] Zhan, Y., Ghamkhari, M., Xu, D., Ren, S. & Mohsenian-Rad, H. Extending demand response to tenants in cloud data centers via non-intrusive workload flexibility pricing. IEEE Transactions on Smart Grid 9(4), 3235–3246 (2018). https://doi.org/10.1109/TSG.2016.2628886.

[3] Van de Sande Araujo, F., Bjørndal, E., Fleten, S.-E., Gabriel, S. A. & Tjeransen, C. F. Don't forget flexibility contracts when redesigning short-term electricity markets. Energy Economics 151, 108980 (2025). https://doi.org/10.1016/j.eneco.2025.108980.

[4] Biswas, S., Çavdar, B., Garcia, A. & Geunes, J. The price of flexibility in electricity markets. Energy Economics 150, 108769 (2025). https://doi.org/10.1016/j.eneco.2025.108769.

[5] Fei, X., Lyu, C. & Ding, Y. Energy flexibility procurement strategy in competitive demand response markets: Dual-channel supply chain perspective. Energy Economics 161, 109542 (2026). https://doi.org/10.1016/j.eneco.2026.109542.

[6] New York Independent System Operator. Accounting and Billing Manual, Section 6.1, Energy Settlement—Load Serving Entities (2024; accessed 29 October 2026). https://www.nyiso.com/documents/20142/2923231/acctbillmnl.pdf.

[7] Boyd, S. & Vandenberghe, L. Convex Optimization (Cambridge University Press, 2004). https://web.stanford.edu/~boyd/cvxbook/.

[8] Brown, D. B., Smith, J. E. & Sun, P. Information relaxations and duality in stochastic dynamic programs. Operations Research 58(4), 785–801 (2010). https://doi.org/10.1287/opre.1090.0796.

[9] New York Independent System Operator. Day-Ahead Zonal LBMP and Time-Weighted Integrated Real-Time Zonal LBMP Archives, 2025 annual data and January 2026 settlement-tail data. NYISO Market Information System (accessed 22 September 2026). https://mis.nyiso.com/public/P-2Alist.htm and https://mis.nyiso.com/public/P-4Alist.htm.

[10] NASA Langley Research Center. Prediction Of Worldwide Energy Resources (POWER) Hourly API Documentation, documentation for the hourly solar and meteorological data service (accessed 20 September 2026). https://power.larc.nasa.gov/docs/services/api/temporal/hourly/.

[11] Clark, S., Selensky, M. & Menear, K. NLR HPC Eagle Jobs Data and Additional Energy Metrics. NLR Data Catalog, National Laboratory of the Rockies (2025; updated 22 April 2026). doi:10.7799/3023273. https://data.nlr.gov/submissions/295.

[12] Consolidated Edison Company of New York. Electric Rates and Tariffs: Frequently Asked Questions, supply charges and Rider M day-ahead hourly pricing (accessed 22 October 2026). https://www.coned.com/en/rates-tariffs/rates/electric-rates-schedule/faqs.

[13] Consolidated Edison Company of New York. Schedule for Electricity Service, P.S.C. No. 10: January 2025 historical tariff and 2025 monthly statements, SC-9 Rate II (accessed 29 October 2026). https://www.coned.com/en/rates-tariffs/rates/historical; https://www.coned.com/en/rates-tariffs/rates/electric-rates-schedule/electric-psc-10.

[14] Maheshwari, K., Klusáček, D., Suter, F. & Brewer, W. Workload traces from the Frontier exascale supercomputer. In Job Scheduling Strategies for Parallel Processing, Lecture Notes in Computer Science 16980 (Springer, 2026). https://jsspp.org/workload/index.php?page=frontier25.
