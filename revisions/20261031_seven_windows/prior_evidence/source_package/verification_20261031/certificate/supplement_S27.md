# S27 Verified conservative certificates from the released annual reference

## Evidence and the quantity being certified

This section reports a new conservative verification using the complete released 2025 arrays. It does not reconstruct the unpublished 29 October certificate files or the previously reported US$117,000.2387599464 hindsight assignment. The retained input is `source_data/grid_metrics_inputs_2025.npz`, SHA-256 `5be123c40f01b111cf6503b07882e703a4c2cc29a03dde3cfb7aa4321ce2cf72`, at public repository commit `323c2084e7ee668ba512fb1e33dd56b8394b32d4`. It supplies 35,040 quarter-hour intervals at four sites, including prices, realised physical inputs, reference execution and three policy executions. The original admissions are absent.

The certificate concerns the original nominal, divisible, homogeneous-rights account with zero initial queue, unit service capacity, static import limits and fixed realised PV/PUE. The archived nominal constraints contain no ramp limit. Dropping additional physical restrictions would preserve an outer savings bound but would not by itself validate a witness for those additional restrictions. The reference provenance premise is the archived `reservation.reference` recurrence: nonnegative finite binary64 arrivals are added to a queue, immediate service is its minimum with unit capacity, and the remainder is carried forward. The nominal cost and terminal rule are those in `economic_core.record`. The separate terminal-rate audit verifies positivity of the historical coefficients from the published MTM ledger and the documented accounting lineage; their positivity and conservative upper bounds are used below.

Reference execution cannot simply be relabelled as admissions. Each site's reference contains 639 capacity-saturated intervals in 158 busy blocks. The longest saturated run is 35 intervals. As an explicit inference counterexample, moving half of the next positive reference amount into a preceding saturated interval yields a distinct nonnegative arrival vector with exactly the same complete reference output when the original function is executed. Neither constructed vector is asserted to be the original admissions. The new method instead bounds every admission vector compatible with the reference recurrence.

The published reference is retained as the source's fixed settlement ledger and absolute-deviation fee anchor. Its expense $C_0$ is a fixed benchmark; the stored floating reference is not promoted to an exactly feasible witness for the unknown admissions. Exact feasibility claims below concern the repaired candidate schedules. Replacing the reference or fee anchor would define a different account and is unnecessary for the stated savings bounds or nested-rights increments.

## A proved admission envelope from floating queue bookkeeping

Let $b_t$ be stored reference service, $R_t=\sum_{s=1}^{t}b_s$ its exact rational prefix, $A_t=\sum_{s=1}^{t}a_s$ the unknown exact admission prefix and $q_t$ the stored reference queue. Set $A_0=R_0=q_0=0$. At an unsaturated interval, $b_t<1$ and the stored queue clears exactly. Define $e(t)$ as the first unsaturated interval at or after $t$, with $e(0)=0$. Every observed final block closes inside the supplied year.

Under round-to-nearest binary64 arithmetic, put $u=2^{-53}$ and $\beta=2u+u^2$. Combining the source addition and subtraction gives

$$q_t=q_{t-1}+a_t-b_t+\epsilon_t,\qquad |\epsilon_t|\le\beta(q_{t-1}+a_t).\tag{S27.1}$$

If two successive reference intervals are unsaturated, the starting queue is zero and the update is exact: $\operatorname{fl}(0+a_t)=a_t$ and $\operatorname{fl}(a_t-a_t)=0$. Thus only saturated blocks and their clearing intervals contribute to the bound. Consider $m$ saturated intervals followed by a clearing interval. Backwards iteration of $q_t\ge(1-\beta)(q_{t-1}+a_t)-b_t$, using $b_t\le1$, bounds its cumulative update error by

$$E_m=\frac{(m+1)(m+2)\beta}{1-(m+2)\beta}.\tag{S27.2}$$

The denominator is checked positive. The bound uses $(1-\beta)^j\ge1-j\beta$ and deliberately permits a larger remaining-service count than necessary. Binary64 addition/subtraction yielding a subnormal result is exact on the common representable-number lattice; the stated finite-input, finite-recurrence premise excludes overflow. Summing Eq. (S27.2) over all 158 blocks and rounding upward to $10^{-15}$ work units gives, independently at every site,

$$E=2.412\times10^{-12},\qquad |A_t-R_t-q_t|\le E.\tag{S27.3}$$

Since $q_t\ge0$, and $q_{e(t)}=0$ while admissions are nondecreasing, the exact original admissions satisfy

$$A_t^-:=\max(0,R_t-E)\le A_t\le R_{e(t)}+E=:A_t^+,\qquad A_0^-=A_0^+=0.\tag{S27.4}$$

This is a source-derived outward envelope. Its error is independent of the printed residuals associated with the missing old hindsight witness.

## Two independently assembled annual outer formulations

For $D=4W$ eligible slots, every original feasible cumulative service path $X_t$ satisfies

$$L_t:=A_{\max(0,t-D+1)}^-\le X_t\le A_t^+=:U_t,\qquad X_t-X_{t-1}=x_t,\quad 0\le x_t\le c_t.\tag{S27.5}$$

The original physical calculation gives $c_t=1$ exactly throughout these public nominal inputs; this is checked against every static import limit, including idle load. Enlarging admission prefixes to Eq. (S27.5) can only enlarge the feasible execution set. The outer programmes do not impose $X_N=A_N$ or invent a feasible tail. They allow closing work. The source's fixed-reference MTM terminal term is $\tau_z(R_{zN}-X_{zN})$, which differs from true unfinished-work liability $\tau_z(A_{zN}-X_{zN})$ by the small fixed offset $\tau_z(R_{zN}-A_{zN})$. The terminal audit proves $0<\tau_z<21$ dollars per work unit. Since every original feasible schedule has $X_{zN}\le A_{zN}$, the source term is bounded below by $-21E$. We therefore subtract $\delta_T=\sum_z21E=2.02608\times10^{-10}$ dollars from each aggregated slot-cost floor; the signed offset is not silently omitted. The common fixed DA charge $K=2409.7842143056$ dollars is added once after summing site cost bounds and once to the matched reference cost.

The first programme uses incremental service segments and cumulative states. The floating LP is used only to propose equality multipliers. With $\lambda_{N+1}=0$, exact original-cost minimisation and finite-box weak duality give

$$L_W^{(1)}=K-\delta_T+\sum_{z,t}\left(\min_{0\le x\le c_{zt}}[f_{zt}(x)-\lambda_{zt}x]+\min\left[(\lambda_{zt}-\lambda_{z,t+1})L_{zt},(\lambda_{zt}-\lambda_{z,t+1})U_{zt}\right]\right).\tag{S27.6}$$

To verify the sign, subtract $\lambda_t(x_t+X_{t-1}-X_t)$ from the objective. The service coefficient becomes $f_t(x_t)-\lambda_tx_t$ and the cumulative-state coefficient becomes $\lambda_t-\lambda_{t+1}$. These multiplier terms cancel on every feasible equality. Minimising each remaining term over its original finite box proves the bound for any multiplier values. Every service minimum is evaluated at both physical endpoints, the zero-import breakpoint and the reference-import breakpoint. This proves global minima of the original continuous piecewise-linear cost, including negative-price intervals, without trusting a floating convex hull.

The second programme independently assembles variables $(x_t,X_t,\eta_t)$, the cumulative balance rows and all exact lower-envelope facets $\alpha_{tj}x_t-\eta_t\le-\beta_{tj}$. The finite epigraph box is the minimum and maximum of the original cost over its complete breakpoint set. Consequently every original execution embeds as $\eta_t=f_t(x_t)$. Every facet is checked beneath the original cost at every original breakpoint and endpoint. Affinity between consecutive original breakpoints establishes dominance throughout the interval.

For inequality multipliers $\mu_{tj}\ge0$, the exact finite-box residuals and constant are

$$r_t^x=-\lambda_t+\sum_j\mu_{tj}\alpha_{tj},\quad r_t^X=\lambda_t-\lambda_{t+1},\quad r_t^\eta=1-\sum_j\mu_{tj},\quad k_t=\sum_j\mu_{tj}\beta_{tj}.\tag{S27.7}$$

The cost floor sums $k_t$ and the minimum of each residual times both corresponding finite endpoints, adds $K$ once and subtracts the same $\delta_T$. This is Eq. (S21.5) applied to the independently assembled epigraph model with the proved terminal-cost allowance. HiGHS' nonpositive sensitivity for a minimisation problem's upper inequality row is negated to obtain this $\mu$ convention. Any negative $\mu$ is clipped to zero and the entire expression is recomputed. No clipping was needed in the retained annual records. All 2,095,170 retained facet–breakpoint comparisons pass exactly. Nonzero stationarity or primal residuals remain diagnostics; the full finite-box arithmetic supplies the rigorous bound.

These two programmes represent the same complete convex outer relaxation. Their independent matrices and exact verifiers check implementation agreement. They are not presented as two different economic mechanisms or two unrelated mathematical relaxations.

Each stored input and multiplier is interpreted as an exact binary rational. Complete scalar lower-bound terms are rounded downward at $10^{-12}$ dollars before integer accumulation. The first method uses two terms per interval, giving a loss below $2N\times10^{-12}$ dollars per site; the second uses four terms per interval, giving a loss below $4N\times10^{-12}$. The reference expense is enclosed from below and above using the same complete-term rule. Thus all monetary differences use the correct opposing enclosure directions.

## A constructive feasible-value guarantee without recovering the unknown arrivals

For 24-hour rights, construct an exact surrogate schedule in the stricter corridor

$$R_{e(\max(0,t-D+1))}\le\overline X_t\le R_t,\qquad \overline X_N=R_N,\quad D=96.\tag{S27.8}$$

A separate LP proposes service increments. They are explicitly clipped into their physical boxes and then repaired by the exact propagated-corridor construction of S20. The resulting four complete rational schedules pass all 35,040 service boxes, all corridor prefixes and the closing equality, and are re-settled using the original nonconvex slot costs. The public reference itself establishes nonemptiness of Eq. (S27.8), because its longest busy block clears within 36 slots.

For clarity, the exact original admission-dependent corridor is also nonempty. Let $Q_t=\max(Q_{t-1}+a_t-1,0)$ be the exact earliest-service queue on the actual admissions. Its update map is 1-Lipschitz. Comparing it with the floating recurrence and summing Eq. (S27.1) gives $|Q_t-q_t|\le E$. At every stored clearing interval $c$, the exact pre-service queue is at most $b_c+E$. The smallest recorded clearing-capacity margin is $1-b_c=0.04137009475380182\ldots$, strictly exceeding $E$. Hence the exact queue clears at the same interval. Every actual cohort completes within a block of at most 36 intervals, satisfying all displayed 48/64/96-slot contracts. The final source interval is unsaturated, so exact earliest service completes the actual total $A_N$ within the year.

For any actual admission vector compatible with the source reference, Eq. (S27.4) implies that the surrogate schedule violates each true release/deadline prefix by at most $E$, including its final endpoint. Apply the exact S20 repair to that actual corridor, imposing $\widetilde X_N=A_N$. It constructs a feasible original schedule and proves

$$\sum_{t=1}^{N}|\widetilde x_t-\overline x_t|\le2NE.\tag{S27.9}$$

The repaired schedule completes all actual work inside the year, so its true terminal backlog is zero. Its source MTM terminal term is nevertheless $\tau_z(R_{zN}-A_{zN})$, a possible signed bookkeeping offset bounded in magnitude by $21E$. The surrogate schedule ends at $R_N$ and has zero source MTM terminal term. For each site, let $M_z=\max_t\Delta t\,k_z\mathrm{PUE}_{zt}(|p_{zt}^{\mathrm{RT}}|+2)$. The original import, curtailment, absolute-deviation and source-terminal account therefore gives

$$J(\widetilde x)-J(\overline x)\le\sum_z M_z(2NE)+\delta_T<0.000837\ \text{US dollars}.\tag{S27.10}$$

The retained exact total, including the signed-terminal allowance, is less than $0.000836658044037$ dollars. This calculation uses proved reference-bookkeeping bounds, not unverified residuals of the old witness. It yields a uniform constructive existence and value guarantee for every compatible actual admission vector. The four explicit surrogate schedules are supplied. The admission-specific repaired assignment cannot be materialised without the original admissions and is not labelled as recovered. No first-release or information-adapted policy performance is inferred from this hindsight construction.

Supplementary Table 42. New conservative annual results on the public-reference admission envelope. Savings ceilings are rounded upward and the feasible-value floor downward. The common matched account and terminal sign proof apply to both formulations. Premiums use exact unrounded enclosures before cent rounding.

| Right | Original-cost ceiling US$ | Independent epigraph ceiling US$ | Uniform 24 h value floor US$ | 24 h premium above both ceilings US$ |
|:---|---:|---:|---:|---:|
| 12 h | 100,625.48 | 100,625.48 | 130,931.31 | 30,305.84 |
| 16 h | 115,735.23 | 115,735.23 | 130,931.31 | 15,196.08 |
| 24 h | 134,437.85 | 134,437.85 | 130,931.31 | Not an exclusion comparison |

The largest unrounded difference between the two formulations' annual ceilings is below $0.000000138$ dollars. The new 24-hour perfect-information opportunity is enclosed by US$130,931.31 and US$134,437.85 on this conservative verification. The remaining bound-to-witness-value gap is at most US$3,506.54. This gap is an upper bound on remaining optimisation opportunity, not an assertion of the exact unknown optimum.

## Retained materials and reproducibility boundary

The new verification directory contains 12 original-cost multiplier NPZs, 12 independently assembled epigraph multiplier NPZs, complete exact breakpoint/facet files, finite-bound files, four complete rational surrogate schedules and their original-account costs. A separate replay verifies the retained files and hashes without invoking an LP solver. Seven small regression tests check the new formulas, signs, negative-price facets, exact repair, admission non-identification and outward signed monetary rendering; those fixtures are distinct from the full-year certificate replay.

The original 29 October `compute_annual_cut.py`, its 12 original NPZ certificates, its reported US$117,000.2387599464 annual witness and the exact original admission array remain unavailable. The present result avoids assigning invented values to them. It instead proves a conservative replacement from retained source semantics and released annual evidence. All original WorldMoE module-ablation data and figure images remain unchanged. Forecast training, fixed-checkpoint replay and historical first-release-vintage validation remain separate evidence requirements.
