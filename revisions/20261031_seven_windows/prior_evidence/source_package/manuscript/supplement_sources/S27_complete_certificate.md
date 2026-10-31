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

$$\begin{aligned}L_t:=A_{\max(0,t-D+1)}^-&\le X_t\le A_t^+=:U_t,\\ X_t-X_{t-1}&=x_t,\qquad 0\le x_t\le c_t.\end{aligned}\tag{S27.5}$$

The original physical calculation gives $c_t=1$ exactly throughout these public nominal inputs; this is checked against every static import limit, including idle load. Enlarging admission prefixes to Eq. (S27.5) can only enlarge the feasible execution set. The outer programmes do not impose $X_N=A_N$ or invent a feasible tail. They allow closing work. The source's fixed-reference MTM terminal term is $\tau_z(R_{zN}-X_{zN})$, which differs from true unfinished-work liability $\tau_z(A_{zN}-X_{zN})$ by the small fixed offset $\tau_z(R_{zN}-A_{zN})$. The terminal audit proves $0<\tau_z<21$ dollars per work unit. Since every original feasible schedule has $X_{zN}\le A_{zN}$, the source term is bounded below by $-21E$. We therefore subtract $\delta_T=\sum_z21E=2.02608\times10^{-10}$ dollars from each aggregated slot-cost floor; the signed offset is not silently omitted. The common fixed DA charge $K=2409.7842143056$ dollars is added once after summing site cost bounds and once to the matched reference cost.

The first programme uses incremental service segments and cumulative states. The floating LP is used only to propose equality multipliers. With $\lambda_{N+1}=0$, exact original-cost minimisation and finite-box weak duality give

$$\begin{aligned}L_W^{(1)}&=K-\delta_T+\sum_{z,t}(\phi_{zt}+\psi_{zt}),\\ \phi_{zt}&=\min_{0\le x\le c_{zt}}(f_{zt}(x)-\lambda_{zt}x),\\ \psi_{zt}&=\min((\lambda_{zt}-\lambda_{z,t+1})L_{zt},(\lambda_{zt}-\lambda_{z,t+1})U_{zt}).\end{aligned}\tag{S27.6}$$

To verify the sign, subtract $\lambda_t(x_t+X_{t-1}-X_t)$ from the objective. The service coefficient becomes $f_t(x_t)-\lambda_tx_t$ and the cumulative-state coefficient becomes $\lambda_t-\lambda_{t+1}$. These multiplier terms cancel on every feasible equality. Minimising each remaining term over its original finite box proves the bound for any multiplier values. Every service minimum is evaluated at both physical endpoints, the zero-import breakpoint and the reference-import breakpoint. This proves global minima of the original continuous piecewise-linear cost, including negative-price intervals, without trusting a floating convex hull.

The second programme independently assembles variables $(x_t,X_t,\eta_t)$, the cumulative balance rows and all exact lower-envelope facets $\alpha_{tj}x_t-\eta_t\le-\beta_{tj}$. The finite epigraph box is the minimum and maximum of the original cost over its complete breakpoint set. Consequently every original execution embeds as $\eta_t=f_t(x_t)$. Every facet is checked beneath the original cost at every original breakpoint and endpoint. Affinity between consecutive original breakpoints establishes dominance throughout the interval.

For inequality multipliers $\mu_{tj}\ge0$, the exact finite-box residuals and constant are

$$\begin{aligned}r_t^x&=-\lambda_t+\sum_j\mu_{tj}\alpha_{tj},\qquad r_t^X=\lambda_t-\lambda_{t+1},\\ r_t^\eta&=1-\sum_j\mu_{tj},\qquad k_t=\sum_j\mu_{tj}\beta_{tj}.\end{aligned}\tag{S27.7}$$

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

## Terminal-rate sign and the reference-anchored liability

The outer bound must retain a possible negative constant in the original terminal account. This subsection establishes both the required coefficient sign and that constant's allowance. The evidence is the released daily mark-to-market file `source_data/current/worldmoe_daily_by_seed.csv`, the same three annual policy arrays and the documented per-zone branch of `economic_core.record`, with no migration or fee/grid override. The source aggregate is produced by `summarize_worldmoe_final.py`. The inference conditions on that matched-account lineage, binary64 round-to-nearest arithmetic, shortest round-trip decimal exports and the one intervening default Pandas `precise_xstrtod` parse. It does not infer a historical package version from the CSV or recover the original training-price file.

Reproduce the source's binary64 cash-value array $v_t$ and cumulative work-energy differences $w_{t,z}=\operatorname{fl}(\Delta t\operatorname{cumsum}((b-x)k_z)_{t,z})$. These floating outputs are known constants of the remaining inverse problem, rather than approximations that are silently substituted for an exact cumulative sum. Let $\rho_z=\mathrm{PUE}^{\max}\,p_z^{\mathrm{hist}}$ be the unknown frozen terminal coefficient and $D_t=\sum_z w_{t,z}\rho_z$. The source computes its debt through two multiplications and a four-term reduction, subtracts successive debts from cash value, and sums the result within each local calendar day.

For a day spanning indices $s$ through $e$, let $y_d$ be the binary64 value round-tripped from its final CSV field. Define the exact matrix row and right-hand side by

$$\begin{aligned}M_{dz}&=w_{e,z}-w_{s-1,z},\qquad d_d=\sum_{t=s}^{e}v_t-y_d,\\ |d_d-M_d\rho|&\le a_d+b_d\|\rho\|_\infty.\end{aligned}\tag{S27.11}$$

Here $w_{0,z}=0$ at the initial boundary, with the corresponding zero value used before the first day. Every known binary64 value is converted to its exact rational representation. Put $u=2^{-53}$, $\gamma_n=nu/(1-nu)$, $\Lambda_t=\sum_z|w_{t,z}|$, $V_d=\sum_{t=s}^{e}|v_t|$, $J_d=\sum_{t=s}^{e}(\Lambda_t+\Lambda_{t-1})$ and $m=e-s+1$. The verified error coefficients are

$$a_d=\big(u+\gamma_m(1+u)\big)V_d+\epsilon_{\mathrm{CSV},d}+10^{-100},\tag{S27.12a}$$

$$b_d=\gamma_6(\Lambda_e+\Lambda_{s-1})+\big(u(2+u)(1+\gamma_6)+\gamma_m(1+u)^2(1+\gamma_6)\big)J_d.\tag{S27.12b}$$

To obtain this enclosure, the debt computation contributes at most $\gamma_6\Lambda_t\|\rho\|_\infty$ per interval. Summed exact debt differences telescope, leaving only the two endpoint debt errors. The remaining subtraction errors contribute the $u$ and $u(2+u)$ factors, and the daily reduction contributes $\gamma_m$. The positive $10^{-100}$ reserve covers subnormal effects in these finite nominal calculations; the declared maximum PUE is an order-one positive number.

The intermediate CSV parse is explicitly included. The retained official parser source uses at most 17 digits, at most 34 positive Horner operations, cached decimal powers and at most two divisions. A $\gamma_{40}$ relative allowance covers these operations. Shortest round-trip rendering contributes a conservative $\gamma_2$ relative allowance. Counting the leading zero against the parser's digit limit can additionally discard less than $10^{-16}$ in fixed notation below one. Thus

$$\begin{aligned}\eta&=(1+\gamma_2)(1+\gamma_{40})-1,\qquad \delta=(1+\gamma_{40})10^{-16}+10^{-300},\\ \epsilon_{\mathrm{CSV},d}&=\frac{\eta|y_d|+\delta}{1-\eta}.\end{aligned}\tag{S27.13}$$

Indeed, if $y_d^0$ is the unknown original daily sum, then $|y_d-y_d^0|\le\eta|y_d^0|+\delta$. Applying $|y_d^0|\le|y_d|+|y_d-y_d^0|$ yields the displayed computable bound. The largest selected-row CSV allowance is below US$3.686×10⁻¹¹. Default parsing is therefore not assumed correctly rounded.

Four rows are selected by pivoted QR only as a discovery step. They are seed 20260922 on 25 July, 6 January and 27 June 2025, and seed 20260921 on 18 August 2025. The matrix, right-hand side and inverse are then reconstructed and checked exactly. Let $\widehat\rho=M^{-1}d$, $\kappa=\|M^{-1}\|_\infty$, $a=\max_d a_d$ and $b=\max_d b_d$. Since the exact calculation verifies $\kappa b<1$, every compatible coefficient vector satisfies

$$\begin{aligned}\|\rho\|_\infty&\le T:=\frac{\|\widehat\rho\|_\infty+\kappa a}{1-\kappa b},\\ \|\rho-\widehat\rho\|_\infty&\le\kappa(a+bT).\end{aligned}\tag{S27.14}$$

This self-bound follows from $\rho=\widehat\rho-M^{-1}(d-M\rho)$ and does not assume a prior sign or magnitude for the historical prices. The retained exact inverse infinity norm is approximately 0.748489, $a<1.411\times10^{-10}$ and $b<4.314\times10^{-11}$. The verified radius is less than $1.728\times10^{-9}$. The four coefficient lower endpoints exceed 31.0069471485, 44.8218153888, 44.3299885923 and 50.2319101507. Hence every compatible coefficient is strictly positive. Dividing the central estimates by the published maximum PUE gives 28.37, 41.01, 40.56 and 45.96 US dollars/MWh, which are descriptive centres rather than recovered training records. The complete 1,095-row residual check is also retained; the exact four-row enclosure, rather than that descriptive fit, proves the sign.

Define the dollar coefficient per work unit as $\tau_z=\Delta t\,k_z\rho_z$. The verified upper endpoints are below 12.7055544656, 18.3664007270, 18.1648674345 and 20.5832669449, so the common rational bound $0<\tau_z<21$ is valid. The source terminal account is reference-anchored: it charges $\tau_z(R_{N,z}-X_{N,z})$. Exact original service feasibility gives $X_{N,z}\le A_{N,z}$, and the reference envelope gives $|R_{N,z}-A_{N,z}|\le E$. Consequently

$$\delta_T:=21(4E)=0.000000000202608\ \text{US dollars},\qquad \sum_z\tau_z(R_{N,z}-X_{N,z})\ge-\delta_T.\tag{S27.15}$$

Each outer cost floor is therefore reduced by $\delta_T$ before forming its saving ceiling. For the actual-admission repaired witness, $X_{N,z}=A_{N,z}$, so the true physical backlog is zero while the reference-anchored terminal charge may be signed. Its magnitude is at most $\delta_T$; the witness cost upper bound includes $+\delta_T$. A lower bound on an upgraded contract's opportunity minus a narrower-rights ceiling retains both adverse allowances. The original reference account and deviation-fee anchor remain fixed. The result neither changes those accounts nor treats the tiny constant as negligible merely because it is smaller than a cent.

The executable proof is `terminal_rate_audit.py`; its exact matrix, inverse, error coefficients, parser assumptions, input hashes and rational coefficient enclosures are retained in `terminal_rate_audit.json`, with the full derivation in `terminal_rate_audit.txt` and the official parser provenance files. The terminal and reference-envelope assumptions are shared by both annual certificate formulations.


## Retained materials and reproducibility boundary

The new verification directory contains 12 original-cost multiplier NPZs, 12 independently assembled epigraph multiplier NPZs, complete exact breakpoint/facet files, finite-bound files, four complete rational surrogate schedules and their original-account costs. A separate replay verifies the retained files and hashes without invoking an LP solver. Seven small regression tests check the new formulas, signs, negative-price facets, exact repair, admission non-identification and outward signed monetary rendering; those fixtures are distinct from the full-year certificate replay.

The original 29 October `compute_annual_cut.py`, its 12 original NPZ certificates, its reported US$117,000.2387599464 annual witness and the exact original admission array remain unavailable. The present result avoids assigning invented values to them. It instead proves a conservative replacement from retained source semantics and released annual evidence. All original WorldMoE module-ablation data and figure images remain unchanged. Forecast training, fixed-checkpoint replay and historical first-release-vintage validation remain separate evidence requirements.
