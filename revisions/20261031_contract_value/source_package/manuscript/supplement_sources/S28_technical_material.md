# S28 Operating equations and retained main-paper figures

## S28.1 WorldMoE's role in the economic comparison

WorldMoE supplies the recorded operating trajectories under a common executor. Its W0/W1 market backbone and APIC price representation feed R2's physical and queue-dependent valuation; the allocator reserves capacity across obligations and the executor commits one current action. An exactly feasible, fully settled schedule from any method can supply a lower endpoint for perfect-information opportunity. The independent opportunity bound does not depend on this architecture. Tables 16–22 and the retained model-method sections give the full dimensions, losses, chronology and public-inference specification. This section retains the equations moved from the main paper and identifies their economic function.

## S28.2 Price identity and physical marginal cost

Before DA publication, the price representation forms joint DA/spread paths; after publication, the released DA value anchors complementary RT quantiles:

$$\begin{aligned}A^{(j)}&=Q_A(U_j),\quad \Delta p^{(j)}=Q_{\Delta p}(V_j),\quad r^{(j)}=A^{(j)}-\Delta p^{(j)},\\ q_{\mathrm{RT}}(\tau\mid A)&=A-q_{\mathrm{DART}}(1-\tau\mid A).\end{aligned}\qquad\text{(S28.1)}$$

This construction preserves the stated price identity. Workload states affect service valuation rather than overwriting the exogenous price-scoring targets. The identity is an algebraic property; receipt-time input availability is verified separately.

For service $x$, facility load is $L=\mathrm{PUE}(P^{\mathrm{idle}}+kx)$. With clipped-Gaussian PV $P=\max(0,\mu+\sigma Z)$, $\sigma>0$, define $z_L=(L-\mu)/\sigma$ and $z_0=-\mu/\sigma$. The standard-normal CDF and density are $\Phi$ and $\phi$. The first partial moment and expected imports are

$$\begin{aligned}m_1(L)&=\mu\big(\Phi(z_L)-\Phi(z_0)\big)+\sigma\big(\phi(z_0)-\phi(z_L)\big),\\ \mathbb E[(L-P)_+]&=L\Phi(z_L)-m_1(L).\end{aligned}\qquad\text{(S28.2)}$$

Under the conditional-price approximation $\mathbb E[r\mid P]=\alpha+\beta P$, with PUE and the conditional distribution fixed, the marginal imported-energy surrogate cost is

$$g'(x)=\Delta t\,k\,\mathrm{PUE}\big(\alpha\Phi(z_L)+\beta m_1(L)\big)=\Delta t\,k\,\mathrm{PUE}\,\mathbb E[r\,\mathbf 1_{\{P<L\}}].\qquad\text{(S28.3)}$$

The derivative follows by differentiating the positive-part import cost away from its probability-zero continuous kink; the zero-PV atom is included in the import region for positive load. This formula describes imported-energy valuation under the declared approximation. The original curtailment, import-deviation and terminal charges remain in execution and settlement. The full second partial moment and price–PV cross moment are retained in the analytical decoder section above.

## S28.3 Allocation and the shadow cost of capacity

For a cohort of size $a$, segment allocation $z$, widths $u$, segment costs $c$ and total slot execution $x_s$, the differentiable allocator solves

$$\min_z\;c^\top z+\frac{\tau}{2}\sum_s x_s^2,\qquad 0\le z\le u,\qquad \mathbf 1^\top z=a,\qquad \tau=2.\qquad\text{(S28.4)}$$

Sequential cohort reservations share capacity, so a cheaper slot can be valuable to an obligation with fewer alternatives. Let $\ell(x)$ be direct cost, $U$ available capacity and $F(U-x)$ continuation cost. Within a differentiable active set, write $m_s=\partial\ell/\partial x_s$ and $\kappa_s=-\partial F/\partial U'_s$ at $U'=U-x$. A feasible marginal transfer from slot $t$ to slot $s$ gives

$$J(x,U)=\ell(x)+F(U-x),\qquad D_{t\to s}J=m_s-m_t+\kappa_s-\kappa_t.\qquad\text{(S28.5)}$$

Delay reduces local cost when $m_t-m_s>\kappa_s-\kappa_t$. The comparison includes the capacity released at the earlier slot. For a concrete illustration, two unit cohorts can use slots 1–2 and 2–3, respectively, with unit slot capacities and $c_2<c_1<c_3$. Moving the first cohort from 1 to 2 saves $c_1-c_2$ directly but displaces the second cohort from 2 to 3 at cost $c_3-c_2$. Serving the first cohort earlier saves $c_3-c_1$ overall while completing the same work. This is a deterministic operating illustration, separate from the abstract information examples in Table 38.

## S28.4 Commitment-time executor

Let $B_t(d)$ be known outstanding admitted work whose exclusive deadline is no later than $d$, and let $K_s^-$ be a jointly achievable conservative future capacity path. The required current service is

$$r_t=\max(0,\max_{d\ge t+1}(B_t(d)-\sum_{s=t+1}^{d-1}K_s^-)).\qquad\text{(S28.6)}$$

For constant future capacity, the subtraction is $(d-t-1)K^{\min}$. The equivalent inclusive last-eligible slot $u=d-1$ gives $(u-t)K^{\min}$. The registered physical envelope uses zero PV, maximum PUE and the previous completed import:

$$\begin{aligned}G^{\mathrm{allow}}_{z,t}&=\min(G_z^{\max},G_{z,t-1}+R_z^{\mathrm{up}}),\\ K^{\mathrm{now}}_{z,t}&=\min(C_z,\max(0,(G^{\mathrm{allow}}_{z,t}/\mathrm{PUE}^{\max}_z-P_z^{\mathrm{idle}})/k_z)).\end{aligned}\qquad\text{(S28.7)}$$

Serving the larger of the economic request and $r_t$, subject to current capacity and admissions, preserves a continuation for known obligations when the declared envelope is feasible. A failed prefix or idle-load check requires an explicit infeasibility record or admission control; clipping work capacity to zero does not repair an infeasible idle import. These are the commitment-time equations. S23 documents that the historical electrical-stress projector reads same-interval realised PV/PUE and therefore requires separate linkage to any replacement robust implementation. No information claim follows merely from retaining the equations or diagram.

## S28.5 Original ablation interpretation

The completed ablations sharpen the interpretation of WorldMoE's advantage. Common-executor replacement establishes a US$19,587 system difference, whereas suppressing the opportunity, task-cost and cross-moment outputs jointly reduces mean saving by US$278. The retained fitted system still saves US$104,993, approximately US$19,309 above the dense comparator. The large advantage therefore survives suppression of these residual outputs at inference. The small joint effect and the US$82 improvement from disabling the final shift show that nominal component count is not a measure of economic contribution. These interventions identify the use of particular outputs after training; the common-executor comparison tests the broader fitted representation pathway. Isolating sparse routing from pretraining, capacity or optimization history requires corresponding matched replacements. APIC's algebraic price consistency remains established independently, since both shift settings preserve it.

The rounded interpretation above and all original values in Tables 9, 10, 28 and 29 are retained. The figures below are the six original main-paper images moved into this supplement without pixel changes. Historical certificate annotations and nearest-cent labels remain visible for source continuity; their captions state the relevant conditional or superseding interpretation. They are not new certificate computations.

