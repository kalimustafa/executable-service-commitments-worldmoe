# S31 Newly computed opportunity bounds for seven actual admission contracts

These are new perfect-information opportunity calculations executed on 31 October 2026 for 8, 12, 16, 20, 24, 36 and 48 hour homogeneous deadlines. They are not historical WorldMoE operating results. Every duration uses the same recovered actual arrivals, original immediate-service reference, realised prices, PV/PUE, static equipment limits and financial account. The previously published certificates uniform over every compatible admission stream remain a separate robustness result.

## Account and year-end treatment

For each site, $A_t$ is the exact cumulative sum of the source arrivals and $X_t$ the cumulative execution. With $D=4W$ eligible slots, original year-internal feasibility implies

$$A_{\max(0,t-D+1)}\le X_t\le A_t,\qquad X_t-X_{t-1}=x_t,\qquad0\le x_t\le1.$$

The outer problem does not impose $X_N=A_N$. Its closing-work term is exactly $\tau_z(A_N-X_N)$, with $\tau_z=0.25k_z\mathrm{PUE}_{\max}\operatorname{median}(p_z^{\mathrm{historical}})$. The recovered source reference completes all work by $N$ and has exact total $R_N=A_N$. Thus the source's reference-anchored MTM difference and the true closing-work difference coincide. Completion opportunities beyond $N$ retain their original deadlines; dropping their additional tail-capacity restrictions in the outer model only enlarges the original feasible set. No fictitious physical tail schedule is used to obtain the cost lower bound.

Each feasible witness voluntarily completes by $N$. This is permitted within the original rights and makes its true closing backlog and monetary terminal term exactly zero. It does not shorten the deadline used to define the global opportunity set. Witnesses require no unrecorded after-year physical inputs.

## Original nonconvex cost verification

Write $f_t(x)$ for the exact source slot cost: realised RT energy, curtailed PV at US$1/MWh, and absolute import deviation from the fixed reference at US$1/MWh. The import and curtailment functions use the recorded idle power, work power, PV and PUE. All points at which f_t changes slope, together with 0 and the exact physical capacity, are retained. Negative RT prices are allowed.

A convex-envelope LP proposes free cumulative-balance multipliers $\lambda_t$ and a candidate service schedule. Its floating optimum is not the certificate. Set $\lambda_{N+1}=0$, $L_t=A_{\max(0,t-D+1)}$ and $U_t=A_t$. The exact site cost floor is

$$
L_W^{\mathrm{site}}=\tau_z A_N+\sum_t\left[
\min_{0\le x\le1}\{f_t(x)-(\tau_z+\lambda_t)x\}
+\min\{(\lambda_t-\lambda_{t+1})L_t,(\lambda_t-\lambda_{t+1})U_t\}
\right].
$$

To prove the bound, rewrite the terminal cost as $\tau_z A_N$ minus $\tau_z$ times the sum of executed work, subtract $\lambda_t(x_t+X_{t-1}-X_t)$, and minimise over the original finite boxes. Equality terms cancel on every feasible path. A continuous piecewise-linear function attains its interval minimum at one of the complete retained breakpoints, so the first minimum is evaluated globally in the original nonconvex cost. Each complete scalar term is rounded down to $10^{-12}$ dollars before integer accumulation. All input floats and multipliers are interpreted as exact binary rationals.

A separate LP proposes a candidate with $X_N=A_N$. The exact capacity-propagated cumulative-corridor repair in S20/S29 produces an executable schedule. Every release prefix, exclusive-deadline prefix, physical-capacity bound, static import limit and terminal equality is checked exactly. The schedule is then re-settled with $f_t$, with complete monetary terms rounded upward. The common DA charge is included once in costs and once in the reference, so it cancels in saving comparisons. Opposing reference and schedule enclosure directions yield the saving floor $H_W$ and ceiling $U_W$.

## Nested endpoints and incremental rights values

Raw witnesses and ceilings are retained separately. Under the common account, the feasible sets nest as $W$ increases. A narrower-rights witness is also feasible for broader rights, and a broader-rights ceiling bounds a narrower opportunity. The displayed combined endpoints are therefore

$$H_W=\max_{v\le W}H_v^{\mathrm{raw}},\qquad U_W=\min_{v\ge W}U_v^{\mathrm{raw}}.$$

For consecutive tested durations $W_a<W_b$, the valid rights-premium interval is

$$\max(0,H_{W_b}-U_{W_a})\le V_{W_b}^{PI}-V_{W_a}^{PI}\le U_{W_b}-H_{W_a}.$$

Dividing both endpoints by $W_b-W_a$ gives the average incremental value per extra deadline hour on that interval. The interval widths reflect the feasible-witness to valid-ceiling gap, including nonconvex-envelope slack and the choice of a witness that finishes within the year. They are not seed ranges, confidence intervals or prospective profits. The seven tested durations do not by themselves establish a global law of diminishing returns.

## Numerical results

All values are annual four-site US dollars on the recovered actual admission stream. Floors are rounded downward and ceilings upward. No neighbouring window supplied a tighter endpoint than the raw endpoint in these runs; the reported raw and nesting-combined intervals therefore coincide.

| Deadline h | Feasible saving floor | Global saving ceiling | Interval width upper |
|:---|---:|---:|---:|
| 8 | 78,751.60 | 78,882.41 | 130.80 |
| 12 | 97,639.33 | 97,796.41 | 157.08 |
| 16 | 113,119.89 | 113,305.19 | 185.30 |
| 20 | 124,528.17 | 124,733.99 | 205.82 |
| 24 | 132,376.82 | 132,598.65 | 221.82 |
| 36 | 141,996.61 | 142,256.41 | 259.80 |
| 48 | 152,798.28 | 153,070.23 | 271.94 |

The per-hour column below is annual value per additional deadline hour, rather than an hourly electricity revenue rate.

| Extension h | Rights premium interval US$ | Annual US$ per extra deadline hour |
|:---|---:|---:|
| 8 to 12 | 18,756.93–19,044.80 | 4,689.23–4,761.20 |
| 12 to 16 | 15,323.48–15,665.86 | 3,830.87–3,916.47 |
| 16 to 20 | 11,222.98–11,614.10 | 2,805.74–2,903.53 |
| 20 to 24 | 7,642.84–8,070.47 | 1,910.71–2,017.62 |
| 24 to 36 | 9,397.97–9,879.59 | 783.16–823.30 |
| 36 to 48 | 10,541.87–11,073.61 | 878.48–922.81 |

The marginal annual value per extra deadline hour decreases across the tested intervals through 24–36 h, but rises from US$783.16–823.30 for 24–36 h to US$878.48–922.81 for 36–48 h. Because the later lower endpoint exceeds the earlier upper endpoint, that local rise follows even after allowing for the certificate intervals. The results do not support a global diminishing-returns claim.

The 24 h actual-stream opportunity is US$132376.82–132598.65, with a witness-to-ceiling gap below US$221.82. Relative to the previously recovered, exactly repaired three-seed WorldMoE mean, remaining same-rights opportunity is US$27105.76–27327.58 and the mean capture ratio lies in 79.39%–79.53%. The actual-stream 12–24 h optimal rights premium is US$34580.41–34959.31. These intervals describe newly computed perfect-information values for the historical calendar. They are distinct from historical paired operating-policy increments and from the earlier admission-uniform robustness guarantee.

All 28 annual site-duration witnesses passed exact release, exclusive-deadline, capacity, import and terminal checks when constructed. One saved 24 h site-3 multiplier file and full exact witness were additionally reloaded and re-evaluated without an LP solver; the saved-path report binds their hashes and inputs. This replay concerns one selected saved certificate and is not presented as a separate 28-case second-method verification.

## Files and reproduction

`code/run_actual_window_certificates.py` is the entry point. It uses one HiGHS thread and one worker. `code/exact_certificate_core.py` retains the previously used exact arithmetic and corridor-repair routines. `inputs/` supplies the recovered forcing, released reference arrays and original constraints. `results/actual_w*_zone*.json` records 28 site-duration certificates; the corresponding NPZs retain multipliers and raw proposals, and compressed JSONs retain full exact feasible executions and cumulative paths.

Run from any working directory:

    python code/run_actual_window_certificates.py

To reproduce the exact monetary values and feasibility checks from the saved multipliers and exact paths without solving any LP:

    python code/run_actual_window_certificates.py --verify-only

`seven_window_summary.json` and `seven_window_bounds.csv` retain raw and combined endpoints, selected nesting sources, source hashes and numerical environment. `adjacent_rights_premiums.csv` contains exact and outward-rounded increments. All headline monetary ceilings are rounded upward and floors downward only after exact full-precision computation.
