# Independent mathematical review of the admission-envelope certificate

Reviewed implementation: `envelope_certificate.py`, as present on 31 October 2026 before the full annual run. This review concerns the new construction, not recovery of the original annual inputs or certificate files. Terminal-rate nonnegativity is a separate evidence obligation assigned to root.

## 1. Reference rounding envelope

Let a_t be nonnegative binary64 arrivals, q_0=0, s_t=fl(q_{t−1}+a_t), b_t=min(s_t,1), q_t=fl(s_t−b_t). This is the unmodified `vendor/reservation.py:4–10` recurrence with capacity one. Assume IEEE round-to-nearest and finite arithmetic. Set u=2^−53 and β=2u+u². Writing S_t=q_{t−1}+a_t in exact rational arithmetic,

q_t=S_t−b_t+e_t,  |e_t|≤βS_t.

The bound follows by adding the addition error and subtraction error; |s_t−b_t|≤s_t≤(1+u)S_t. Nonnegative binary64 operations with a subnormal result have an exact result on the common 2^−1074 lattice, so the standard relative bound is not invalidated there. In a saturated subtraction near one, Sterbenz exactness gives the same conclusion. A finite clearing block cannot contain overflow: an infinite queue could not return to an unsaturated clearing output under nonnegative finite arrivals.

If b_t<1, q_t=0 exactly because b_t=s_t. An isolated unsaturated slot following an empty slot has s_t=a_t exactly, b_t=a_t, and zero error. Only saturated blocks and their first following unsaturated slot need error allowances.

For a block of m saturated slots and its unsaturated clearing slot, backwards use q_t≥(1−β)q_{t−1}−b_t and q_t≥(1−β)S_t−b_t. Since every output is at most one and the final queue is zero, a sharper bound is S_t≤(m+1)/(1−β)^(m+1). The code's larger (m+2)/(1−β)^(m+2) is therefore safe. Summing at most m+1 error terms and using (1−β)^(m+2)≥1−(m+2)β gives the implemented block allowance

E_block=(m+1)(m+2)β/[1−(m+2)β].

The denominator must be positive. It is overwhelmingly positive for the observed maximum m=35. A general-purpose implementation should reject a nonpositive denominator rather than silently use it.

Let E be the sum over all blocks. With A_t cumulative arrivals and R_t cumulative observed b, summation gives q_t=A_t−R_t+Σ_{s≤t}e_s, hence |A_t−R_t−q_t|≤E. Thus A_t≥max(0,R_t−E). At the first clearing boundary c(t) at or after t, q_c(t)=0 and monotonicity gives A_t≤A_c(t)≤R_c(t)+E. Preserve A_0=0 exactly rather than using an E-padded origin. The current arrays finish in an unsaturated slot, so c(t) exists throughout the observed horizon. The code enforces this final condition.

## 2. Exact earliest-service feasibility for every compatible original arrival array

This is the critical existence step for the robust witness. Define the exact earliest-service queue for the true a by Q_0=0 and

Q_t=max(Q_{t−1}+a_t−1,0).

The map q↦max(q+a_t−1,0) is 1-Lipschitz. The floating update differs from this map by at most βS_t, so induction gives |Q_t−q_t|≤Σ_{s≤t}βS_s over the charged block steps, bounded by E.

At a source clearing slot c, s_c=b_c<1. The exact pre-service input satisfies

Q_{c−1}+a_c≤q_{c−1}+a_c+E_previous≤b_c+E_previous+|s_c−S_c|≤b_c+E.

Therefore the strict check E<1−b_c implies Q_{c−1}+a_c<1 and Q_c=0 exactly. The one-step addition error is covered because it is no larger than the βS_c contribution already included in E. This proves clearing at the same slot, not merely at an unspecified later tail slot.

Every such exact queue is empty at the boundary before a source saturated block and after that block's clearing slot. Thus all work admitted during a block is completed within its m+1 slots; isolated unsaturated arrivals complete immediately. The observed maximum m=35 implies at most 36 eligible slots are used by exact FIFO earliest service. Consequently 48/64/96-slot homogeneous deadlines are feasible under all compatible arrivals, and all admitted work clears by the observed-year end. The implementation's max_run+2<D test for D=96 is conservative. If the theorem is to claim all three contracts, explicitly check D≥max_run+1 for each rather than relying on the 24 h branch alone.

This argument requires nonnegative true arrivals, zero initial queue and the stated reference recurrence. It does not infer a unique arrival array. It proves existence for every array consistent with the reference evidence and rounding model.

## 3. Outer corridor and exact Lagrangian

For homogeneous duration D, original feasible cumulative execution obeys A_{t−D+1}≤X_t≤A_t, with negative/zero arrival indices set to zero. Replacing the lower prefix by A_min and the upper prefix by A_max enlarges the observed-year feasible set. Dropping the beyond-year completion assignment also enlarges it. Nonnegative terminal coefficients establish a lower relaxation for the actual-backlog term. The source additionally anchors that charge to rounded reference total R rather than actual A; its signed constant must be retained through the adverse allowance in Section 9 below. The root's independent proof of the coefficient sign is therefore essential, but positivity alone does not justify replacing the entire source terminal charge by zero.

The equality x_t+X_{t−1}−X_t=0 and multiplier convention λ_t(0−x_t−X_{t−1}+X_t) give original-slot terms f_t(x_t)−λ_t x_t and cumulative-state coefficients λ_t−λ_{t+1}, with λ_{N+1}=0. Accordingly the code's exact lower expression

-21E + Σ_t min_{0≤x≤c_t}(f_t(x)−λ_t x)+Σ_t min[(λ_t−λ_{t+1})L_t,(λ_t−λ_{t+1})U_t]

has the correct signs and terminal index. Every multiplier vector is valid. The floating LP needs only to propose useful multipliers; an inaccurate hull or solution cannot invalidate an independently evaluated original-cost box bound. Original piecewise-linear cost minima occur at endpoints and zero-/reference-import breakpoints, including negative prices.

The exact baseline upper bound is used for a savings ceiling. The baseline lower bound is used for a feasible saving floor. Common charges enter each cost account once and cancel in the saving difference. Per-term upward/downward integer rounding has the correct directions.

## 4. Robust surrogate witness and actual-admission repair

For D=96, the surrogate corridor uses U_t=R_t and L_t=R_c(t−D+1), with L_0=U_0=0 and L_N=U_N=R_N. Because every source busy block ends within 36 slots, R itself is feasible for this corridor. It therefore has a nonempty fixed-capacity feasible set. The solver's candidate may be clipped to its exact box before exact cumulative clamping because its final cost is freshly evaluated; no claimed small repair bound from the unclipped solver output is required.

For every compatible true arrival array, a surrogate-feasible X violates original release and deadline prefixes by at most E in each direction: X_t≤R_t≤A_t+E, and X_t≥R_c(t−D+1)≥A_{t−D+1}−E. At N, |R_N−A_N|≤E since the reference queue has cleared. Impose the exact actual endpoint X_N=A_N. The exact earliest-service construction above proves the actual corridor feasible. Applying the previously proved cumulative clamp theorem to this actual corridor gives an actual-admission schedule with total service change at most 2NE.

This is a uniform constructive existence/value guarantee. Because the actual a is unavailable, the final actual-admission schedule is not materialized; the code correctly identifies it as such. It should not be described as a recovered original witness. The saved surrogate schedule and the uniform transformation theorem are the new evidence.

All exact service caps must be one for the stated reference-based feasibility proof. The code checks that condition using the original static import graph and supplied physics. A repaired x∈[0,1] then satisfies every original nominal static import cap. No ramp guarantee is established or claimed. Every actual arrival is completed by N, so there is no physical recovery tail and the true terminal backlog is zero. The source's reference-anchored monetary terminal charge may nevertheless have the tiny signed offset described in Section 9. The surrogate cost plus the Lipschitz allowance and that adverse terminal allowance bounds its cost; all evaluations use the fixed original reference in the deviation charge.

## 5. Initial pre-run review conclusion and empirical gates

The mathematical construction is valid under its declared recurrence, arithmetic, zero-origin, final clearing, static-cap and nonnegative-terminal assumptions. The exact earliest-service proof above closes the previously uncertain feasibility step without an invented admission reconstruction. Required recorded gates are: reference identity/provenance; all finite nonnegative arrivals under the source recurrence; maximum busy run and clearing slack; positive error denominator; all nominal exact capacities equal one; the true terminal coefficient sign; exact independent reevaluation of all saved outputs; and explicit labeling of uniform existence versus a materialized true-admission schedule.

Initial pre-run status: a new code run and independent output checks were still required before asserting numerical annual floors or ceilings. The mathematical derivation at that stage did not itself supply those computed numbers. The completed verification status is recorded in Section 10 below; this paragraph preserves the chronology of the initial review.

## 6. Subsequent epigraph implementation review

The independently assembled `epigraph_certificate.py` uses variables x, S and eta, balances x_t+S_(t-1)-S_t=0 and facets alpha*x-eta<=-beta. With the convention lambda*(0-balance), nonnegative inequality multiplier mu and objective sum eta, the residuals are

rx=-lambda+sum(mu*alpha), rS=lambda-lambda_next, reta=1-sum(mu),

and the constant is sum(mu*beta). These signs are correct. All three finite-box residual contributions are retained, including the eta contribution when the proposed dual is not exactly feasible. Clipping a proposed inequality multiplier to nonnegative is permissible because the entire expression is then independently reevaluated.

For each slot, checking a candidate affine facet below the original cost at every original breakpoint proves dominance everywhere: the original graph is affine between adjacent breakpoints, and the difference from a facet is affine on each such interval. Using the minimum and maximum original breakpoint costs as finite eta bounds retains the original embedding eta=f_original(x), including negative-price slots. A finite eta box is therefore valid and must not be silently removed from the residual evaluation.

Both certificate methods share the physical account and reference-derived corridor. Their matrix construction, dual representations and retained verification artifacts are independent; they are not independent data-generating experiments. A saved-output verifier should also check the input hash and reconstruct physical/corridor data before claiming replay of a specific retained certificate. The saved multipliers remain algebraically valid against any correctly reconstructed model, but artifact identity is a separate reproducibility check.

A signed-cent formatting error was identified in the initial result printer: negative integer cents were displayed using floor division and remainder without separating the sign. It was corrected in both shared formatters and covered by a regression. Exact rational bounds and all positive headline H/U values were unaffected; the corrected downward display for H24-U24 is -3506.54 dollars.

## 7. Independent terminal-rate enclosure review after CSV correction

The final `terminal_rate_audit.py`, `.json` and `.txt` account for the single intermediate Pandas CSV parse as well as the binary64 debt/mark-to-market arithmetic. The official retained default-parser source uses at most 17 digits, at most 34 positive Horner operations, cached decimal powers and at most two divisions. Its relative arithmetic error is conservatively covered by gamma_40. The leading zero in fixed notation below one may discard less than 1e-16 in absolute decimal value; shortest-round-trip rendering is explicitly assumed and bounded separately. Thus eta=(1+gamma_2)(1+gamma_40)-1 and delta=(1+gamma_40)*1e-16+1e-300 safely give |y-y0|<=eta*|y0|+delta. Eliminating the unobserved original daily sum yields (eta*|y|+delta)/(1-eta), which is correctly added to the rate-independent error term.

An independent Fraction calculation from the saved JSON verified M*inverse=I exactly, reconstructed the four central coefficients, independently recomputed the inverse infinity norm, the self-bounding rate norm and the radius, and matched every saved lower endpoint. The final radius is 1.7274828236953199e-9 and the smallest lower endpoint is 31.006947148502842. The maximum selected-row CSV allowance is 3.685095343199813e-11 dollars. Strict positivity follows under the explicitly documented matched-account, binary64 and parser-algorithm conditions. This does not infer the historical installed Pandas version from the CSV or recover the original training-price file.

## 8. Frozen reference account versus exact reference feasibility

The new certificate fixes C0 and the import-deviation fee anchor to the published reference execution b, exactly as the documented original account does. Because the true total admissions may differ from the exact sum of the rounded b by at most E, b itself is not promoted to an exactly feasible witness for the hidden arrivals. This does not invalidate the savings bounds: a fixed common monetary reference is an affine normalization, and the deviation-fee anchor is part of the declared objective. Exact service feasibility is established for the candidate schedule. Nested-rights increments cancel the common C0. Repairing the reference and moving the deviation anchor would define a different account and is unnecessary for the retained-source comparison.

## 9. Reference-anchored terminal offset: final account correction

The root identified that `economic_core.record` prices the difference from reference service, so its exact model terminal term is tau_z*(R_N,z-X_N,z), not tau_z*(A_N,z-X_N,z). This distinction is essential even though the reference-rounding difference is tiny. Since original feasibility imposes X_N,z<=A_N,z and |R_N,z-A_N,z|<=E, strict positive tau_z with the proved upper bound tau_z<21 gives the portfolio terminal lower bound -delta_T, where delta_T=21*4*E=0.000000000202608 dollars. Each outer cost lower bound must subtract delta_T; each candidate cost upper bound must add delta_T. At repaired X_N=A_N, physical backlog is zero but the source terminal amount can still be signed. Every upgraded-witness-minus-narrow-ceiling lower endpoint therefore retains both allowances.

Both annual certificate implementations, saved-output verifiers and aggregate JSON results were updated to apply this exact rational allowance, with no change to the displayed positive cent values. The correction is kept explicit instead of being absorbed into an unrelated numerical-loss budget. The complete supplement terminal subsection derives coefficient positivity and this account correction together.

## 10. Final retained-record validation status

The subsequent full-year computations and solver-free saved-output replays are complete. They verify all 12 original-cost certificates, all 12 independently assembled epigraph certificates, all four complete exact rational surrogate schedules, and all 2,095,170 retained facet–breakpoint dominance comparisons. The replay reconstructs the exact physical and corridor data, validates retained input/multiplier content and hashes, and reproduces the annual values with the reference-anchored terminal allowance included. Seven separate small regression fixtures exercise formulas, signs, repair, negative prices, admission non-identification and signed outward cent formatting; these fixture tests are not substituted for annual replay.

The upward-cent savings ceilings are 100,625.48, 115,735.23 and 134,437.85 dollars for 12, 16 and 24 h rights, respectively. The uniform 24 h feasible-value floor is 130,931.31 dollars. The 24 h floor exceeds both independently evaluated 12 h ceilings by at least 30,305.84 dollars and both 16 h ceilings by at least 15,196.08 dollars after the adverse arithmetic and terminal allowances. These are retrospective perfect-information opportunity statements under the documented fixed reference account and source-lineage conditions, not ex-ante policy-performance guarantees.

The actual admission array and the original 29 October witness and certificate assets remain unrecovered. The supplied schedules are the new exact surrogate schedules; the admission-dependent repair is a uniform constructive existence theorem with an explicit monetary allowance. This completed validation does not relabel those surrogates as a recovered original assignment, and it does not establish historical first-release input availability.
