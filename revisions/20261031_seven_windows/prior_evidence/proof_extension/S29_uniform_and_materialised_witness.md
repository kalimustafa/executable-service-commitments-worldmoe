# S29 Uniform feasible execution and the recovered annual witness

## S29.1 Scope and statement

The result concerns the nominal divisible-work model with homogeneous 24 h rights, four fixed site assignments, zero initial queues, a unit service limit in each quarter-hour, the recorded static import limits, and the fixed realised price, PV and PUE path. The deadline of an arrival in slot $r$ excludes slot $r+96$. No ramp or nonpreemptive placement restriction is added to this nominal account. The original reference and import-deviation anchor are retained.

For one site, let $b=(b_1,\ldots,b_N)$ be the stored immediate-service reference, $R_t$ its exact cumulative sum, and $e(t)$ the first unsaturated slot at or after $t$. Let $\mathcal A(b)$ be the nonnegative finite binary64 admission arrays which generate $b$ under the documented round-to-nearest reference recurrence with zero initial queue. The four-site set is the product of these site sets. S27 establishes, uniformly over this set,

$$
\max(0,R_t-E)\le A_t\le R_{e(t)}+E,
\qquad E=2.412\times10^{-12},\quad A_0=0.
\tag{S29.1}
$$

Every stored busy block contains at most 35 saturated slots followed by an unsaturated clearing slot. Its smallest clearing-capacity slack is 0.04137009475380182..., larger than $E$. The final reference queue clears inside the observed year. The exact original-cost calculation establishes a unit feasible capacity in every slot.

**Proposition S29.1 (uniform constructive feasibility and value).** Suppose the preceding conditions hold. Let the stored exact surrogate cumulative schedule satisfy

$$
R_{e(\max(0,t-95))}\le\bar X_t\le R_t,
\quad 0\le\bar X_t-\bar X_{t-1}\le1,
\quad\bar X_0=0,\quad\bar X_N=R_N.
\tag{S29.2}
$$

Then, for every $a\in\mathcal A(b)$, the explicit construction below gives an execution $x(a)\in\mathcal X_{24}(a)$ which completes all admitted work by $N$ and satisfies

$$
\sum_{t=1}^{N}|x_t(a)-\bar x_t|\le2NE.
\tag{S29.3}
$$

Across the four sites, original-cost settlement and the signed terminal-anchor allowance imply

$$
\forall a\in\mathcal A(b),\quad
\exists x(a)\in\mathcal X_{24}(a):
C_0-C(x(a))\ge H_{24}^{\rm unif}
\ge130931.31\ \mathrm{US\$}.
\tag{S29.4}
$$

The claim is an established existence and value theorem. Its quantifier order allows the exact execution to depend on the admitted stream. It does not exchange these quantifiers or assert that the value guarantee is achieved by a policy adapted to a decision-time filtration.

## S29.2 Constructive proof

First establish nonemptiness of the exact corridor for each $a$. Define the exact earliest-service queue $Q_t=\max(Q_{t-1}+a_t-1,0)$. This update is 1-Lipschitz in its queue argument. Comparing it with the documented floating recurrence and accumulating the outward error bound of S27 gives $|Q_t-q_t|\le E$. At each stored clearing slot $c$, its exact pre-service queue is at most $b_c+E<1$. Thus the exact queue also clears at $c$. Any work admitted in a source busy block is completed no later than its clearing slot, within at most 36 slots; work between busy blocks completes immediately. This is feasible for 96-slot rights, and the final clearing slot proves exact completion by $N$. This step proves feasibility for each compatible admission array without reconstructing that array from $b$.

For the selected $a$, form exact bounds

$$
L_t=A_{\max(0,t-95)},\quad U_t=A_t,
\quad L_0=U_0=0,\quad L_N=U_N=A_N.
\tag{S29.5}
$$

Setting $L_N=A_N$ selects a schedule which completes early; it does not redefine the original contract. Propagate the lower bound backwards and the upper bound forwards:

$$
\ell_N=L_N,\quad\ell_t=\max(L_t,\ell_{t+1}-1),
\qquad
u_0=U_0,\quad u_t=\min(U_t,u_{t-1}+1).
\tag{S29.6}
$$

Exact earliest service proves that the propagated interval is nonempty: every feasible cumulative path lies between $\ell_t$ and $u_t$. The propagated bounds are nondecreasing and their increments are at most one. Define

$$
X_t(a)=\min\{u_t,\max\{\ell_t,\bar X_t\}\},
\qquad x_t(a)=X_t(a)-X_{t-1}(a).
\tag{S29.7}
$$

The minimum and maximum of nondecreasing sequences whose increments lie in $[0,1]$ retain those properties. Hence $0\le x_t(a)\le1$. The clamped path also obeys $L_t\le X_t(a)\le U_t$ and the initial and final equalities. For homogeneous deadlines, the release and due prefixes are equivalent to feasible earliest-deadline cohort allocation: execution cannot precede release, and every prefix of due cohorts has completed by its exclusive deadline. Consequently $x(a)\in\mathcal X_{24}(a)$, with zero true terminal backlog. The exact import-cap condition already holds for every execution in the unit box.

To bound the displacement, Eq. (S29.1) and the strict surrogate corridor imply that every unpropagated release/deadline/endpoint violation of $\bar X$ is at most $E$. This remains true after capacity propagation. For example, $\ell_t$ is a maximum of future lower bounds minus intervening capacity; since $\bar X$ has increments at most one, each term is at most $\bar X_t+E$. The forward argument gives $u_t\ge\bar X_t-E$. Thus $|X_t(a)-\bar X_t|\le E$ for every prefix. Taking successive differences gives

$$
\sum_t|x_t(a)-\bar x_t|
\le\sum_t\big(|X_t(a)-\bar X_t|+
|X_{t-1}(a)-\bar X_{t-1}|\big)\le2NE.
\tag{S29.8}
$$

Finally, the original import, curtailment and import-deviation slot cost is $M_z$-Lipschitz in work, with $M_z=\max_t[0.25 k_z\mathrm{PUE}_{zt}(|p_{zt}|+2)]$. The source signed terminal-anchor offset is $\tau_z(R_N-A_N)$, bounded in absolute value by $21E_z$. Therefore

$$
C(x(a))\le C(\bar x)+\sum_z2NM_zE_z+21\sum_zE_z.
\tag{S29.9}
$$

The arrival-repair contribution is below US$0.000836657841429, and the signed terminal allowance is US$0.000000000202608. The exact downward reference enclosure and upward candidate-cost enclosure in S27 prove Eq. (S29.4). This completes the constructive proof.

## S29.3 Materialisation using recovered source admissions

The revised evidence package now includes the original local `external_validation/data/forcing_2025.npz`, whose SHA-256 is `c600c9e01e8236bd2e1cb4e2dc08591e3ee264d05f1c26b4c785ec355c3147c8`. Its `eagle_arrivals` field is the admitted stream loaded by the archived `context.configure` function. Applying the unmodified reference recurrence reproduces every published `reference_compute` entry bit for bit and ends with a zero queue. Its RT prices, PV, PUE, work-power coefficients and idle loads are also bitwise identical to the released annual arrays, SHA-256 `5be123c40f01b111cf6503b07882e703a4c2cc29a03dde3cfb7aa4321ce2cf72`. This establishes a matched source ledger, rather than inferring arrivals by inversion of reference service.

`materialize_actual_witness.py` reads every stored binary64 input as an exact rational, applies Eq. (S29.7), saves the full exact execution and cumulative ledger, and checks every release, exclusive deadline, service-capacity and static-import constraint. The supplied four records each cover all 35,040 slots. For these recovered admissions, the stored surrogate already satisfies every exact constraint: all four repair $L^1$ norms are exactly zero. Admitted and reference totals coincide exactly, and each witness has zero true backlog and zero signed terminal-anchor offset. The smallest exact import slack is greater than 0.106672550152318 MW.

Exact original-account re-settlement gives the materialised annual saving enclosure

$$
130931.31968120
\le C_0-C(x(a^{\rm src}))
\le130931.31968149\quad\mathrm{US\$}.
\tag{S29.10}
$$

Both enclosing endpoints use complete-term directed monetary rounding. The main text retains the more general uniform floor US$130,931.31, so all reported rights-premium floors remain conservative. The earlier statement that the actual-admission execution could not be materialised is superseded by these supplied records. This recovery does not turn the hindsight construction into an information-adapted operating policy.

The evidence consists of `actual_witness_results.json` and four `actual_admission_witness_zone*.json.gz` files. Each compressed file contains exact arrivals, execution, cumulative arrivals and cumulative execution as rational strings, together with its surrogate provenance and repair amount. The verifier invokes no optimisation solver; the recorded hashes bind the recovered inputs and resulting full paths.

