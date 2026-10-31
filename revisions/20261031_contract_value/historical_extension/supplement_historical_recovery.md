# S30 Recovered historical trajectories and exact numerical service witnesses

## S30.1 Provenance and relation to Table 2

We recovered the original nine annual WorldMoE service books from the local September 2026 archive: the 12 h, 16 h and 24 h runs for seeds 20260920, 20260921 and 20260922. The selected checkpoint epochs are 6, 2 and 0, respectively. For each seed, the checkpoint SHA256 is identical across all three service windows. The original summary files, in-year execution arrays, complete books and daily monetary records are included without modification. These files were absent from the earlier public evidence snapshot; their absence was a packaging gap, not evidence that the historical experiments had not retained trajectories.

The original `forcing_2025.npz` supplies 35,040 quarter-hour admission vectors for four sites. Its `eagle_arrivals` field is explicitly read by the archived evaluation source. The original earliest-service recurrence applied to these arrivals reproduces the published reference-compute array bit for bit. The observed price, PV, PUE, idle-power and work-power arrays also match the published account bit for bit. A source manifest records file hashes and original relative paths. This recovery permits the service and settlement checks below without reconstructing admissions from a non-identifying reference schedule.

Table 2 reports **historical paired policy results with recovered trajectory verification**. The values describe frozen historical policy outcomes on one observed calendar. They are neither independent calendar replications nor estimates of a perfect-information contract optimum.

## S30.2 Independent service and settlement audit

For each site, use zero-based slots $t=0,\ldots,N-1$ and prefix boundaries $k=0,\ldots,N$. Let $a_t$ be the recovered admissions and $x_t$ the saved service during slot $t$, and define

$$
A_k=\sum_{s=0}^{k-1}a_s,\qquad
X_k=\sum_{s=0}^{k-1}x_s,\qquad
A_0=X_0=0,\qquad x_t=X_{t+1}-X_t.
$$

A cohort released in slot $t$ with $D$ quarter-hour slots of service rights must finish by the end of slot $t+D-1$, namely boundary $t+D$. The zero-tail witnesses finish every cohort within the observed year, including cohorts whose contractual deadlines extend beyond it. Their complete conditions are

$$
0\le x_t\le1\quad(0\le t<N),\qquad
X_k\le A_k\quad(0\le k\le N),\qquad
X_{\min(N,t+D)}\ge A_{t+1}\quad(0\le t<N),\qquad
X_N=A_N.
$$

The raw historical audit retains the 8,192 stored tail intervals in every complete book. All nine policy tails and all reference tails contain exactly zero service. The corresponding terminal cost differences are therefore zero. Reconstructed grid imports satisfy the original static import caps, and the source electricity account is independently evaluated from prices and physical arrays. The raw savings reproduce US$72,220.96961418838, US$86,038.55437473873 and US$105,271.06184163212 for the respective three-seed means. Same-seed 12-to-24 h contrasts average US$33,050.092227443754.

The raw schedules have nonzero floating-point residuals at approximately $10^{-11}$ work units or below under ordinary cumulative summation. We also accumulate each stored binary64 number as an exact dyadic rational and report its exact release, deadline and mass residuals. A tolerance pass is not relabelled as exact feasibility. The following separate repair provides exact feasible witnesses while preserving the historical files.

## S30.3 Exact corridor repair

For the recovered admission path, define the corridor at prefix boundary $k$ by

$$
L_0=0,\qquad L_k=A_{\max(0,k-D+1)}\quad(1\le k<N),\qquad
L_N=A_N,\qquad U_k=A_k\quad(0\le k\le N).
$$

The index $k-D+1$ is required by the slot convention: at boundary $k$, all cohorts with release slots $t\le k-D$ are due, and their total is $A_{k-D+1}$. For example, when $D=1$, the corridor requires $X_k\ge A_k$; together with release precedence this gives immediate service. The separate assignment $L_N=A_N$ imposes the verified zero-tail completion convention.

First clip each saved slot service to $[0,1]$, and let its cumulative boundary path be $Z_k$. Propagate the corridor backward from $\ell_N=L_N$ and forward from $u_0=U_0$:

$$
\ell_k=\max\{L_k,\ell_{k+1}-1\}\quad(k=N-1,\ldots,0),\qquad
u_k=\min\{U_k,u_{k-1}+1\}\quad(k=1,\ldots,N).
$$

The finite instance is checked to satisfy $\ell_k\le u_k$ at every boundary. Each propagated boundary is nondecreasing with increments at most one. The repaired cumulative path and slot service are

$$
\widehat X_k=\min\{u_k,\max\{\ell_k,Z_k\}\}\quad(0\le k\le N),\qquad
\widehat x_t=\widehat X_{t+1}-\widehat X_t\quad(0\le t<N).
$$

is therefore nondecreasing with increments in [0,1], lies in the original corridor, and completes all admitted work. Integer arithmetic independently verifies every release, deadline, capacity and terminal condition for all nine exported witnesses. The physical data additionally satisfy the electrical import cap at service equal to one in every observed interval; hence the repaired unit-capacity service satisfies the electrical cap. These are finite-instance certificates for the recovered historical admissions.

The repaired schedules are numerical witnesses, not substituted historical policy outputs. The unchanged raw books remain the source of Table 2. No inference-time policy action, checkpoint or information pathway was recomputed by this procedure.

## S30.4 Monetary effect and exact account reconciliation

For the original account with a US$1 per MWh import-deviation charge and a US$1 per MWh curtailment term, a site-wise Lipschitz bound is

$$
M_z=\max_t\frac14 k_z\,\mathrm{PUE}_{tz}
\left(|p_{tz}^{RT}|+2\right).
$$

Consequently, with $\delta_z=\sum_t|\widehat x_{tz}-x_{tz}|$,

$$
|J(\widehat x)-J(x)|\le\sum_z M_z\delta_z.
$$

The program evaluates both this bound and the repaired account exactly as rational numbers. It also checks the actual objective difference against the bound. Across all nine runs the largest cost bound is US$ $1.65626764744\times10^{-8}$. This is far below one cent; it is explicitly budgeted rather than ignored.

| Contract | Exact repaired three-seed mean saving enclosure, US$ |
|---|---:|
| 12 h | [72,220.969614188, 72,220.969614189] |
| 16 h | [86,038.554374738, 86,038.554374739] |
| 24 h | [105,271.061841632, 105,271.061841633] |
| 12 h to 24 h paired increment | [33,050.092227443, 33,050.092227444] |

The US$33,050.09 contrast is therefore supported by complete original execution records, recovered admissions, explicit terminal accounting and separate exact numerical feasibility witnesses. It remains a historical policy contrast. The reported perfect-information certificate increment concerns the difference between optimal feasible sets. These are different estimands and neither is a guaranteed ex-ante contract profit.

The 24 h feasible witnesses use the same recovered admissions, reference-cost account, physical arrays, service capacity, electrical cap and terminal convention as the contract-value analysis. Their exact saving enclosure supplies the operational numerator when reporting residual opportunity and capture relative to the 24 h perfect-information bound. The rounding repair does not alter those metrics at their reported precision.

## S30.5 Remaining opportunity and operating capture

Let $S_{24}^{*}$ denote the optimal 24 h perfect-information saving for the same recovered admission path and account. Retain the conservative uniform value floor and annual ceiling $L=130931.31$ and $U=134437.85$ dollars, so that $L\le S_{24}^{*}\le U$. Let $P$ be the mean saving of the three separate exact feasible policy witnesses. Each witness is feasible under that same contract; hence its saving, and their mean, cannot exceed $S_{24}^{*}$. Exact rational evaluation followed by directed rounding gives

$$
P_{\mathrm{lo}}=\frac{105271061841632}{10^9}
\le P\le
\frac{105271061841633}{10^9}=P_{\mathrm{hi}}.
$$

These endpoints are computed by flooring and ceiling the exact rational mean after multiplication by $10^9$. Define remaining opportunity $G=S_{24}^{*}-P$. Subtracting endpoints in the correct directions gives

$$
L-P_{\mathrm{hi}}\le G\le U-P_{\mathrm{lo}}.
$$

Conservatively rounding the lower endpoint down and the upper endpoint up to cents yields

$$
\frac{\lfloor100(L-P_{\mathrm{hi}})\rfloor}{100}
=25660.24
\le G\le
\frac{\lceil100(U-P_{\mathrm{lo}})\rceil}{100}
=29166.79\quad\mathrm{US\$}.
$$

Because $L>0$ and $P>0$, the opportunity capture fraction $\kappa=P/S_{24}^{*}$ satisfies

$$
\frac{P_{\mathrm{lo}}}{U}\le\kappa\le\frac{P_{\mathrm{hi}}}{L}.
$$

The percentages are approximately $78.30\%$ to $80.40\%$ when displayed to two decimal places. These approximate displayed percentages are distinguished from the exact endpoint ratios above. The residual opportunity describes the room between the feasible operating results and the bounded hindsight optimum on the observed calendar. It is not a guarantee that an implementable causal policy can capture the entire amount, nor a forecast of the value of buying longer service rights.

