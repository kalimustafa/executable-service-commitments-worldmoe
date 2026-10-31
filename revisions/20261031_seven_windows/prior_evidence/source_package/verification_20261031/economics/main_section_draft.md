# Contract economics: historical upgrade budgets and ex-ante value

The financial comparison has three distinct targets. A fixed policy's historical saving is the original reference cost minus its settled cost on the recorded calendar. The perfect-information opportunity is the maximum saving after the entire calendar is known. The ex-ante value is the largest expected saving achievable by a policy that uses only information available when each action is committed. All three use the same admissions, physical system, reference account and terminal convention; none includes an unobserved tenant payment.

For a contract $W$, define

$$S_W^\pi(\omega)=C_0(\omega)-C(\pi,\omega),\quad V_W^{\mathrm{PI}}(\omega)=\sup_{x\in\mathcal X_W(\omega)}[C_0(\omega)-C(x,\omega)],\quad V_W^{\mathcal F}=\sup_{\pi\in\Pi_W^{\mathcal F}}\mathbb E[S_W^\pi].\tag{E1}$$

The expectation refers to a probability law fixed at contract signing, and the policy class respects the receipt-time filtration. Integrable values give $\mathbb E[S_W^\pi]\le V_W^{\mathcal F}\le\mathbb E[V_W^{\mathrm{PI}}]$. This inequality does not identify an expectation from one realised year. Historical re-settlement verifies the ledger for supplied actions; first-release records establish their information admissibility.

Let $I_W=\mathbb E[V_W^{\mathrm{PI}}]-V_W^{\mathcal F}\ge0$ be the information gap. For nested rights $A\subseteq B$,

$$V_B^{\mathcal F}-V_A^{\mathcal F}=\mathbb E[V_B^{\mathrm{PI}}-V_A^{\mathrm{PI}}]-(I_B-I_A).\tag{E2}$$

The change in the information gap has no fixed sign. Consequently, even an expected perfect-information upgrade value is not generally an upper or lower bound on the ex-ante upgrade value. The reported hindsight witness identifies a realised-calendar opportunity conditional on exact verification. It cannot substitute for a nonanticipative policy's expected saving. A sufficient ex-ante exclusion test instead compares a validated lower bound on $\mathbb E[S_B^{\pi_B}]$ with a matched expected upper bound for contract $A$.

For the retrospective optimum, valid ceilings and exactly feasible witnesses give

$$\max(0,H_B-U_A)\le V_B^{\mathrm{PI}}-V_A^{\mathrm{PI}}\le U_B-H_A.\tag{E3}$$

All monetary corrections in Eq. (5e) apply to these witnesses. Subtracting two savings ceilings or two feasible policy savings does not establish the optimal contract premium. A paired policy contrast additionally reflects the change in that controller's distance from each contract optimum.

## Paired operating value and the break-even cost budget

The retained annual table pairs three fixed checkpoint hashes across 12, 16 and 24 h contracts. Subtracting each seed's narrow-rights saving from that same seed's broad-rights saving gives Table E1. The values already include RT settlement, curtailment valuation, the import-deviation charge and the source's terminal accounting. They are before additional tenant compensation and implementation expenses. The 24 h component is independently reproduced from all 35,040-by-four price, physical and execution arrays. The 12 and 16 h components are supported by the released annual summaries; their complete execution and admission books are unavailable for a new independent replay.

**Table E1. Historical incremental operating value of the same frozen WorldMoE checkpoint.** Annual four-site US dollars. Ranges describe the three paired refinement outcomes on the same calendar, not uncertainty over future years.

| Upgrade | Mean incremental saving | Paired refinement range | Mean historical break-even $F+K$ |
|:---|---:|---:|---:|
| 12 → 16 h | 13,817.58 | 13,733.95–13,895.69 | 13,817.58 |
| 16 → 24 h | 19,232.51 | 19,169.59–19,308.33 | 19,232.51 |
| 12 → 24 h | 33,050.09 | 32,903.54–33,204.02 | 33,050.09 |

For an incremental annual tenant transfer $F$ and operator implementation cost $K$, the same-policy historical net increment is

$$N_{A\to B}^{\pi}(\omega;F,K)=S_B^{\pi_B}(\omega)-S_A^{\pi_A}(\omega)-F-K.\tag{E4}$$

Thus the mean historical 12→24 h increment remains positive when $F+K<\mathrm{US\$}\,33,050.09222744375$. For illustration, an assumed extra cost of US$20,000 leaves US$13,050.09; US$35,000 leaves −US$1,949.91. The same US$20,000 assumption gives −US$6,182.42 for 12→16 h and −US$767.49 for 16→24 h. These are analytical cost sensitivities, not observed fees or estimated future profits. The source contains no tenant reservation-price sample or probability law from which to estimate the ex-ante net increment. Zero additional cost is an explicit scenario, not an estimate that those costs vanish.

## Operator value, tenant participation and joint surplus

For fixed incremental charges, the ex-ante operator value relative to retaining $A$ is

$$N_{A\to B}^{\mathcal F}=V_B^{\mathcal F}-V_A^{\mathcal F}-F-K_o.\tag{E5}$$

If $g_L\le V_B^{\mathcal F}-V_A^{\mathcal F}\le g_U$, $F\in[F_L,F_U]$ and $K_o\in[K_L,K_U]$, then $N^{\mathcal F}\in[g_L-F_U-K_U,g_U-F_L-K_L]$. A positive lower endpoint proves a positive margin under that same expectation criterion; a negative upper endpoint rules it out. An interval containing zero is inconclusive. Applying the same arithmetic to Eq. (E3) gives a retrospective perfect-information net interval with that explicit interpretation.

Tenant compensation is a transfer, whereas delay, disruption and implementation consume value. For fixed-calendar monetary amounts or common risk-neutral expectations, let $D$ be the tenant's incremental delay/disruption cost and $K_t$ its incremental implementation expense. The tenant gains $F-D-K_t$, so bilateral participation requires

$$D+K_t\le F\le g-K_o,\qquad N_{\mathrm{joint}}=g-D-K_o-K_t,\tag{E6}$$

where $g$ is the selected matched operating-value increment. The transfer cancels from joint surplus. The historical 12→24 h mean yields the conditional upper endpoint $F\le\mathrm{US\$}\,33,050.09-K_o$; the tenant-side lower endpoint is unobserved. Participation alone does not prove incentive compatibility for a menu of private tenant types. The menu's IC constraints remain additional requirements. Charges already included in the settled operating account are not deducted again, and dispatch-dependent compensation must enter the optimisation and its certificate directly.

## Source notes for integration

- Table E1 and the fee calculations: `source_data/current/fig3_service_windows.csv`, rows 2–10, paired by seed and identical checkpoint SHA; independently recomputed by `economic_value_analysis.py`.
- Aggregate cross-check: `source_data/current/0927_adapted_window_contrasts.csv`, rows 2–4, and `figure_source/Fig1_reference_20261010_v3/service_windows.csv`.
- Independent 24 h ledger check: `source_data/grid_metrics_inputs_2025.npz`, SHA-256 `5be123c40f01b111cf6503b07882e703a4c2cc29a03dde3cfb7aa4321ce2cf72`.
- Eqs. (E1)–(E6) are mathematical definitions and consequences under the stated assumptions, not newly measured empirical quantities. The original ablation tables and module results are unaffected.
