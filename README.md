# Executable service commitments and WorldMoE

Code and derived source data for **Contract rights and the economic value of data centre flexibility**, with the earlier WorldMoE research archive retained.

Author and contact: **Xiang Yi**, Northwest Minzu University, Lanzhou, Gansu, China. Email: **mohamwdkali@gmail.com**.

Current manuscript and evidence revision: **31 October 2026**, on top of software archive v1.3.0. This is a research archive associated with a manuscript; it does not imply journal acceptance or publication.

## Current manuscript and completed seven-window evidence

- [Current main manuscript](revisions/20261031_seven_windows/manuscript/EnergyEconomics_Seven_Window_Revision_20261031.docx), **18 pages**, with six readable figures in the original purple/pink palette and a concise economic literature comparison.
- [Current Supplementary Information](revisions/20261031_seven_windows/manuscript/EnergyEconomics_Seven_Window_Supplement_20261031.docx), **65 pages**, with S31–S32 and twelve retained original images. All current figure captions are short: at most 25 words in the main text and 23 in the supplement.
- [Complete evidence guide and reproduction commands](revisions/20261031_seven_windows/README.md).
- [Current figures and drawing source](revisions/20261031_seven_windows/figures/).
- [Previous recovered-admission revision](revisions/20261031_contract_value/), retained as a separate evidence vintage.

The extension completes **15 new annual policy evaluations** with the original three frozen checkpoints, including three 24 h anchors that reproduce the historical complete books and settlement trajectories bit for bit. It also computes **28 site–deadline certificates** for 8/12/16/20/24/36/48 h on the same recovered actual admissions and account. No new training or fitting was performed.

| Deadline | Mean policy cash saving US$ | Service status | Certified PI opportunity US$ |
|---:|---:|---|---:|
| 8 h | 50,735.34 | Substantive failure; about 3.01 IT MWh late | 78,751.60–78,882.41 |
| 12 h | 72,220.97 | Exact feasible counterpart | 97,639.33–97,796.41 |
| 16 h | 86,038.55 | Exact feasible counterpart | 113,119.89–113,305.19 |
| 20 h | 98,199.02 | Exact feasible counterpart | 124,528.17–124,733.99 |
| 24 h | 105,271.06 | Exact feasible counterpart | 132,376.82–132,598.65 |
| 36 h | 114,251.12 | Exact feasible counterpart | 141,996.61–142,256.41 |
| 48 h | 123,101.20 | Exact feasible counterpart | 152,798.28–153,070.23 |

The 8 h cash-only outcomes are retained without repair and are excluded from feasible policy value, capture and headroom. The other new executions have separately saved exact numerical witnesses. The 36/48 h implementation retains a fixed 24 h neural horizon and complete longer queues, with an explicitly declared deterministic continuation; it does not claim a new 48 h neural forecast or conservative price bound.

The current actual-admission 24 h remaining opportunity is **US$27,105.76–27,327.58**, capture is **79.39%–79.53%**, and the optimal 12→24 h rights increment is **US$34,580.41–34,959.31**. The historical same-checkpoint policy increment remains **US$33,050.09**. These are distinct economic objects. Earlier bounds uniform over all reference-compatible admissions remain valid robustness results in S27–S29.

Certified incremental opportunity per extra deadline hour rises from **US$783.16–823.30** over 24→36 h to **US$878.48–922.81** over 36→48 h, while the policy increments fall from **US$748.34** to **US$737.51**. Thus the operating curve alone does not identify the shape of the contractual opportunity curve. These results describe the observed simulation calendar, not promised future profit.

The earlier software-archive instructions below retain their historical scope and figure labels. The current manuscripts and S31–S32 supersede the previous status that additional windows had not yet been evaluated.

## Contents

| Path | Contents |
| --- | --- |
| `research/R2_JointDecision_20260927/` | Final decision module, differentiable allocation, training and annual evaluation runners; internal family name CdfR2 |
| `research/ServiceCommitment_1to5_20260927/` | Conserved service ledger, settlement, reservation execution, electrical constraints and certificate code |
| `research/ExecutableServiceCommitments_Revision_20260928/` | Current service-window, signal, mixed-deadline, PV and Frontier extensions |
| `research/zero_shot_isone_20260928/` | Frozen cross-market replay and input adapters |
| `research/g3_realistic_utilization_20260929/` | Mixed rigid and flexible workload experiments |
| `research/power-dc-vpp/` and other research folders | Upstream backbone, physical model and original supporting implementations |
| `models/` | The three selected final decision-module checkpoints, with original bytes and SHA-256 hashes |
| `source_data/` | Derived tabular results, daily accounts, source manifests and retained certificates |
| `figure_source/` | Current figure builders, drawing inputs and paper-result derivation scripts |
| `figures/` | Figures used by the submission, with submission Extended Data numbering |

## Reproduce the figures and headline summaries

Use Python 3.11 or newer with the following figure-only dependencies. No GPU or raw market database is needed for these steps.

```bash
python -m pip install -r requirements-figures.txt
python tools/verify_release.py
python tools/reproduce_figures.py
```

The second command checks the release hashes and recalculates the nominal saving and certified capture fraction from the included result tables. The third writes figures to `reproduced_figures/`, preserving the released figures. Figure 1 is a Python-drawn mechanism figure with configured model-site coordinates on Census state outlines, archived three-refinement outcomes and full-year mean diurnal imports. The updated landscape figure centres the WorldMoE architecture: portfolio, market/weather observations, queue and service conditions enter as parallel inputs, followed by slot-cost decoding and a separate deterministic executor. Layer folds, stacked matrices and expert blocks depict structure rather than numerical observations. Three output groups retain measured 12/16/24 h outcomes and five archived pipeline comparisons, using the author's original eight-colour palette. A separate enlarged model architecture is included as `figures/main/WorldMoE_architecture_detail.{png,svg,pdf}`. Self-contained drawing source and inputs are in `figure_source/Fig1_reference_20261010_v3/`. Submission Extended Data Figure 1 is workload feasibility and Figure 2 is monthly savings; the original builder used the opposite numbering, which the wrapper maps explicitly.

The earlier figure archive records a three-refinement mean saving of approximately US$105,271, or 18.96% of the matched immediate-execution bill, and 79.39% of a US$132,605 certified ceiling. Those older ceiling and capture labels describe the earlier evidence vintage; the current opportunity interval and capture calculation are reported above. Rebuilding the historical figures or summaries does **not** rerun training or independently validate the experiments.

Repository cleanup removed two superseded Figure 1 source directories and the old Figure 1 artwork they replaced. The direct figure-builder entry points now use the same final WorldMoE-centred builder as the reproduction wrapper. Paper results, provenance data, research code, selected checkpoints and current figures are retained. This cleanup adds no new figures.

## Research code and full replay requirements

The `research/` tree preserves the original sibling-directory names used by the source. The main final training entry is `R2_JointDecision_20260927/scripts/train_deep_rollout.py --continuous`; the final evaluator is `scripts/evaluate_scenario_recourse.py` with `--eta 0 --safety 0.9 --current-coupling --continuous-cost` and the selected seed/epoch. The final pairs are 20260920/6, 20260921/2 and 20260922/0. Training and replay must be run in a separate working copy with the original input archives configured.

The recorded experiment environment was Windows, Python 3.13, PyTorch 2.9 and an NVIDIA RTX 3080. Some archived training scripts use `msvcrt` and CUDA directly. The upstream `power-dc-vpp/pyproject.toml` describes an older package environment, not a fully pinned environment for every later research stage. Figure dependencies above are separate from the original GPU environment.

The original v1.3 release includes the selected **decision-module** checkpoints, but does not bundle all frozen W0/W1/R2 backbone weights, pretrained initialization checkpoints, multi-year feature caches, complete raw provider archives or all quarter-hour replay books. The model files alone therefore do not provide a standalone inference service. Their roles and hashes are in `source_data/current_provenance.json`. In that original release, full retraining and full-calendar replay are not self-contained; access to the unbundled research inputs can be requested from the contact above. Provider data remain subject to their source terms. No claim is made that full replay was rerun during packaging.

Main additional inputs used by the archived runners include `external_validation/data/forcing_2024.npz`, `forcing_2025.npz`, Frontier mapped arrivals, `R2_PAPER_INTEGRATION_20260923/results/predictions_*/`, frozen feature caches under `R2_JointDecision_20260927/cache/`, physical priors and the shared pretrained QPR2 initialization. Original provenance paths are retained as source identifiers, not as downloadable links. Local absolute paths must be configured when restoring these archives on another machine.

## Interpretation

The three refinements share pretrained initialization and frozen features. Their spread measures refinement variability. Historical model-policy cells retain their original controllers and calibration; their financial contrasts do not isolate neural architecture. The savings certificate is an upper opportunity bound and is not an executable oracle. The paper reports both beneficial and adverse grid metrics, feasibility failures, closing service obligations and the historical Frontier comparison.

The manuscript records training on 2021–2023, physical labels from 2022–2023, selection on 2024 and a 2025 test calendar under the author-specified chronology. This archive does not add a preregistration or an independently audited freeze chronology.

## Data sources and licensing

Underlying data are obtained from [NYISO](https://mis.nyiso.com/public/), [ISO New England](https://www.iso-ne.com/isoexpress/web/reports/pricing), [the NLR Eagle dataset](https://doi.org/10.7799/3023273), [NASA POWER](https://power.larc.nasa.gov/) and [the Frontier workload archive](https://jsspp.org/workload/index.php?page=frontier25). See the manuscript references and `source_data/references.json` for the exact vintages.

The MIT software licence is retained from the original project. It applies to original software; third-party software notices and provider data terms remain applicable. It does not relicense provider datasets or imply rights over third-party data. The derived source tables and model files are provided as research materials supporting the manuscript. Use `CITATION.cff` and the version tag when citing this release.
