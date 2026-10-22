# Executable service commitments and WorldMoE

Code and derived source data for **Executable service commitments bound the electricity-market value of computing flexibility**.

Author and contact: **Xiang Yi**, Northwest Minzu University, Lanzhou, Gansu, China. Email: **a1130575372@gmail.com**.

Version: **v1.4.0**, 22 October 2026. This is a research software archive associated with a manuscript; it does not imply journal acceptance or publication.

## Full frozen model inference in v1.4.0

The complete inference bundle now includes the W0/W1 backbone, frozen R2/APIC and calibration, three selected decision refinements, fitted normalization and physical observers, plus a portable one-decision loader and the original executable-service kernel. [Download the full inference asset](https://github.com/kalimustafa/executable-service-commitments-worldmoe/releases/tag/v1.4.0), then follow [the model card](model_release/MODEL_CARD.md) and [reproduction guide](model_release/REPRODUCE.md). All weight files are release assets rather than large Git blobs. The [manifest](model_release/model_manifest.json) records source and exported hashes; the [download note](model_release/DOWNLOAD.md) provides the archive checksum.

The full chain was loaded and executed on CPU for all three selected refinements using a synthetic input fixture. That test checks interfaces, APIC consistency, service feasibility and accounting; it is not a new annual performance measurement. The author-corrected display label for the third refinement is epoch 2; its original archive identifier remains epoch00, and the original tensor lineage is preserved. Full retraining and annual replay still require the provider inputs. The previous v1.3.0 release remains unchanged.

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

The retained nominal results are a three-refinement mean saving of approximately US$105,271, or 18.96% of the matched immediate-execution bill, and 79.39% of a US$132,605 certified ceiling. These are results from included records. Rebuilding the figures or summaries does **not** rerun training or independently validate the experiments.

Repository cleanup removed two superseded Figure 1 source directories and the old Figure 1 artwork they replaced. The direct figure-builder entry points now use the same final WorldMoE-centred builder as the reproduction wrapper. Paper results, provenance data, research code, selected checkpoints and current figures are retained. This cleanup adds no new figures.

## Research code and full replay requirements

The `research/` tree preserves the original sibling-directory names used by the source. The main final training entry is `R2_JointDecision_20260927/scripts/train_deep_rollout.py --continuous`; the final evaluator is `scripts/evaluate_scenario_recourse.py` with `--eta 0 --safety 0.9 --current-coupling --continuous-cost` and the selected seed/epoch. The final pairs are 20260920/6, 20260921/2 and 20260922/0. Training and replay must be run in a separate working copy with the original input archives configured.

The recorded experiment environment was Windows, Python 3.13, PyTorch 2.9 and an NVIDIA RTX 3080. Some archived training scripts use `msvcrt` and CUDA directly. The upstream `power-dc-vpp/pyproject.toml` describes an older package environment, not a fully pinned environment for every later research stage. Figure dependencies above are separate from the original GPU environment.

The historical `models/` directory contains the original decision-module checkpoint files. The v1.4.0 release asset adds the full inference dependency chain and a standalone one-decision example, with source/export hashes in `model_release/model_manifest.json`. It does not bundle multi-year feature caches, raw provider archives or all quarter-hour replay books. Full retraining and full-calendar replay remain conditional on obtaining and configuring those research inputs. Provider data remain subject to their source terms; no annual replay or retraining was performed as part of model packaging.

Main additional inputs used by the archived runners include `external_validation/data/forcing_2024.npz`, `forcing_2025.npz`, Frontier mapped arrivals, `R2_PAPER_INTEGRATION_20260923/results/predictions_*/`, frozen feature caches under `R2_JointDecision_20260927/cache/`, physical priors and the shared pretrained QPR2 initialization. Original provenance paths are retained as source identifiers, not as downloadable links. Local absolute paths must be configured when restoring these archives on another machine.

## Interpretation

The three refinements share pretrained initialization and frozen features. Their spread measures refinement variability. Historical model-policy cells retain their original controllers and calibration; their financial contrasts do not isolate neural architecture. The savings certificate is an upper opportunity bound and is not an executable oracle. The paper reports both beneficial and adverse grid metrics, feasibility failures, closing service obligations and the historical Frontier comparison.

The manuscript records training on 2021–2023, physical labels from 2022–2023, selection on 2024 and a 2025 test calendar under the author-specified chronology. This archive does not add a preregistration or an independently audited freeze chronology.

## Data sources and licensing

Underlying data are obtained from [NYISO](https://mis.nyiso.com/public/), [ISO New England](https://www.iso-ne.com/isoexpress/web/reports/pricing), [the NLR Eagle dataset](https://doi.org/10.7799/3023273), [NASA POWER](https://power.larc.nasa.gov/) and [the Frontier workload archive](https://jsspp.org/workload/index.php?page=frontier25). See the manuscript references and `source_data/references.json` for the exact vintages.

The MIT software licence is retained from the original project. It applies to original software; third-party software notices and provider data terms remain applicable. It does not relicense provider datasets or imply rights over third-party data. The derived source tables and model files are provided as research materials supporting the manuscript. Use `CITATION.cff` and the version tag when citing this release.
