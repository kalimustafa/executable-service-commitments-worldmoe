# Executable service commitments and WorldMoE

Code and derived source data for **Executable service commitments bound the electricity-market value of computing flexibility**.

Author and contact: **Xiang Yi**, Northwest Minzu University, Lanzhou, Gansu, China. Email: **1130575372@gmail.com**.

Version: **v1.0.0**, 2 October 2026. This is a research software archive associated with a manuscript; it does not imply journal acceptance or publication.

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
| `figure_source/` | Figure builders and archived Figure 1 artwork |
| `figures/` | Figures used by the submission, with submission Extended Data numbering |

## Reproduce the figures and headline summaries

Use Python 3.11 or newer with the following figure-only dependencies. No GPU or raw market database is needed for these steps.

```bash
python -m pip install -r requirements-figures.txt
python tools/verify_release.py
python tools/reproduce_figures.py
```

The second command checks the release hashes and recalculates the nominal saving, saving rate and certified capture fraction from the included result tables. The third writes figures to `reproduced_figures/`, preserving the released figures. Figure 1 is archived artwork rather than a numerical calculation. Submission Extended Data Figure 1 is workload feasibility and Figure 2 is monthly savings; the original builder used the opposite numbering, which the wrapper maps explicitly.

The retained nominal results are a three-refinement mean saving of approximately US$105,271, or 18.96% of the matched immediate-execution bill, and 79.39% of a US$132,605 certified ceiling. These are results from included records. Rebuilding the figures or summaries does **not** rerun training or independently validate the experiments.

## Research code and full replay requirements

The `research/` tree preserves the original sibling-directory names used by the source. The main final training entry is `R2_JointDecision_20260927/scripts/train_deep_rollout.py --continuous`; the final evaluator is `scripts/evaluate_scenario_recourse.py` with `--eta 0 --safety 0.9 --current-coupling --continuous-cost` and the selected seed/epoch. The final pairs are 20260920/6, 20260921/2 and 20260922/0. Training and replay must be run in a separate working copy with the original input archives configured.

The recorded experiment environment was Windows, Python 3.13, PyTorch 2.9 and an NVIDIA RTX 3080. Some archived training scripts use `msvcrt` and CUDA directly. The upstream `power-dc-vpp/pyproject.toml` describes an older package environment, not a fully pinned environment for every later research stage. Figure dependencies above are separate from the original GPU environment.

This release includes the selected **decision-module** checkpoints, but does not bundle all frozen W0/W1/R2 backbone weights, pretrained initialization checkpoints, multi-year feature caches, complete raw provider archives or all quarter-hour replay books. The model files alone therefore do not provide a standalone inference service. Their roles and hashes are in `source_data/current_provenance.json`. Full retraining and full-calendar replay are not self-contained in this release; access to the unbundled research inputs can be requested from the contact above. Provider data remain subject to their source terms. No claim is made that full replay was rerun during packaging.

Main additional inputs used by the archived runners include `external_validation/data/forcing_2024.npz`, `forcing_2025.npz`, Frontier mapped arrivals, `R2_PAPER_INTEGRATION_20260923/results/predictions_*/`, frozen feature caches under `R2_JointDecision_20260927/cache/`, physical priors and the shared pretrained QPR2 initialization. Original provenance paths are retained as source identifiers, not as downloadable links. Local absolute paths must be configured when restoring these archives on another machine.

## Interpretation

The three refinements share pretrained initialization and frozen features. Their spread measures refinement variability. Historical model-policy cells retain their original controllers and calibration; their financial contrasts do not isolate neural architecture. The savings certificate is an upper opportunity bound and is not an executable oracle. The paper reports both beneficial and adverse grid metrics, feasibility failures, closing service obligations and the historical Frontier comparison.

The manuscript records training on 2021–2023, physical labels from 2022–2023, selection on 2024 and a 2025 test calendar under the author-specified chronology. This archive does not add a preregistration or an independently audited freeze chronology.

## Data sources and licensing

Underlying data are obtained from [NYISO](https://mis.nyiso.com/public/), [ISO New England](https://www.iso-ne.com/isoexpress/web/reports/pricing), [the NLR Eagle dataset](https://doi.org/10.7799/3023273), [NASA POWER](https://power.larc.nasa.gov/) and [the Frontier workload archive](https://jsspp.org/workload/index.php?page=frontier25). See the manuscript references and `source_data/references.json` for the exact vintages.

The MIT software licence is retained from the original project. It applies to original software; third-party software notices and provider data terms remain applicable. It does not relicense provider datasets or imply rights over third-party data. The derived source tables and model files are provided as research materials supporting the manuscript. Use `CITATION.cff` and the version tag when citing this release.
