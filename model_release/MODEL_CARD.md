# Frozen integrated WorldMoE

This package exposes the full pretrained inference chain for the data-centre service-and-settlement study: W0/W1 market–physical representation, frozen R2 and Two-Gate APIC, the selected decision refinements, Gaussian partial-moment costs, current-action search, and conserved service obligations. It contains real trained parameters exported from the registered local checkpoints. The inference export removes optimizer and random-generator state while preserving every loaded model tensor.

## Model lineage

W0 and W1 are successive training stages of the same sparse-MoE backbone. The final W1 state is sufficient for inference; loading a separate W0 final model is neither required nor meaningful. The released backbone has 59,156,961 parameters, with seven general experts, Top-2 routing, and a conditional tail residual. Its original W1 checkpoint hash and W0 parent hash are recorded in `model_manifest.json`. The learned typed tokenizer, observation decoder and inference buffers are part of this state.

The frozen `StructuredR2_seed20260920_epoch10.pt` contains the R2 profile residual, world adapter, native affine calibration and APIC quantile/coupling buffers. Three `ArchitectedR2QPHead` decision refinements each contain 227,527 parameters. Their 2024-selected archive identifiers are seed/epoch 20260920/6, 20260921/2 and 20260922/0. The author corrected the manuscript display label for the third refinement to epoch 2 on 16 October 2026 without changing checkpoint bytes or numerical experiments. Accordingly this package preserves `epoch00` in the filename and hash lineage and records `author_display_epoch=2`. It does not substitute the distinct archive `epoch02` tensor file.

The stated protocol fits market components using 2021–2023, physical components using 2022–2023, selects on 2024 and evaluates on the independent 2025 calendar. The author supplied a chronology correction stating that the 2025 calendar remained sealed through final freeze; this is an author statement, not independently reconstructed historical access evidence. No fitting or checkpoint selection was performed during this release.

## Contents and inference

`weights/` contains the backbone, R2/APIC, and all three selected decision states. `configs/` includes train-fitted world and market normalization, feature order, physical constraints, seasonal PV envelopes, causal observer parameters, and selection records. No text tokenizer is needed: WorldMoE uses a learned typed numerical-event tokenizer. `upstream/`, `src/` and `vendor/` preserve original implementation code. `run_frozen_worldmoe.py` loads the full chain, accepts numeric inputs with an availability contract, executes one action using the released `select_action` kernel, and checks the service ledger. `SHA256SUMS.txt` identifies package files; `model_manifest.json` maps inference exports back to original checkpoint hashes.

The example is a deterministic synthetic interface fixture. It demonstrates loading and executable accounting and provides no estimate of economic performance. Its input graph is deliberately simple, and its non-provider observations are constructed from published training-summary statistics. Real forecasts require valid numerical and graph inputs in the documented schema. The full annual research runners, preprocessing and original derived result tables remain in the repository's `research/` and `source_data/` trees.

The portable CPU example uses FP32 computation and the original FP16 feature-bridge quantization. Historical world-cache extraction used CUDA BF16 computation with FP16 storage. Exact annual cache reproduction should use that recorded precision path; CPU interface success does not establish bitwise parity of historical predictions.

## Information and fallback rules

Every supplied observation group must have `available_at <= decision_at`; post-DA anchors must also satisfy this check. This runtime check cannot establish that a historical provider value is the vintage available at that time. Strict point-in-time claims require a separate source-vintage audit. Ex-post DA/RT settlement fields are read only after action selection.

Missing world context is handled by the pretrained world-availability gate. Unavailable market forecasts use the fitted zone historical RT centre and zero latent context. Missing issued physical context uses the training seasonal PV median and fitted PUE reference. Lagged physical observations condition the current slot. Historical data acquisition must retain missingness masks, correct timezone conversion, and actual graph features. The minimal fixture uses a 24-hour grid; original annual profile preprocessing supports 23/24/25-hour local dates with masks. Do not force a DST day into the minimal fixture to claim full-year reproduction.

## Intended use and limitations

This is research software for constrained computation scheduling and energy-value evaluation. The same differentiable 16-cohort training implementation and deterministic runtime action kernel are available as inspectable source. The demonstration does not retrain the model, replay the annual experiment, validate a production deployment, or prove a historical source vintage. Full replay requires the original provider archives, forcing, arrivals, and issue-time adapters described in `REPRODUCE.md`. Annual service closure and terminal liabilities must be evaluated by those annual accounting runners, not inferred from a single decision.

Original software is under the repository MIT license. This release includes authors' trained tensors and derived normalization/observer parameters as research materials under the same license. It includes no raw market/weather/workload archives, third-party pretrained base models, credentials, or personally identifying records. Provider data licenses continue to govern separately obtained data. No additional third-party pretrained-model license restriction was identified in the inspected model lineage.
