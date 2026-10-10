# Figure 1 professional framework revision

Render the complete figure and an enlarged architecture detail:

```bash
python build_fig1.py --output-dir outputs
```

Requires Python, Matplotlib, NumPy and pandas. `professional_architecture.py` draws the nested WorldMoE framework as vector objects. All numerical inputs are included, so rendering requires no network or training environment.

## Quantitative sources

`service_windows.csv`: three-refinement means, min/max and service outcomes at 12, 16 and 24 h. Bar lengths encode contractual windows, not fabricated execution traces.

`comparison_models.csv`: final WorldMoE, previous WorldMoE point, LightGBM Recent point, RASC-ME point and immediate execution on the matched 2025 24 h account. Historical point policies retain their original controllers and include closing-work valuation. These are full-pipeline comparisons, not an isolated architecture experiment.

`comparison_diurnal_imports.csv`: annual mean portfolio imports by local clock hour. Final and previous WorldMoE average three retained refinements; other pipelines have one retained run. Historical trajectories were reconstructed from archived compute books and matched physical forcing, then aggregated; no training or dispatch rerun was performed. The reconstructed annual energy agrees with the archived summary for every pipeline. Source paths, hashes and calculations are in `additional_input_manifest.json`.

`figure_inputs.json` and `input_manifest.json` retain the nominal configuration and earlier evidence. Geographic inputs are US Census state outlines and configured modelled site coordinates. The intermarket arrow means frozen-model transfer, not electricity transfer.

## Architecture and palette

The frozen backbone uses 10 interaction blocks, five sparse MoE sublayers, an always-active shared expert, Top-2 routing among seven general experts, and a conditional tail branch. Highlighted expert cells are an illustration of the Top-2 mechanism, not measured gating assignments. APIC price quantiles bypass queue conditioning and enter the cost decoder separately. The two-stage temporal context and 96 workload queries follow the current decision-module implementation.

The diagram summarizes major paths; the full implementation and Supplementary Table 16 provide the local residual, encoder and precision details. Source: `power-dc-vpp/src/dcvpp/vpp_world_moe/config.py`, `sparse_moe.py`, and `R2_JointDecision_20260927/src/models/coupled_rollout.py` plus `continuous_decision.py`.

`palette.json` records the eight printed colour-card hex values supplied by the author. Series use those values; group fills use white tints, and labels/lines use neutral dark grey.
