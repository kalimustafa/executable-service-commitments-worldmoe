# WorldMoE-centred Figure 1

Run `python build_fig1.py --output-dir OUTPUT_FOLDER` with Python, NumPy, pandas and Matplotlib. The directory contains all drawing inputs. The source reference artwork is not redistributed.

## Reference style analysis and application

The supplied reference is a landscape architecture figure with a dominant nested model container, aligned horizontal substructures, short input/module/output connections and three vertically stacked result containers. A white background, thin dark outlines, restrained pale fills and consistent text hierarchy make the nesting legible. Connections use horizontal or vertical gutters; detail is expanded inside one local container instead of repeated as explanatory prose. Small inset graphics express internal structure and outputs.

Here the WorldMoE architecture is the principal object. The left portfolio, observations, queue and service conditions are inputs, not a sequence of preparation stages. Inputs branch to their relevant model components; service constraints also connect directly to the external executor. Five aligned internal rows show encoding, world layers, APIC, attention and slot-cost decoding. Their connecting lines express actual computational dependencies with additional evidence inputs, not cross-validation folds or five parallel models. The right-hand result boxes show cost, diurnal imports and service windows.

Shallow top/side folds, offset matrix sheets, expert stacks, attention layers and a contracting decoder provide the requested three-dimensional folded appearance. Folds have no quantitative interpretation. Colours use the supplied eight-card palette: #B6B3D6, #CFCCE3, #D5D3DE, #D5D1D1, #F6DFD6, #F8B2A2, #F1837A, #E9687A, with white tints and neutral text/edges.

## Evidence and interpretation

- `figure_inputs.json`: configured four-site portfolio and retained nominal replay results. Site A = WEST/Buffalo, B = CAPITL/Albany, C = N.Y.C./New York and D = LONGIL/Long Island. These are model locations, not surveyed facility addresses.
- `northeast_states.geojson` and `geography_source.json`: sourced Census state boundaries, with New York and New England shown for geographic context. No transmission network or observed power transfer is inferred.
- `comparison_models.csv`: five archived forecast/control pipelines, including valued closing liabilities. WorldMoE and previous WorldMoE average three fixed refinements; the other pipelines retain one run. These differences do not isolate architecture effects.
- `comparison_diurnal_imports.csv`: full-year means grouped by local New York hour, not a selected illustrative day.
- `service_windows.csv`: three-refinement outcomes under 12, 16 and 24 h service windows. Bars on the input side depict allowed duration; the numerical output box reports archived savings and certified capture. The same released work is retained.
- `additional_input_manifest.json` and `input_manifest.json`: retained data aggregation and source hashes. This drawing update runs no training or dispatch replay.

## Architecture scope

The frozen world representation uses three scales, 40 structural tokens of width 256 and ten interaction layers. Sparse MoE occurs in even layers: one shared expert, seven generally routed experts with Top-2 selection and one conditionally gated tail expert. Highlighted routes are schematic, not measured routing weights. The APIC market path carries DA/DART information, ordered RT quantiles and the market latent. Physics, queue and physical state remain separate inputs to their respective modules. Attention uses two temporal layers. The contracting cost decoder emits five raw cost-distribution parameters, with a separate rho branch and RT evidence; it does not emit a 96-step action sequence. A deterministic executor applies service constraints and selects one current 15-minute action.

Outputs include PNG, editable-text SVG, PDF and a separate enlarged architecture detail. No scientific observations are added by the graphical shapes or layer folds.
