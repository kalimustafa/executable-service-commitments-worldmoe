# Figure 1 mechanism and measured outcomes

Render with Python, Matplotlib, NumPy and pandas:

```bash
python build_fig1.py --output-dir outputs
```

The Python script draws all facility, mechanism, map and chart elements. The user-supplied images were visual references only. No generated bitmap reference is embedded.

`figure_inputs.json` contains model configuration and archived three-refinement outcomes. `annual_diurnal_imports.csv` contains full-year local-clock-hour averages calculated from all 35,040 quarter-hours. `input_manifest.json` records original input paths and hashes. The saved data are sufficient for rendering; `prepare_inputs.py` regenerates those inputs only when the original research workspace is present.

The map uses US Census Bureau 2024 cartographic state boundaries. Site coordinates are configured model locations, not observed data-centre addresses. The dashed intermarket arrow denotes frozen-model transfer, not electrical power transfer. Work boxes illustrate conservation and deadline-constrained shifting; they are not a fabricated numeric workload trajectory. Financial and grid values come from the actual archived replay results.

Outputs are PNG (500 dpi), editable-text SVG, vector PDF and a preview PNG.
