# WorldMoE contract value revision and evidence package

Revision date: 31 October 2026.

This package accompanies the revised manuscript on contract rights and the economic value of data-centre flexibility. It restores the WorldMoE architecture and economic figures, completes the constructive admission-dependent feasibility proof, materialises the 24 h witness for recovered original admissions, and supplies the original nine historical policy trajectories with independent settlement and exact numerical feasibility records.

The current results are an annual 24 h perfect-information value interval of **US$130,931.31–134,437.85**, an exact feasible operating-value enclosure that rounds to **US$105,271.06**, and a conservatively rounded remaining annual opportunity interval of **US$25,660.24–29,166.79**. The operating policy captures approximately **78.30%–80.40%** of the bounded 24 h opportunity. These quantities use the same recovered admissions, reference account, physical inputs, service constraints and terminal convention. They describe the observed market calendar, not guaranteed future profit.

## Package layout

| Directory | Role |
|---|---|
| `manuscript/` | Final revised main manuscript and supplement. These are the current reading versions. |
| `figures/` | Restored WorldMoE architecture and revised economic figures, with the original manuscript palette and supplied editable or high-resolution assets. |
| `source_package/` | The preceding verification package preserved unchanged, including its source snapshot, certificate inputs, multiplier archives, surrogate schedules, proofs and historical verification records. Its old manuscript copies and older missing-file statements are provenance records; the current manuscript and the extensions below supersede those statements where recovery has resolved the gaps. |
| `proof_extension/` | S29, the completed constructive proof and its materialisation on recovered original admissions, together with four complete rational service witnesses and exact monetary enclosures. In the working directory this material originated under `revision/proof_extension/`; the published package places it directly under `proof_extension/`. |
| `historical_extension/` | S30, recovered original forcing and nine unmodified historical service books, complete trajectories, same-checkpoint source records, independent quarter-hour settlement, exact numerical repairs, and portable audit programs. |

The current evidence has three distinct components. The retained certificate package establishes conservative perfect-information value bounds. S29 establishes an admission-dependent feasible witness for every admission path in the stated reference-compatible domain and explicitly supplies a witness for the recovered original path. S30 restores the historical policy records and verifies their physical service and monetary accounts. None of these components is substituted for another.

## Main results and their interpretation

| Quantity | Current result | Interpretation |
|---|---:|---|
| 12 h perfect-information value ceiling | US$100,625.48 | Conservative upper bound under 12 h rights |
| 16 h perfect-information value ceiling | US$115,735.23 | Conservative upper bound under 16 h rights |
| 24 h perfect-information value ceiling | US$134,437.85 | Conservative upper bound under 24 h rights |
| Uniform constructive 24 h value floor | US$130,931.31 | Feasible-value guarantee over the stated reference-compatible admission domain |
| Minimum 12-to-24 h optimal rights premium | US$30,305.84 | Conservative difference between bounded optimal feasible-set values |
| Historical 12-to-24 h policy saving increase | US$33,050.09 | Same-checkpoint paired policy contrast supported by recovered original trajectories |
| 24 h operating saving | US$105,271.06 | Three-checkpoint historical saving; separate exact numerical repairs preserve this value to the nearest cent |
| Remaining 24 h operating opportunity | US$25,660.24–29,166.79 | Conservative residual relative to the 24 h perfect-information interval |
| Approximate opportunity capture | 78.30%–80.40% | Operating saving divided by the bounded 24 h opportunity |

The US$33,050.09 historical policy contrast and the US$30,305.84 minimum optimal-rights premium are different estimands. They are not two estimates of the same payoff. Fees and tenant-side costs, where considered in the retained economic scenarios, remain explicitly stated assumptions rather than observed contract prices.

## Recovered original data and separate numerical repairs

The recovered `historical_extension/inputs/forcing_2025.npz` includes the original `eagle_arrivals` array, with 35,040 quarter-hour intervals and four sites. Applying the archived earliest-service recurrence to these admissions reproduces the published reference array bit for bit. The physical and price inputs also match the released account bit for bit. Original file hashes and checkpoint identities are recorded in `historical_extension/source_manifest.json`.

The nine raw historical outputs cover 12 h, 16 h and 24 h for seeds 20260920, 20260921 and 20260922, using the respective selected checkpoint epochs 6, 2 and 0. They are preserved without alteration in `historical_extension/inputs/`. The historical results remain labelled **historical paired policy results with recovered trajectory verification**.

Raw binary64 schedules have tiny nonzero rational service residuals. They are not relabelled as exact symbolic witnesses. The independent exact repair program writes separate rational schedules to `historical_extension/results/exact_repaired_s*_w*.json.gz`, verifies all release, deadline, service-capacity, electrical-capacity and terminal conditions exactly, and bounds the monetary effect. The largest repair-cost bound across all nine runs is **US$1.65626764744 × 10⁻⁸**. The original historical files are not overwritten.

The exact repaired 24 h mean lies in **[US$105,271.061841632, US$105,271.061841633]**. The exact repaired 12-to-24 h mean contrast lies in **[US$33,050.092227443, US$33,050.092227444]**. These endpoints are directed rational enclosures, not confidence intervals. The reported monetary values remain unchanged at cent precision.

All historical policy and reference tails are zero. S29 additionally supplies complete exact 24 h opportunity witnesses for the recovered admissions; their true terminal backlog and signed terminal-anchor offsets are zero. The materialised opportunity saving is enclosed by **[US$130,931.31968120, US$130,931.31968149]**. The manuscript retains the more general uniform floor of US$130,931.31.

## Reproduction requirements

The new S29 and S30 programs require **Python and NumPy**. Exact arithmetic uses Python's standard-library integers and `fractions.Fraction`; compressed witness files use standard-library gzip and JSON. They require no model training, neural inference, external data download or optimisation solver.

The retained certificate verification scripts inside `source_package/` also import **SciPy**, even when verifying saved evidence without an LP solve. The original recorded environment and full regeneration commands are documented in `source_package/verification_20261031/certificate/README_NEW_CERTIFICATES.txt`. Solver-based regeneration is distinct from verifying the retained certificates.

Run the following commands from the extracted package root. Relative paths refer only to files inside this package.

### Materialise the admission-specific 24 h opportunity witness

```text
python proof_extension/materialize_actual_witness.py --source-package source_package --forcing historical_extension/inputs/forcing_2025.npz --output-dir proof_extension/reproduced_results
```

This reads the retained surrogate schedules and recovered original admissions, applies the exact constructive map, exports complete rational service and cumulative ledgers, and computes directed monetary enclosures. It uses only NumPy and the supplied package files. It does not call an optimiser. The published run is retained in `proof_extension/results/`.

### Audit the original historical service books and settlement

```text
python historical_extension/code/audit_historical_books.py
```

This reads only `historical_extension/inputs/` and writes the independent service, terminal, electrical and monetary evidence to `historical_extension/results/`. It includes all 35,040 quarter-hour audit arrays, exact accumulation of stored binary64 prefix residuals, annual settlement, monthly summaries and same-seed contrasts. It does not run the historical policy.

### Reconstruct the exact numerical historical witnesses

```text
python historical_extension/code/exact_repair_historical_books.py
```

This independently encodes the recovered input numbers as exact dyadic rationals, constructs separate feasible service paths, verifies the complete constraints and evaluates the exact original-account savings and repair-cost bounds. It also uses only `historical_extension/inputs/`, Python and NumPy. The result JSON records the exact rational values and integer-directed enclosure endpoints.

### Verify the retained annual value certificates without a new LP solve

```text
python source_package/verification_20261031/certificate/verify_saved_envelope.py --repo source_package/source_snapshot --inputs source_package/source_snapshot/source_data/grid_metrics_inputs_2025.npz --results source_package/verification_20261031/certificate/results
python source_package/verification_20261031/certificate/epigraph_certificate.py --repo source_package/source_snapshot --inputs source_package/source_snapshot/source_data/grid_metrics_inputs_2025.npz --output-dir source_package/verification_20261031/certificate/epigraph_results --verify-only
```

These retained programs require NumPy and SciPy. They evaluate the stored original-cost and epigraph certificates; they do not train the policy or recreate a missing historical solve. The unchanged source package retains the detailed finite-bound records, source semantics, exact arithmetic assumptions and scope of its earlier fixtures.

## Evidence boundaries and additional windows

The revised proof states **for every admissible arrival path, there exists an arrival-dependent feasible 24 h service schedule**. It does not assert one fixed schedule works for every path, nor that the hindsight construction is an information-adapted operating policy. Recovery of the original arrivals and historical books resolves the previous packaging gaps without changing this quantifier distinction.

No new training, new historical-policy execution or additional service-window experiment was performed for this revision. The proposed 8 h and 20 h extension can retain the existing **96-slot, 24 h predictive horizon**. A 36 h or 48 h extension requires an explicit longer queue and deadline representation plus a declared continuation rule. Existing Frontier workload results must not be mixed into the Eagle curve. `historical_extension/window_extension_protocol.txt` records this proposed follow-up; it contains no newly measured results.

The package does not claim recovery of the old 29 October hindsight assignment, verification of historical first-release information logs, real-hardware performance, a journal decision or submission. Original archived limitations remain attached to the evidence they describe; S29 and S30 identify precisely which missing-data and trajectory limitations have now been resolved.
