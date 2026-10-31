# Frozen policy evaluation runtime

The included `frozen_runtime/` supplies the archived Python modules, three selected checkpoints, causal observer, seasonal PV envelope, frozen market/latent cache, physical-prior cache, original 2025 admissions/forcing, physical constraints and selection records used by the new evaluations. No training dataset download or new model fitting is required. The archived code is copied without modifying the original working directory.

The observed evaluation environment is Python 3.13.5 with NumPy 2.1.3, PyTorch 2.9.0+cu126, Numba 0.61.0 and Pandas 2.2.3. Evaluation runs on the CPU; Torch, BLAS and OpenMP are restricted to one thread per process. The GPU build suffix records the installed package, not GPU inference. Standard-library modules provide the process controller, hashes and JSON. The independent exact service and monetary audit supplied separately requires only Python and NumPy.

From the extracted package root, this command reruns one 48 h checkpoint into a separate output directory while retaining the published outputs:

```text
python code/run_portable.py --fresh-output reproduced_policy --year 2025 --window 48 --seed 20260920 --cdf-epoch 6 --eta 0 --safety 0.9 --current-coupling --continuous-cost
```

Use windows `8`, `12`, `16`, `20`, `24`, `36` or `48`. The selected seed/epoch pairs are `20260920/6`, `20260921/2` and `20260922/0`. Reuse the same output directory to collect different runs. Omitting `--fresh-output` uses the package's `results/` directory and prints an existing completed result instead of overwriting it. A per-run lock waits for an already active run of the same name, avoiding duplicate execution. The numerical outputs, complete books and execution manifest are saved under the selected output root.

`code/run_suite.py` records the actual three-chain local orchestration used in this revision; it defaults to the original workspace engine. `code/run_extra36.py` and `code/run_extra48.py` record the additional three-process long-window acceleration. The 48 h accelerator starts after all three accelerated 36 h runs finish, keeping at most six single-thread inference processes active. The shared run lock prevents duplicate work when the original chain reaches an accelerated run. The portable entry point above uses the included frozen runtime directly and is the appropriate command outside the original workspace.

`frozen_source_dependencies.json` records the critical source/cache/checkpoint hashes. `code/execution_versions.json` identifies the minor runtime-source versions used during execution: the initial anchor metadata, corrected metadata, and a concurrency-lock addition. These changes affect provenance wording and process coordination; the numerical controller extension is the same. The original source runner and derivation script are retained. The three 24 h anchors are bitwise identical to the original historical complete books and all stored trajectory arrays, as recorded in `results/anchor_comparison.json`.

The long-window method is specified in `PROTOCOL.md` and `S32_window_extension_methods.md`. The fixed historical median-price continuation is a declared policy rule, not a conservative financial bound. The year-end early-completion target is stricter than some contractual deadlines and is distinguished from the original contractual deadline used by independent audits.
