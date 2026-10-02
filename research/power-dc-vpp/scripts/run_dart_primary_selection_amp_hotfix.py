"""Resume the formal DART v4 selection with narrow AMP overflow handling."""

from __future__ import annotations

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from dcvpp.experiments.dart_amp_overflow_compat import (  # noqa: E402
    amp_overflow_compatibility,
)
from dcvpp.experiments import dart_checkpoint_training as frozen  # noqa: E402
from dcvpp.experiments.dart_scheduler_failure_compat import (  # noqa: E402
    isolated_hybrid_scheduler_failures,
)
from run_dart_primary_selection import main  # noqa: E402


if __name__ == "__main__":
    ledger = (
        REPOSITORY_ROOT
        / "artifacts/dart_value_select/run_id=paper-dart-forecast-v4"
        / "forecast_oos/recovery_overlay/amp_overflow_compatibility.json"
    )
    with amp_overflow_compatibility(ledger_path=ledger):
        with isolated_hybrid_scheduler_failures(frozen):
            raise SystemExit(main())
