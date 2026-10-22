from __future__ import annotations

from typing import Any


PAPER_PREFLIGHT_READY = "PAPER_PREFLIGHT_READY"
DEMO_COMPLETE_PAPER_BLOCKED = "DEMO_COMPLETE_PAPER_BLOCKED"
TERMINAL_STATUSES = frozenset(
    {
        "COMPLETE_PAPER",
        "PARTIAL_PAPER",
        DEMO_COMPLETE_PAPER_BLOCKED,
        "FAILED",
    }
)


def classify_paper_preflight(timezone_gate: dict[str, Any]) -> dict[str, Any]:
    """Classify only the paper preflight boundary.

    Passing a timezone gate authorizes downstream work; it is not a paper
    experiment.  This function therefore cannot emit ``COMPLETE_PAPER`` or
    ``PARTIAL_PAPER``.  Those terminal states require separately evidenced
    real-data experiment results.
    """
    gate_status = str(timezone_gate.get("status", "")).lower()
    if gate_status == "blocked" and timezone_gate.get("code") == "BLOCKED_TIMEZONE_NLR300":
        return {
            "status": DEMO_COMPLETE_PAPER_BLOCKED,
            "terminal": True,
            "terminal_status": DEMO_COMPLETE_PAPER_BLOCKED,
            "paper_results_available": False,
            "paper_training_started": False,
            "reason": "mandatory NLR300 timezone gate blocked paper execution",
        }
    if gate_status == "passed":
        return {
            "status": PAPER_PREFLIGHT_READY,
            "terminal": False,
            "paper_results_available": False,
            "paper_training_started": False,
            "reason": (
                "NLR300 timezone gate passed; real H0-H6 experiments remain pending and "
                "no terminal paper status is yet truthful"
            ),
        }
    raise ValueError(f"unsupported NLR300 timezone-gate result: {timezone_gate!r}")


def assert_preflight_not_promoted_to_paper_result(status: dict[str, Any]) -> None:
    """Reject a ready preflight that has been mislabeled as a paper result."""
    if status.get("status") == PAPER_PREFLIGHT_READY:
        if status.get("terminal") is not False:
            raise ValueError("paper preflight readiness must be nonterminal")
        if status.get("paper_results_available") is not False:
            raise ValueError("paper preflight readiness cannot claim paper results")
        if status.get("paper_training_started") is not False:
            raise ValueError("paper preflight readiness cannot claim training")
        if "terminal_status" in status:
            raise ValueError("nonterminal paper preflight cannot carry terminal_status")
        if status.get("status") in TERMINAL_STATUSES:
            raise ValueError("preflight status cannot be a terminal paper status")
