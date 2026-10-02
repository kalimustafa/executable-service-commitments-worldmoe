from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

from dcvpp.core import atomic_json
from dcvpp.data.io_utils import atomic_bytes


XCEL_RATE_BOOK_LANDING_URL = (
    "https://www.xcelenergy.com/company/rates_and_regulations/rates/rate_books?state=Colorado"
)
XCEL_CPP_INFORMATION_SHEET_URL = (
    "https://www.xcelenergy.com/staticfiles/xe-responsive/Programs%20and%20Rebates/Business/"
    "CO-Critical-Peak-Pricing-Info-Sheet.pdf"
)
XCEL_CURRENT_ENTIRE_TARIFF_DIRECT_URL = (
    "https://www.xcelenergy.com/staticfiles/xe-responsive/Company/Rates%20%26%20Regulations/"
    "Regulatory%20Filings/PSCo_Electric_Entire_Tariff.pdf"
)
XCEL_CURRENT_SUMMARY_VIEWER_URL = (
    "https://xcelnew.my.salesforce.com/sfc/p/1U0000011ttV/a/R300000RlUER/"
    "5G4ru_.jqcaA1xENmOJaXHZ1xgEeRfm5coulNeJ2Wz4"
)
# Version-pinned download resolved from the official Xcel/Salesforce viewer on
# 2026-08-10.  The landing-page label is checked before this file is accepted as
# the latest summary, so a later landing-page update cannot silently relabel it.
XCEL_CURRENT_SUMMARY_DIRECT_URL = (
    "https://xcelnew.file.force.com/sfc/dist/version/download/"
    "?oid=00D1U0000011ttV&ids=068R300000eKrM6"
    "&d=%2Fa%2FR300000RlUER%2F5G4ru_.jqcaA1xENmOJaXHZ1xgEeRfm5coulNeJ2Wz4"
    "&operationContext=DELIVERY&viewId="
)
XCEL_CURRENT_SUMMARY_EXPECTED_LABEL = "Summary of Electric Rates as of 07-01-26 (PDF)"
XCEL_OVERLAP_2024_SUMMARY_URL = (
    "https://www.xcelenergy.com/staticfiles/xe-responsive/Company/"
    "Rates%20%26%20Regulations/Electric_Summation_Sheet_All_Rates_05.01.2024.pdf"
)
XCEL_OVERLAP_2024_BUSINESS_RATES_URL = (
    "https://www.xcelenergy.com/staticfiles/xe-responsive/Company/"
    "Rates%20%26%20Regulations/CO-Business-Rates-Brochure.pdf"
)
XCEL_DEMAND_MANAGEMENT_SUMMARY_URL = (
    "https://www.xcelenergy.com/staticfiles/xe-responsive/"
    "Business%20Programs%20%26%20Rebates/CO%20DEMAND%20MANAGEMENT%20PROGRAM%20SUMMARY.pdf"
)
XCEL_CPP_LOCAL_RELATIVE_PATH = Path("data/raw/xcel/CO-Critical-Peak-Pricing-Info-Sheet.pdf")
XCEL_CURRENT_TARIFF_LOCAL_RELATIVE_PATH = Path("data/raw/xcel/PSCo_Electric_Entire_Tariff_current.pdf")
XCEL_CURRENT_SUMMARY_LOCAL_RELATIVE_PATH = Path(
    "data/raw/xcel/Xcel_CO_Electric_Rate_Summary_2026-07-01.pdf"
)
XCEL_OVERLAP_2024_SUMMARY_LOCAL_RELATIVE_PATH = Path(
    "data/raw/xcel/Electric_Summation_Sheet_All_Rates_2024-05-01.pdf"
)
XCEL_OVERLAP_2024_BUSINESS_RATES_LOCAL_RELATIVE_PATH = Path(
    "data/raw/xcel/CO-Business-Rates-Brochure_2024-05-01.pdf"
)
XCEL_DEMAND_MANAGEMENT_SUMMARY_LOCAL_RELATIVE_PATH = Path(
    "data/raw/xcel/CO-Demand-Management-Program-Summary_2023.pdf"
)
XCEL_REGISTRY_LOCAL_RELATIVE_PATH = Path("data/raw/xcel/source_registry.json")
XCEL_EXTRACTION_LOCAL_RELATIVE_PATH = Path("data/raw/xcel/extraction_status.json")
XCEL_PUBLIC_DATA_READY = "XCEL_PUBLIC_DATA_READY"
XCEL_PUBLIC_DATA_BLOCKED = "XCEL_PUBLIC_DATA_BLOCKED"
XCEL_SOURCE_SEMANTICS = {
    "evidence_role": "observed",
    "source_relationship": "third_party_referenced_public_document",
    "record_type": "referenced_official_tariff_or_program_rule",
    "measurement_relationship": "not_measured_by_this_study",
}


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._href: str | None = None
        self._text: list[str] = []
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        self._href = next((value for key, value in attrs if key.lower() == "href" and value), None)
        self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href is not None:
            label = " ".join("".join(self._text).split())
            self.links.append((label, self._href))
            self._href = None
            self._text = []


def _relevant_viewer_links(html: str) -> list[dict[str, str]]:
    parser = _LinkParser()
    parser.feed(html)
    selected: list[dict[str, str]] = []
    current_summary_seen = False
    for label, href in parser.links:
        lowered = label.lower()
        role: str | None = None
        if lowered == "electric rate book (pdf)":
            role = "current_electric_rate_book_viewer"
        elif lowered.startswith("summary of electric rates as of") and not current_summary_seen:
            role = "latest_summary_of_electric_rates_viewer"
            current_summary_seen = True
        elif lowered == "summary of electric rates as of 05-01-24 (pdf)":
            role = "overlap_era_2024_summary_link"
        elif lowered == "section 5 - rate schedules part 2 (pdf)":
            role = "rate_schedules_part_2_viewer"
        if role is not None:
            selected.append(
                {
                    "role": role,
                    "label": label,
                    "viewer_url": urljoin(XCEL_RATE_BOOK_LANDING_URL, href),
                }
            )
    return selected


def _pdf_response_looks_usable(response: Any) -> bool:
    content_type = str(response.headers.get("content-type", "")).lower()
    return response.status_code == 200 and (
        "application/pdf" in content_type or bytes(response.content[:5]) == b"%PDF-"
    )


def _download_official_pdf(
    repository_root: Path,
    *,
    client: Any,
    url: str,
    local_relative_path: Path,
    role: str,
    applicability_scope: str,
    document_date: str | None,
    timeout: tuple[int, int],
    stable_url: str | None = None,
    version_pin_status: str | None = None,
) -> dict[str, Any]:
    response = client.get(
        url,
        headers={"User-Agent": "power-dc-vpp/0.1 official-source-adapter"},
        timeout=timeout,
    )
    if not _pdf_response_looks_usable(response):
        raise ValueError(f"official Xcel PDF is unavailable or not a PDF: {url}")
    target = repository_root / local_relative_path
    atomic_bytes(target, response.content)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    record = {
        "role": role,
        **XCEL_SOURCE_SEMANTICS,
        "source_owner": "Xcel Energy / Public Service Company of Colorado",
        "stable_url": stable_url or url,
        "download_url": url,
        "filename": target.name,
        "local_path": local_relative_path.as_posix(),
        "content_type": response.headers.get("content-type"),
        "byte_count": len(response.content),
        "sha256": hashlib.sha256(response.content).hexdigest(),
        "retrieved_at_utc": retrieved_at,
        "document_date": document_date,
        "applicability_scope": applicability_scope,
        "training_target": False,
        "realized_response_evidence": False,
        "claim_boundary": (
            "official tariff/program document used as an exogenous rule input; "
            "not a meter observation, training label, event-call record, or proof of realized response"
        ),
        "automated": True,
    }
    if version_pin_status is not None:
        record["version_pin_status"] = version_pin_status
    return record


def _xcel_extraction_status(
    *,
    registry: dict[str, Any],
    current_tariff: dict[str, Any],
    current_summary: dict[str, Any],
    overlap_business_rates: dict[str, Any],
    demand_management: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return a compact, claim-bounded extraction of verified tariff fields.

    Page anchors below refer to PDF page numbers in the pinned local documents.
    Individual tariff-sheet effective dates are preserved separately from the
    retrieval date of the current whole-tariff snapshot.
    """

    schedule_common = {
        "minimum_annual_measured_demand_kw": 25,
        "required_interval_metering": True,
        "minimum_trailing_12_month_load_factor": 0.30,
        "excluded_programs": ["ISOC", "PPRP"],
        "extraction_status": "verified_from_official_tariff_text",
    }
    return {
        "schema_version": "xcel_extraction_status_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **XCEL_SOURCE_SEMANTICS,
        "training_target": False,
        "realized_response_evidence": False,
        "documents": {
            "current_entire_tariff_snapshot": {
                "local_path": current_tariff["local_path"],
                "sha256": current_tariff["sha256"],
                "retrieved_at_utc": current_tariff["retrieved_at_utc"],
                "document_effective_date": "varies_by_tariff_sheet",
                "extraction_status": "verified_selected_sheets_only",
            },
            "current_rate_summary": {
                "local_path": current_summary["local_path"],
                "sha256": current_summary["sha256"],
                "document_date": "2026-07-01",
                "applicability_scope": "current_2026_document_snapshot",
                "extraction_status": "verified_selected_large_business_reference_fields",
            },
            "overlap_era_business_rates": {
                "local_path": overlap_business_rates["local_path"],
                "sha256": overlap_business_rates["sha256"],
                "document_date": "2024-05-01",
                "applicability_scope": "overlap_era_2024_only",
                "extraction_status": "verified_selected_adjustments",
            },
            "demand_management_snapshot": (
                {
                    "local_path": demand_management["local_path"],
                    "sha256": demand_management["sha256"],
                    "document_date": "2023",
                    "applicability_scope": "dated_program_context_only",
                    "extraction_status": "verified_program-summary fields",
                }
                if demand_management is not None
                else {
                    "extraction_status": "acquisition_blocked",
                    "claim_boundary": "no fields inferred",
                }
            ),
        },
        "large_business_schedule_candidates": [
            {
                **schedule_common,
                "schedule": "SG-CPP",
                "service_voltage": "secondary",
                "base_rate_sheet_effective_date": "2024-05-01",
                "charges": {
                    "service_and_facility_usd_per_month": 88.07,
                    "distribution_demand_usd_per_kw_month": 9.36,
                    "generation_transmission_demand_usd_per_kw_month": 8.02,
                    "cpp_energy_usd_per_kwh": 1.44,
                    "non_cpp_energy_usd_per_kwh": 0.00853,
                },
                "source_anchor": {"pdf_page": 65, "tariff_sheet": "45"},
            },
            {
                **schedule_common,
                "schedule": "PG-CPP",
                "service_voltage": "primary",
                "base_rate_sheet_effective_date": "2024-05-01",
                "charges": {
                    "service_and_facility_usd_per_month": 894.00,
                    "distribution_demand_usd_per_kw_month": 5.61,
                    "generation_transmission_demand_usd_per_kw_month": 8.89,
                    "cpp_energy_usd_per_kwh": 1.40,
                    "non_cpp_energy_usd_per_kwh": 0.00701,
                },
                "source_anchor": {"pdf_page": 99, "tariff_sheet": "56"},
            },
            {
                **schedule_common,
                "schedule": "TG-CPP",
                "service_voltage": "transmission",
                "base_rate_sheet_effective_date": "2024-05-01",
                "charges": {
                    "service_and_facility_usd_per_month": None,
                    "service_and_facility_note": "customer specific",
                    "generation_transmission_demand_usd_per_kw_month": 7.93,
                    "cpp_energy_usd_per_kwh": 1.35,
                    "non_cpp_energy_usd_per_kwh": 0.00767,
                },
                "source_anchor": {
                    "pdf_pages": [120, 121],
                    "tariff_sheets": ["71", "71A"],
                },
            },
        ],
        "current_2026_large_business_reference_rates": [
            {
                "schedule": "SG",
                "variant": "flat_demand_charge",
                "effective_date": "2026-07-01",
                "total_rates_including_listed_adjustments": {
                    "service_and_facility_usd_per_month": 91.79,
                    "distribution_demand_usd_per_kw_month": 9.66,
                    "winter_generation_transmission_demand_usd_per_kw_month": 19.68,
                    "summer_generation_transmission_demand_usd_per_kw_month": 25.91,
                    "energy_usd_per_kwh": 0.04011,
                },
                "source_anchor": {"pdf_page": 1},
                "extraction_status": "verified_text_and_visual",
            },
            {
                "schedule": "PG",
                "effective_date": "2026-07-01",
                "total_rates_including_listed_adjustments": {
                    "service_and_facility_usd_per_month": 923.91,
                    "distribution_demand_usd_per_kw_month": 5.79,
                    "winter_generation_transmission_demand_usd_per_kw_month": 19.50,
                    "summer_generation_transmission_demand_usd_per_kw_month": 25.30,
                    "on_peak_energy_usd_per_kwh": 0.06556,
                    "off_peak_energy_usd_per_kwh": 0.03304,
                },
                "source_anchor": {"pdf_page": 2},
                "extraction_status": "verified_text_and_visual",
            },
            {
                "schedule": "TG",
                "effective_date": "2026-07-01",
                "total_rates_including_listed_adjustments": {
                    "service_and_facility_usd_per_month": None,
                    "service_and_facility_note": "customer specific",
                    "winter_generation_transmission_demand_usd_per_kw_month": 13.19,
                    "summer_generation_transmission_demand_usd_per_kw_month": 19.03,
                    "on_peak_energy_usd_per_kwh": 0.06511,
                    "off_peak_energy_usd_per_kwh": 0.03322,
                },
                "source_anchor": {"pdf_page": 2},
                "extraction_status": "verified_text_and_visual",
            },
        ],
        "cpp_program_rules": {
            "eligible_schedule_candidates": ["SG-CPP", "PG-CPP", "TG-CPP"],
            "event_duration_hours": 4,
            "event_window_local": "12:00-20:00 Mountain Time",
            "maximum_events_per_calendar_year": 15,
            "maximum_events_per_day": 1,
            "minimum_notice_hours": 22,
            "source_anchors": [
                {"schedule": "SG-CPP", "pdf_pages": [66, 67], "tariff_sheets": ["45A", "45B"]},
                {"schedule": "PG-CPP", "pdf_pages": [100, 101], "tariff_sheets": ["56A", "56B"]},
                {"schedule": "TG-CPP", "pdf_page": 122, "tariff_sheet": "71B"},
            ],
            "extraction_status": "verified_from_official_tariff_text",
            "claim_boundary": (
                "published eligibility and event constraints only; no Xcel event calls, "
                "dispatch, customer response, or savings were observed"
            ),
        },
        "overlap_era_2024_adjustments": {
            "effective_date": "2024-05-01",
            "applicability_scope": "overlap_era_2024_only",
            "source_anchor": {"pdf_page": 1},
            "demand_side_management_usd_per_kw_month": {"SG": 1.71, "PG": 1.58, "TG": 1.39},
            "purchased_capacity_usd_per_kw_month": {"SG": 1.31, "PG": 1.20, "TG": 1.06},
            "transmission_cost_usd_per_kw_month": {"SG": 0.70, "PG": 0.64, "TG": 0.56},
            "transportation_electrification_usd_per_kw_month": {"SG-CPP": 0.34, "PG-CPP": 0.26, "TG-CPP": 0.01},
            "extraction_status": "verified_from_official_2024_business-rate_summary",
        },
        "observed_xcel_event_calls": {
            "extraction_status": "not_available_in_acquired_documents",
            "record_count": 0,
            "claim_boundary": "do not infer calls from tariff eligibility or event caps",
        },
        "registry_status": registry["status"],
    }


def collect_xcel_public_source_registry(
    repository_root: Path,
    *,
    session: requests.Session | None = None,
    timeout: tuple[int, int] = (20, 120),
) -> dict[str, Any]:
    """Collect official Xcel source provenance without treating it as a label.

    The downloaded CPP information sheet is preserved as a dated public-rule
    source. Current and overlap-era documents remain separate.  Every acquired
    document is a third-party referenced official publication and never a
    training target or evidence of a realized demand response.
    """
    local_pdf = repository_root / XCEL_CPP_LOCAL_RELATIVE_PATH
    if not local_pdf.exists():
        raise FileNotFoundError(local_pdf)
    content = local_pdf.read_bytes()
    if content[:5] != b"%PDF-":
        raise ValueError(f"Xcel CPP source is not a PDF: {local_pdf}")

    registry_path = repository_root / XCEL_REGISTRY_LOCAL_RELATIVE_PATH
    registry: dict[str, Any] = {
        "status": XCEL_PUBLIC_DATA_BLOCKED,
        "source_owner": "Xcel Energy / Public Service Company of Colorado",
        "role": "exogenous_local_tariff_and_demand_response_rule",
        **XCEL_SOURCE_SEMANTICS,
        "training_target": False,
        "realized_response_evidence": False,
        "landing_url": XCEL_RATE_BOOK_LANDING_URL,
        "probed_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [
            {
                "role": "cpp_program_information_sheet_snapshot",
                **XCEL_SOURCE_SEMANTICS,
                "stable_url": XCEL_CPP_INFORMATION_SHEET_URL,
                "local_path": XCEL_CPP_LOCAL_RELATIVE_PATH.as_posix(),
                "byte_count": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "content_type": "application/pdf",
                "claim_boundary": "program-rule context only; not proof of current rates or realized response",
                "training_target": False,
                "realized_response_evidence": False,
            }
        ],
        "documents": [],
        "current_viewer_urls": [],
        "current_direct_pdf": None,
        "current_summary_pdf": None,
        "overlap_era_2024_pdfs": [],
        "demand_management_pdf": None,
        "extraction_status_path": XCEL_EXTRACTION_LOCAL_RELATIVE_PATH.as_posix(),
        "blocker": None,
    }

    client: Any = session or requests
    try:
        landing = client.get(
            XCEL_RATE_BOOK_LANDING_URL,
            headers={"User-Agent": "power-dc-vpp/0.1 official-source-adapter"},
            timeout=timeout,
        )
        landing.raise_for_status()
        registry["current_viewer_urls"] = _relevant_viewer_links(landing.text)
    except (requests.RequestException, OSError, ValueError, AttributeError) as exc:
        registry["blocker"] = f"official rate-book landing page could not be automated: {exc}"
        atomic_json(registry_path, registry)
        return registry

    try:
        latest_summary = next(
            (
                item
                for item in registry["current_viewer_urls"]
                if item["role"] == "latest_summary_of_electric_rates_viewer"
            ),
            None,
        )
        if latest_summary is None or latest_summary["label"] != XCEL_CURRENT_SUMMARY_EXPECTED_LABEL:
            raise ValueError(
                "landing-page latest summary no longer matches the pinned 2026-07-01 official PDF"
            )

        current_tariff = _download_official_pdf(
            repository_root,
            client=client,
            url=XCEL_CURRENT_ENTIRE_TARIFF_DIRECT_URL,
            local_relative_path=XCEL_CURRENT_TARIFF_LOCAL_RELATIVE_PATH,
            role="current_entire_tariff_snapshot_including_section_5",
            applicability_scope="current_document_snapshot; individual sheet effective dates vary",
            document_date=None,
            timeout=timeout,
        )
        current_summary = _download_official_pdf(
            repository_root,
            client=client,
            url=XCEL_CURRENT_SUMMARY_DIRECT_URL,
            local_relative_path=XCEL_CURRENT_SUMMARY_LOCAL_RELATIVE_PATH,
            role="current_summary_of_electric_rates",
            applicability_scope="current_2026_document_snapshot",
            document_date="2026-07-01",
            stable_url=XCEL_CURRENT_SUMMARY_VIEWER_URL,
            version_pin_status=(
                "official Salesforce content-version snapshot resolved from the publisher viewer; "
                "not asserted to be an immutable publisher version"
            ),
            timeout=timeout,
        )
        overlap_summary = _download_official_pdf(
            repository_root,
            client=client,
            url=XCEL_OVERLAP_2024_SUMMARY_URL,
            local_relative_path=XCEL_OVERLAP_2024_SUMMARY_LOCAL_RELATIVE_PATH,
            role="overlap_era_all_schedule_rate_summary",
            applicability_scope="overlap_era_2024_only",
            document_date="2024-05-01",
            timeout=timeout,
        )
        overlap_business = _download_official_pdf(
            repository_root,
            client=client,
            url=XCEL_OVERLAP_2024_BUSINESS_RATES_URL,
            local_relative_path=XCEL_OVERLAP_2024_BUSINESS_RATES_LOCAL_RELATIVE_PATH,
            role="overlap_era_commercial_and_industrial_rate_summary",
            applicability_scope="overlap_era_2024_only",
            document_date="2024-05-01",
            timeout=timeout,
        )

        demand_management: dict[str, Any] | None = None
        try:
            demand_management = _download_official_pdf(
                repository_root,
                client=client,
                url=XCEL_DEMAND_MANAGEMENT_SUMMARY_URL,
                local_relative_path=XCEL_DEMAND_MANAGEMENT_SUMMARY_LOCAL_RELATIVE_PATH,
                role="dated_demand_management_program_summary",
                applicability_scope="dated_program_context_only",
                document_date="2023",
                timeout=timeout,
            )
        except (requests.RequestException, OSError, ValueError, AttributeError) as exc:
            registry["optional_document_warnings"] = [
                f"dated demand-management summary could not be acquired: {exc}"
            ]

        registry["current_direct_pdf"] = current_tariff
        registry["current_summary_pdf"] = current_summary
        registry["overlap_era_2024_pdfs"] = [overlap_summary, overlap_business]
        registry["demand_management_pdf"] = demand_management
        registry["sources"].extend(
            [
                current_summary,
                overlap_summary,
                overlap_business,
                *([demand_management] if demand_management is not None else []),
            ]
        )
        registry["documents"] = [
            current_tariff,
            current_summary,
            overlap_summary,
            overlap_business,
            *([demand_management] if demand_management is not None else []),
            {
                "role": "current_section_5_part_2_viewer",
                **XCEL_SOURCE_SEMANTICS,
                "viewer_url": next(
                    (
                        item["viewer_url"]
                        for item in registry["current_viewer_urls"]
                        if item["role"] == "rate_schedules_part_2_viewer"
                    ),
                    None,
                ),
                "content_covered_by": current_tariff["local_path"],
                "acquisition_status": "covered_by_downloaded_current_entire_tariff",
                "training_target": False,
                "realized_response_evidence": False,
            },
        ]
        registry["status"] = XCEL_PUBLIC_DATA_READY
        registry["blocker"] = None
        extraction = _xcel_extraction_status(
            registry=registry,
            current_tariff=current_tariff,
            current_summary=current_summary,
            overlap_business_rates=overlap_business,
            demand_management=demand_management,
        )
        atomic_json(repository_root / XCEL_EXTRACTION_LOCAL_RELATIVE_PATH, extraction)
        atomic_json(registry_path, registry)
        return registry
    except (requests.RequestException, OSError, ValueError, AttributeError) as exc:
        registry["blocker"] = f"current official Xcel direct PDF could not be automated: {exc}"
        atomic_json(registry_path, registry)
        return registry
