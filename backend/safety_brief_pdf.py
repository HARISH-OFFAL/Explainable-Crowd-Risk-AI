"""Readable, reproducible PDF safety briefs for approved communications."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fpdf import FPDF


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FONT_CANDIDATES = (
    os.getenv("CROWDGUARD_FONT_PATH", ""),
    str(PROJECT_ROOT / "assets" / "fonts" / "NotoSansTamil-Regular.ttf"),
    str(PROJECT_ROOT / "assets" / "fonts" / "Nirmala.ttf"),
    r"C:\Windows\Fonts\Nirmala.ttc",
    r"C:\Windows\Fonts\Nirmala.ttf",
)


def _value(value: Any) -> str:
    return "Not available" if value is None or value == "" else str(value)


def _label(value: Any) -> str:
    return _value(value).replace("_", " ").title()


def _font_path() -> str:
    for candidate in FONT_CANDIDATES:
        if candidate and Path(candidate).is_file():
            return candidate
    raise RuntimeError("No Unicode/Tamil font found; set CROWDGUARD_FONT_PATH to a Tamil-capable TTF font.")


def _flatten_notes(value: Any, prefix: str = "") -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, dict):
        return [note for key, item in value.items() for note in _flatten_notes(item, f"{_label(key)}: ")]
    if isinstance(value, list):
        return [note for item in value for note in _flatten_notes(item, prefix)]
    return [f"{prefix}{value}"]


def _summary(source: dict[str, Any], plan: dict[str, Any]) -> str:
    parts = ["This brief records a situation observed in the monitoring snapshot"]
    if source.get("timestamp_sec") is not None:
        parts.append(f"at video timestamp {source['timestamp_sec']} seconds")
    if source.get("current_people") is not None:
        parts.append(f"with a recorded people count of {source['current_people']}")
    if source.get("target_zone"):
        parts.append(f"in {_label(source['target_zone'])}")
    return ". ".join(parts) + f". Recorded priority: {_value(plan.get('priority'))}."


def _response_text(plan: dict[str, Any], source: dict[str, Any]) -> str:
    action = _label(source.get("recommended_action") or plan.get("intent"))
    explanation = (source.get("recommended_response") or {}).get("explanation") or {}
    detail = explanation.get("why") or explanation.get("what") or "No additional response detail was recorded in the approved plan."
    return f"Approved response: {action}. {_value(detail)}"


class SafetyBriefPDF(FPDF):
    def header(self):
        self.set_fill_color(10, 54, 71)
        self.rect(0, 0, self.w, 10, "F")
        self.set_y(16)

    def footer(self):
        self.set_y(-13)
        self.set_font("Brief", size=8)
        self.set_text_color(90, 105, 112)
        self.cell(0, 6, f"CrowdGuard Safety Brief  |  Page {self.page_no()}/{{nb}}", align="R")


def _section(pdf: SafetyBriefPDF, title: str):
    if pdf.get_y() > 255:
        pdf.add_page()
    pdf.ln(3)
    pdf.set_text_color(10, 54, 71)
    pdf.set_font("Brief", size=12)
    pdf.cell(0, 7, title)
    pdf.ln(8)


def _card(pdf: SafetyBriefPDF, rows: list[tuple[str, Any]]):
    start = pdf.get_y()
    pdf.set_text_color(30, 45, 52)
    for label, value in rows:
        pdf.set_font("Brief", size=8)
        pdf.set_text_color(74, 95, 103)
        pdf.cell(45, 6, label)
        pdf.set_font("Brief", size=9)
        pdf.set_text_color(30, 45, 52)
        pdf.multi_cell(0, 6, _value(value), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(205, 222, 223)
    pdf.rect(10, start - 1, pdf.w - 20, pdf.get_y() - start + 2, "D")


def build_safety_brief_pdf(event: Any, plan: dict[str, Any], message: Any, *, audience: str, language: str, generated_at: str | None = None) -> bytes:
    pdf = SafetyBriefPDF(format="A4")
    pdf.alias_nb_pages()
    pdf.add_font("Brief", fname=_font_path())
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    source = plan.get("source_state") or {}
    event_name = getattr(event, "event_name", None) or f"Event {event.id}"
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()

    pdf.set_text_color(10, 54, 71)
    pdf.set_font("Brief", size=22)
    pdf.multi_cell(0, 11, "CrowdGuard Safety Brief", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Brief", size=13)
    pdf.set_text_color(50, 75, 82)
    pdf.multi_cell(0, 8, event_name, new_x="LMARGIN", new_y="NEXT")

    _section(pdf, "Quick Summary")
    pdf.set_text_color(30, 45, 52)
    pdf.set_font("Brief", size=10)
    pdf.multi_cell(0, 6, _summary(source, plan), new_x="LMARGIN", new_y="NEXT")

    _section(pdf, "Situation Overview")
    counts = source.get("zone_counts") or {}
    _card(pdf, [("Recorded people", source.get("current_people")), ("Concern zone", _label(source.get("target_zone"))), ("Density", source.get("density")), ("Priority", plan.get("priority")), ("Contributing factors", "; ".join(_flatten_notes(source.get("main_driver"))) or None), ("Zone counts", ", ".join(f"{_label(k)}: {v}" for k, v in counts.items()) or None)])

    _section(pdf, "Approved Communication")
    _card(pdf, [("Audience", _label(audience)), ("Language", "Tamil" if language == "ta" else "English"), ("Approved message", message)])

    _section(pdf, "Recommended Response")
    pdf.set_text_color(30, 45, 52)
    pdf.set_font("Brief", size=10)
    pdf.multi_cell(0, 6, _response_text(plan, source), new_x="LMARGIN", new_y="NEXT")

    _section(pdf, "Key Notes")
    notes = ["The observations above are from the recorded monitoring snapshot; they are not current live conditions.", f"Recorded video timestamp: {_value(source.get('timestamp_sec'))} seconds."]
    notes.extend(_flatten_notes((source.get("recommended_response") or {}).get("explanation")))
    pdf.set_font("Brief", size=9)
    for note in dict.fromkeys(notes):
        pdf.cell(5, 6, "•")
        pdf.multi_cell(0, 6, note, new_x="LMARGIN", new_y="NEXT")

    _section(pdf, "Record Details")
    _card(pdf, [("Event ID", event.id), ("Monitoring session", source.get("monitoring_session_id")), ("Video timestamp", f"{_value(source.get('timestamp_sec'))} seconds"), ("Plan ID", plan.get("communication_plan_id")), ("Approval status", plan.get("status") or "APPROVED"), ("Generated", generated_at)])
    return bytes(pdf.output())


def safety_brief_filename(event_id: int, plan_id: Any) -> str:
    return f"CrowdGuard_Safety_Brief_Event_{event_id}_Plan_{plan_id}.pdf"
