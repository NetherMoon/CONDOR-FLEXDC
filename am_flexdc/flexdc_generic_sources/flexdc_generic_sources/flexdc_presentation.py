"""Presentation-ready HTML helpers retained from the original notebooks.

The project notebooks used explicit HTML rather than relying on implicit
Pandas-Styler rendering.  This module keeps that behavior version-neutral and
prevents Colab from printing raw CSS as plain text.
"""
from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


BASE_CSS = """
:root{--bg:#eef2f7;--card:#fff;--ink:#111827;--muted:#475569;--line:#cbd5e1;
--good:#166534;--good-bg:#dcfce7;--bad:#991b1b;--bad-bg:#fee2e2;--accent:#4f46e5}
*{box-sizing:border-box}body{font-family:Inter,Arial,sans-serif;background:var(--bg);color:var(--ink);margin:0;padding:24px}
.report{max-width:1800px;margin:auto}.title{font-size:30px;font-weight:800;margin:0 0 8px}.subtitle{color:var(--muted);margin:0 0 24px}
.card{background:var(--card);border-radius:12px;padding:18px;margin:0 0 18px;box-shadow:0 1px 3px rgba(15,23,42,.12);overflow:auto}
h2{font-size:19px;margin:0 0 12px}.banner{background:linear-gradient(90deg,#111827,#312e81);color:#fff;border-radius:10px;padding:14px;margin:-2px -2px 14px}
table{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}th{background:#111827;color:white;text-align:left;padding:8px;position:sticky;top:0}
td{padding:7px;border:1px solid var(--line)}tr:nth-child(even){background:#f8fafc}.pass{color:var(--good);background:var(--good-bg);font-weight:700}.fail{color:var(--bad);background:var(--bad-bg);font-weight:700}
.note{padding:10px;border-left:4px solid var(--accent);background:#eef2ff;color:#312e81;margin:10px 0}.small{font-size:11px;color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}.metric{border:1px solid var(--line);border-radius:10px;padding:12px}.metric b{display:block;font-size:19px;margin-top:4px}
"""


def _value(value: Any, precision: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return "—"
        return f"{float(value):.{precision}g}"
    if isinstance(value, (list, tuple, np.ndarray, dict)):
        return html.escape(json.dumps(value.tolist() if isinstance(value, np.ndarray) else value))
    return html.escape(str(value))


def dataframe_html(
    frame: pd.DataFrame,
    *,
    title: str | None = None,
    precision: int = 6,
    max_rows: int | None = None,
    pass_columns: Sequence[str] = (),
) -> str:
    if frame is None or len(frame) == 0:
        body = "<div class='small'>No rows.</div>"
    else:
        sample = frame if max_rows is None else frame.head(max_rows)
        headers = "".join(f"<th>{html.escape(str(column))}</th>" for column in sample.columns)
        rows = []
        pass_set = set(pass_columns)
        for _, row in sample.iterrows():
            cells = []
            for column in sample.columns:
                value = row[column]
                css = ""
                if column in pass_set and pd.notna(value):
                    css = "pass" if bool(value) else "fail"
                cells.append(f"<td class='{css}'>{_value(value, precision)}</td>")
            rows.append("<tr>" + "".join(cells) + "</tr>")
        body = f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    heading = f"<h2>{html.escape(title)}</h2>" if title else ""
    return f"<section class='card'>{heading}{body}</section>"


def comparison_change(source: Any, destination: Any) -> tuple[str, float | None]:
    try:
        a, b = float(source), float(destination)
    except Exception:
        return "—", None
    if not (math.isfinite(a) and math.isfinite(b)):
        return "—", None
    if abs(a) < 1e-15:
        return ("0%" if abs(b) < 1e-15 else "↑ from 0"), None
    pct = 100.0 * (b - a) / abs(a)
    arrow = "↓" if pct < 0 else "↑" if pct > 0 else "→"
    return f"{arrow} {pct:+.2f}%", pct


def multi_point_comparison_html(
    frame: pd.DataFrame,
    *,
    title: str,
    point_column: str = "Point",
    model_column: str = "Model",
    metric_columns: Sequence[str] | None = None,
    baseline_model: str | None = None,
) -> str:
    if frame is None or len(frame) == 0:
        return dataframe_html(pd.DataFrame(), title=title)
    excluded = {point_column, model_column, "weights"}
    if metric_columns is None:
        metric_columns = [
            column for column in frame.columns
            if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
        ]
    sections = [f"<section class='card'><h2>{html.escape(title)}</h2>"]
    for point, group in frame.groupby(point_column, dropna=False):
        models = list(group[model_column].astype(str))
        baseline = baseline_model if baseline_model in models else models[0]
        base_row = group[group[model_column].astype(str) == baseline].iloc[0]
        headers = "<th>Metric</th>" + "".join(f"<th>{html.escape(model)}</th>" for model in models)
        headers += "".join(f"<th>Δ {html.escape(model)} vs {html.escape(baseline)}</th>" for model in models[1:])
        rows = []
        for metric in metric_columns:
            values = []
            for model in models:
                value = group[group[model_column].astype(str) == model].iloc[0][metric]
                values.append(f"<td>{_value(value)}</td>")
            changes = []
            for model in models[1:]:
                value = group[group[model_column].astype(str) == model].iloc[0][metric]
                text, pct = comparison_change(base_row[metric], value)
                css = "pass" if pct is not None and pct < 0 else "fail" if pct is not None and pct > 0 else ""
                changes.append(f"<td class='{css}'>{html.escape(text)}</td>")
            rows.append(f"<tr><td>{html.escape(metric)}</td>{''.join(values)}{''.join(changes)}</tr>")
        sections.append(
            f"<div class='banner'><b>{html.escape(str(point))}</b></div>"
            f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        )
    sections.append("</section>")
    return "".join(sections)


def build_report(
    *,
    title: str,
    subtitle: str = "",
    tables: Mapping[str, pd.DataFrame] = {},
    comparisons: Sequence[tuple[str, pd.DataFrame]] = (),
    notes: Sequence[str] = (),
) -> str:
    sections = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<style>{BASE_CSS}</style></head><body><main class='report'>",
        f"<h1 class='title'>{html.escape(title)}</h1>",
        f"<p class='subtitle'>{html.escape(subtitle)}</p>",
    ]
    for note in notes:
        sections.append(f"<div class='note'>{html.escape(note)}</div>")
    for name, frame in tables.items():
        sections.append(dataframe_html(frame, title=name, pass_columns=[c for c in frame.columns if "Pass" in str(c)]))
    for name, frame in comparisons:
        sections.append(multi_point_comparison_html(frame, title=name))
    sections.append("</main></body></html>")
    return "".join(sections)


def write_report(path: str | Path, html_text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")
    return path
