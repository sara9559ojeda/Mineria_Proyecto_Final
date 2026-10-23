"""Streamlit app: Mall Customers segmentation with K-Means.

Loads the pipeline trained in ``proyecto_final_kmeans.ipynb`` and lets the user
explore the segments, classify a new customer and review how K was chosen.
Run with: ``streamlit run app.py``
"""
from html import escape
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "kmeans_segmentation.joblib"
DATA_PATH = BASE_DIR / "Mall_Customers.csv"

COLUMN_NAMES = {
    "CustomerID": "customer_id",
    "Gender": "gender",
    "Age": "age",
    "Annual Income (k$)": "annual_income",
    "Spending Score (1-100)": "spending_score",
}

TABS = ["Segmentos", "Clasificar cliente", "Selección de K"]
TABS_KEY = "main_tabs"

# Design tokens shared by the CSS, the HTML components and the Plotly charts.
# The chart surface is the page background so the validated categorical palette holds.
TOKENS = {
    "surface": "#fcfcfb",
    "surface-muted": "#f2f1ed",
    "ink": "#141413",
    "ink-secondary": "#52514e",
    "muted": "#6f6e68",
    "border": "#e3e2dc",
    "grid": "#ecebe6",
    "axis": "#c3c2b7",
    "accent": "#2a78d6",
    "accent-strong": "#1c5cab",
    "good": "#0ca30c",
    "warning": "#c98500",
}
FONT_STACK = 'Inter, system-ui, -apple-system, "Segoe UI", sans-serif'

# Marker shape is the secondary encoding of each segment (identity never relies on color alone)
PROFILE_SYMBOLS = {
    "Clientes de alto valor": "diamond",
    "Clientes promedio": "circle",
    "Clientes de bajo valor": "triangle-down",
    "Clientes con potencial": "square",
    "Compradores impulsivos": "triangle-up",
}

# st.html strips SVG, so icons are glyphs inside CSS-drawn status dots
ICON_GLYPHS = {"check": "✓", "alert": "!"}

TONE_COLORS = {"good": TOKENS["good"], "warning": TOKENS["warning"], "neutral": TOKENS["muted"]}

PAGE_CSS = """
:root {
  --space-1: 4px; --space-2: 8px; --space-3: 12px; --space-4: 16px;
  --space-5: 24px; --space-6: 32px; --space-7: 48px; --space-8: 64px;
  --radius: 8px;
  --font-mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
}

/* Layout */
.block-container { max-width: 1240px; padding: var(--space-6) var(--space-5) var(--space-8); }
header[data-testid="stHeader"] { background: transparent; }

/* Page header */
.topline { display: flex; justify-content: space-between; align-items: center; gap: var(--space-3); flex-wrap: wrap; }
.eyebrow { font-size: 12px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
.page-title {
  font-size: clamp(1.875rem, 3.4vw, 2.625rem); font-weight: 650; line-height: 1.1; letter-spacing: -0.022em;
  color: var(--ink); margin: var(--space-4) 0 var(--space-2);
}
.page-lead { font-size: 1.0625rem; line-height: 1.55; color: var(--ink-secondary); max-width: 66ch; margin: 0; }
.nowrap { white-space: nowrap; }

/* Badges */
.badge {
  display: inline-flex; align-items: center; gap: 6px; min-height: 26px; padding: 2px 10px;
  border: 1px solid var(--border); border-radius: 999px; background: var(--surface);
  font-size: 13px; line-height: 1.3; color: var(--ink);
}
.badge-mono { font-family: var(--font-mono); font-size: 12px; }
.status-icon {
  display: inline-flex; align-items: center; justify-content: center; flex: none;
  width: 16px; height: 16px; border-radius: 50%; color: #ffffff; font-size: 10px; font-weight: 700; line-height: 1;
}

/* Segment markers: the same shapes the charts use */
.marker { display: inline-block; flex: none; }
.marker-circle { clip-path: circle(42% at 50% 50%); }
.marker-square { clip-path: inset(12%); }
.marker-diamond { clip-path: polygon(50% 0, 100% 50%, 50% 100%, 0 50%); }
.marker-triangle-up { clip-path: polygon(50% 4%, 100% 92%, 0 92%); }
.marker-triangle-down { clip-path: polygon(0 8%, 100% 8%, 50% 96%); }

/* KPI strip */
.kpi-strip {
  display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
  border: 1px solid var(--border); border-radius: var(--radius); margin: var(--space-6) 0 var(--space-5);
}
.kpi { padding: var(--space-4) var(--space-5); }
.kpi + .kpi { border-left: 1px solid var(--border); }
.kpi-label { font-size: 13px; color: var(--muted); }
.kpi-value { font-size: 1.75rem; font-weight: 650; letter-spacing: -0.01em; line-height: 1.25; color: var(--ink); margin-top: 2px; }
.kpi-note { font-size: 13px; color: var(--ink-secondary); margin-top: 2px; }

/* Panels and sections */
.panel-title { font-size: 1.0625rem; font-weight: 600; color: var(--ink); margin: 0; }
.panel-description { font-size: 14px; line-height: 1.5; color: var(--ink-secondary); margin: 4px 0 0; max-width: 72ch; }
.section-title { font-size: 1.375rem; font-weight: 650; letter-spacing: -0.012em; color: var(--ink); margin: var(--space-6) 0 0; }
.section-lead { font-size: 15px; line-height: 1.55; color: var(--ink-secondary); margin: 4px 0 0; max-width: 72ch; }
.field-help { font-size: 13px; line-height: 1.5; color: var(--muted); margin: 0; }

/* Segment list (legend + summary); adapts to the width of its own panel */
.segment-list { container-type: inline-size; }
.segment-row {
  display: grid; grid-template-columns: 14px minmax(0, 1fr) auto; column-gap: 10px; row-gap: 6px;
  align-items: center; padding: var(--space-3) 0; border-top: 1px solid var(--border);
}
.segment-row:first-child { border-top: none; padding-top: var(--space-1); }
.segment-name { font-size: 15px; font-weight: 600; color: var(--ink); }
.segment-count { font-size: 14px; font-weight: 600; color: var(--ink); text-align: right; font-variant-numeric: tabular-nums; }
.segment-share { font-weight: 400; color: var(--muted); }
.segment-meta { grid-column: 2 / 4; font-size: 13px; color: var(--ink-secondary); font-variant-numeric: tabular-nums; }
.share-track { grid-column: 2 / 4; height: 4px; border-radius: 2px; background: var(--grid); overflow: hidden; }
.share-fill { height: 100%; border-radius: 2px; }
@container (max-width: 300px) {
  .segment-row { grid-template-columns: 14px minmax(0, 1fr); }
  .segment-count, .share-track, .segment-meta { grid-column: 2; text-align: left; }
}

/* Compact legend */
.legend { display: flex; flex-wrap: wrap; gap: var(--space-2) var(--space-5); margin-top: var(--space-2); }
.legend-item { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; color: var(--ink-secondary); }

/* Profile directory */
.profiles { margin: var(--space-4) 0 var(--space-5); border-bottom: 1px solid var(--border); }
.profile {
  display: grid; grid-template-columns: minmax(200px, 0.9fr) minmax(0, 1.1fr) minmax(0, 1.5fr);
  gap: var(--space-5); padding: var(--space-5) 0; border-top: 1px solid var(--border);
}
.profile-name { display: flex; align-items: center; gap: 8px; font-size: 15px; font-weight: 600; color: var(--ink); }
.profile-stats { font-size: 13px; line-height: 1.7; color: var(--ink-secondary); margin-top: 6px; font-variant-numeric: tabular-nums; }
.label {
  font-size: 12px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted);
  margin: 0 0 4px;
}
.profile-text { font-size: 14.5px; line-height: 1.55; color: var(--ink); margin: 0; }

/* Criteria strip (choice of K) */
.criteria {
  display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
  border: 1px solid var(--border); border-radius: var(--radius); margin: var(--space-5) 0;
}
.criterion { padding: var(--space-4) var(--space-5); }
.criterion + .criterion { border-left: 1px solid var(--border); }
.criterion-head { display: flex; align-items: center; gap: 6px; font-size: 13px; color: var(--muted); }
.criterion-value { font-size: 1.375rem; font-weight: 650; line-height: 1.3; color: var(--ink); margin-top: 4px; font-variant-numeric: tabular-nums; }
.criterion-note { font-size: 13px; line-height: 1.5; color: var(--ink-secondary); margin-top: 2px; }

/* Classification result */
.result-title {
  display: flex; align-items: center; gap: 10px; margin: 4px 0 var(--space-3);
  font-size: clamp(1.5rem, 2.6vw, 1.875rem); font-weight: 650; letter-spacing: -0.015em; line-height: 1.2; color: var(--ink);
}
.result-badges { display: flex; flex-wrap: wrap; gap: var(--space-2); margin-bottom: var(--space-5); }
.result-grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: var(--space-5); }
.result-section { margin-top: var(--space-5); padding-top: var(--space-4); border-top: 1px solid var(--border); }
.distance-row {
  display: grid; grid-template-columns: minmax(0, 220px) minmax(40px, 1fr) 44px; align-items: center;
  gap: 10px; padding: 4px 0; font-size: 13px;
}
.distance-name { display: flex; align-items: center; gap: 6px; min-width: 0; color: var(--ink-secondary); }
.distance-name span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.distance-track { height: 6px; border-radius: 3px; background: var(--grid); overflow: hidden; }
.distance-fill { height: 100%; border-radius: 3px; background: var(--axis); }
.distance-value { text-align: right; font-family: var(--font-mono); font-size: 12px; color: var(--ink-secondary); }
.distance-row.is-nearest .distance-name { color: var(--ink); font-weight: 600; }
.distance-row.is-nearest .distance-fill { background: var(--ink); }
.distance-row.is-nearest .distance-value { color: var(--ink); font-weight: 500; }

/* Callouts */
.callout {
  display: flex; gap: 10px; align-items: flex-start; margin-top: var(--space-4); padding: var(--space-3) var(--space-4);
  border: 1px solid var(--border); border-radius: var(--radius); background: var(--surface-muted);
  font-size: 14px; line-height: 1.5; color: var(--ink);
}
.callout .status-icon { margin-top: 2px; }

/* Error state */
.empty-state { max-width: 640px; margin: var(--space-8) auto; padding: var(--space-6); border: 1px solid var(--border); border-radius: var(--radius); }
.empty-state h2 { font-size: 1.375rem; font-weight: 650; margin: 0 0 var(--space-2); color: var(--ink); }
.empty-state p { font-size: 15px; line-height: 1.55; color: var(--ink-secondary); margin: 0 0 var(--space-3); }
.empty-state code { font-family: var(--font-mono); font-size: 13px; }

/* Footer */
.app-footer {
  display: flex; justify-content: space-between; gap: var(--space-2) var(--space-5); flex-wrap: wrap;
  margin-top: var(--space-8); padding-top: var(--space-4); border-top: 1px solid var(--border);
  font-size: 13px; color: var(--muted);
}
.app-footer code { font-family: var(--font-mono); font-size: 12px; color: var(--ink-secondary); background: none; padding: 0; }

/* Native tabs */
.st-key-main_tabs [role="tablist"] { gap: var(--space-6); border-bottom: 1px solid var(--border); }
.st-key-main_tabs [role="tab"] { padding: 12px 0; }
.st-key-main_tabs [role="tab"] p { font-size: 15px; font-weight: 600; }
.st-key-main_tabs [role="tab"][aria-selected="false"] p { color: var(--ink-secondary); }

/* Responsive */
@media (max-width: 900px) {
  .kpi-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .kpi:nth-child(3) { border-left: none; }
  .kpi:nth-child(n + 3) { border-top: 1px solid var(--border); }
  .profile { grid-template-columns: minmax(0, 1fr); gap: var(--space-3); }
}
@media (max-width: 760px) {
  .criteria { grid-template-columns: minmax(0, 1fr); }
  .criterion + .criterion { border-left: none; border-top: 1px solid var(--border); }
  .result-grid { grid-template-columns: minmax(0, 1fr); }
  .st-key-main_tabs [role="tablist"] { gap: var(--space-5); }
}
@media (max-width: 520px) {
  .block-container { padding: var(--space-5) var(--space-4) var(--space-7); }
  .kpi { padding: var(--space-3) var(--space-4); }
  .kpi-value { font-size: 1.5rem; }
  .distance-row { grid-template-columns: minmax(0, 1fr) 64px 40px; }
  .st-key-main_tabs [role="tablist"] { gap: var(--space-4); }
  .st-key-main_tabs [role="tab"] p { font-size: 14px; }
}
@media (max-width: 360px) {
  .kpi-strip { grid-template-columns: minmax(0, 1fr); }
  .kpi + .kpi { border-left: none; border-top: 1px solid var(--border); }
}
@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; animation: none !important; }
}
"""


# ---------------------------------------------------------------------------- data


@st.cache_resource(show_spinner="Cargando el modelo…")
def load_artifact(model_path):
    return joblib.load(model_path)


@st.cache_data(show_spinner="Segmentando a los clientes…")
def load_segmented_customers(data_path, model_path):
    artifact = load_artifact(model_path)
    customers = pd.read_csv(data_path).rename(columns=COLUMN_NAMES)
    customers["cluster"] = artifact["pipeline"].predict(customers[artifact["features"]])
    customers["profile"] = customers["cluster"].map(artifact["cluster_profiles"])
    return customers


def get_centroids(artifact):
    scaler = artifact["pipeline"].named_steps["scaler"]
    kmeans = artifact["pipeline"].named_steps["kmeans"]
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=artifact["features"])
    centroids["cluster"] = centroids.index
    centroids["profile"] = centroids["cluster"].map(artifact["cluster_profiles"])
    return centroids


def build_profile_summary(customers):
    summary = (
        customers.groupby(["cluster", "profile"])
        .agg(
            customers_count=("customer_id", "size"),
            age_mean=("age", "mean"),
            income_mean=("annual_income", "mean"),
            spending_mean=("spending_score", "mean"),
            female_share=("gender", lambda gender: (gender == "Female").mean() * 100),
        )
        .reset_index()
        .sort_values("spending_mean", ascending=False)
    )
    summary["customers_share"] = summary["customers_count"] / len(customers) * 100
    return summary


def classify_customer(artifact, centroids, annual_income, spending_score):
    new_customer = pd.DataFrame({"annual_income": [annual_income], "spending_score": [spending_score]})
    cluster = int(artifact["pipeline"].predict(new_customer[artifact["features"]])[0])
    scaled_point = artifact["pipeline"].named_steps["scaler"].transform(new_customer[artifact["features"]])
    scaled_centroids = artifact["pipeline"].named_steps["kmeans"].cluster_centers_
    distances = np.linalg.norm(scaled_centroids - scaled_point, axis=1)
    distance_table = centroids[["cluster", "profile"]].assign(distance=distances).sort_values("distance")
    return {
        "annual_income": annual_income,
        "spending_score": spending_score,
        "cluster": cluster,
        "profile": artifact["cluster_profiles"][cluster],
        "distances": distance_table,
    }


def short_profile_name(profile):
    return profile.replace("Clientes ", "").replace("de ", "").capitalize()


# ---------------------------------------------------------------------------- HTML components


def marker_html(profile, color, size=14):
    symbol = PROFILE_SYMBOLS.get(profile, "circle")
    return (
        f'<span class="marker marker-{symbol}" aria-hidden="true" '
        f'style="width:{size}px;height:{size}px;background:{color}"></span>'
    )


def status_icon(icon, tone):
    return f'<span class="status-icon" aria-hidden="true" style="background:{TONE_COLORS[tone]}">{ICON_GLYPHS[icon]}</span>'


def badge(text, tone="neutral", icon=None, mono=False):
    icon_html = status_icon(icon, tone) if icon else ""
    css_class = "badge badge-mono" if mono else "badge"
    return f'<span class="{css_class}">{icon_html}{escape(text)}</span>'


def panel_header(title, description=None):
    description_html = f'<p class="panel-description">{escape(description)}</p>' if description else ""
    st.html(f'<div><h3 class="panel-title">{escape(title)}</h3>{description_html}</div>')


def section_header(title, lead):
    st.html(f'<div><h2 class="section-title">{escape(title)}</h2><p class="section-lead">{escape(lead)}</p></div>')


def kpi_strip_html(items):
    cells = "".join(
        f'<div class="kpi"><div class="kpi-label">{escape(label)}</div>'
        f'<div class="kpi-value">{escape(value)}</div><div class="kpi-note">{escape(note)}</div></div>'
        for label, value, note in items
    )
    return f'<section class="kpi-strip" aria-label="Indicadores del modelo">{cells}</section>'


def legend_html(artifact):
    items = "".join(
        f'<span class="legend-item">{marker_html(profile, details["color"], 12)}{escape(profile)}</span>'
        for profile, details in artifact["profiles"].items()
    )
    return f'<div class="legend" aria-label="Leyenda de segmentos">{items}<span class="legend-item">✕ Centroide</span></div>'


def segment_list_html(summary, artifact):
    rows = []
    for row in summary.itertuples():
        color = artifact["profiles"][row.profile]["color"]
        rows.append(
            f'<div class="segment-row" role="listitem">{marker_html(row.profile, color)}'
            f'<span class="segment-name">{escape(row.profile)}</span>'
            f'<span class="segment-count">{row.customers_count}'
            f'<span class="segment-share"> · {row.customers_share:.1f} %</span></span>'
            f'<div class="share-track" aria-hidden="true">'
            f'<div class="share-fill" style="width:{row.customers_share:.1f}%;background:{color}"></div></div>'
            f'<span class="segment-meta">C{row.cluster} · ingreso {row.income_mean:.1f} k$ · gasto {row.spending_mean:.1f}</span>'
            "</div>"
        )
    return f'<div class="segment-list" role="list" aria-label="Segmentos por gasto medio">{"".join(rows)}</div>'


def profile_directory_html(summary, artifact):
    rows = []
    for row in summary.itertuples():
        details = artifact["profiles"][row.profile]
        rows.append(
            '<article class="profile">'
            f'<div><div class="profile-name">{marker_html(row.profile, details["color"])}{escape(row.profile)}</div>'
            f'<div class="profile-stats">C{row.cluster} · {row.customers_count} clientes ({row.customers_share:.1f} %)<br>'
            f'Ingreso medio {row.income_mean:.1f} k$ · gasto medio {row.spending_mean:.1f}<br>'
            f'Edad media {row.age_mean:.1f} años · {row.female_share:.0f} % mujeres</div></div>'
            f'<div><p class="label">Quiénes son</p><p class="profile-text">{escape(details["description"])}</p></div>'
            f'<div><p class="label">Estrategia recomendada</p><p class="profile-text">{escape(details["strategy"])}</p></div>'
            "</article>"
        )
    return f'<div class="profiles">{"".join(rows)}</div>'


def criteria_html(artifact, k_metrics):
    optimal_k = artifact["optimal_k"]
    reduction_in = k_metrics.loc[k_metrics["k"] == optimal_k, "inertia_reduction_pct"].item()
    reduction_out = k_metrics.loc[k_metrics["k"] == optimal_k + 1, "inertia_reduction_pct"].item()
    second_best = k_metrics.dropna(subset=["silhouette"]).nlargest(2, "silhouette").iloc[1]
    items = [
        (
            "Método del codo",
            f"K = {optimal_k}",
            f"La inercia baja {reduction_in:.1f} % al pasar de {optimal_k - 1} a {optimal_k}, "
            f"y solo {reduction_out:.1f} % de {optimal_k} a {optimal_k + 1}.",
        ),
        (
            "Coeficiente de silueta",
            f"{artifact['silhouette']:.3f}",
            f"El máximo de K = 2…10 se da en K = {optimal_k}. "
            f"El siguiente mejor es K = {int(second_best['k'])} ({second_best['silhouette']:.3f}).",
        ),
        (
            "Índice de Davies-Bouldin",
            f"{artifact['davies_bouldin']:.3f}",
            f"El mínimo de K = 2…10 se da en K = {optimal_k}. Un valor menor indica clusters mejor separados.",
        ),
    ]
    cells = "".join(
        f'<div class="criterion"><div class="criterion-head">'
        f'{status_icon("check", "good")}{escape(title)}</div>'
        f'<div class="criterion-value">{escape(value)}</div><div class="criterion-note">{escape(note)}</div></div>'
        for title, value, note in items
    )
    return f'<section class="criteria" aria-label="Criterios para elegir K">{cells}</section>'


def distance_list_html(distance_table, artifact):
    max_distance = distance_table["distance"].max()
    rows = []
    for position, row in enumerate(distance_table.itertuples()):
        color = artifact["profiles"][row.profile]["color"]
        nearest_class = " is-nearest" if position == 0 else ""
        rows.append(
            f'<div class="distance-row{nearest_class}">'
            f'<div class="distance-name">{marker_html(row.profile, color, 12)}<span>C{row.cluster} · {escape(short_profile_name(row.profile))}</span></div>'
            f'<div class="distance-track" aria-hidden="true"><div class="distance-fill" style="width:{row.distance / max_distance * 100:.1f}%"></div></div>'
            f'<div class="distance-value">{row.distance:.2f}</div></div>'
        )
    return "".join(rows)


def callout_html(message, tone="warning"):
    return (
        f'<div class="callout" role="alert">{status_icon("alert", tone)}'
        f"<span>{escape(message)}</span></div>"
    )


# ---------------------------------------------------------------------------- charts


def style_figure(fig, height):
    axis_style = {
        "gridcolor": TOKENS["grid"],
        "linecolor": TOKENS["axis"],
        "showline": True,
        "zeroline": False,
        "tickfont": {"color": TOKENS["muted"], "size": 12},
        "title_font": {"color": TOKENS["ink-secondary"], "size": 13},
    }
    fig.update_layout(
        height=height,
        paper_bgcolor=TOKENS["surface"],
        plot_bgcolor=TOKENS["surface"],
        font={"family": FONT_STACK, "color": TOKENS["ink"]},
        margin={"l": 56, "r": 16, "t": 16, "b": 48},
        hoverlabel={"bgcolor": "#ffffff", "bordercolor": TOKENS["axis"], "font": {"family": FONT_STACK, "color": TOKENS["ink"]}},
        showlegend=False,
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    return fig


def build_metric_line(k_metrics, column, y_title, best_k, best_label, value_format):
    data = k_metrics.dropna(subset=[column])
    best_value = data.loc[data["k"] == best_k, column].item()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["k"],
            y=data[column],
            mode="lines+markers",
            line={"color": TOKENS["accent"], "width": 2},
            marker={"size": 8, "color": TOKENS["accent"], "line": {"color": TOKENS["surface"], "width": 2}},
            hovertemplate=f"K = %{{x}}<br>{y_title}: %{{y:{value_format}}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[best_k],
            y=[best_value],
            mode="markers",
            marker={"size": 18, "color": "rgba(0,0,0,0)", "line": {"color": TOKENS["ink"], "width": 2}},
            hoverinfo="skip",
        )
    )
    fig.add_annotation(
        x=best_k,
        y=best_value,
        text=f"<b>{best_label}</b> · K = {best_k} · {best_value:{value_format}}",
        showarrow=True,
        arrowhead=0,
        arrowcolor=TOKENS["axis"],
        ax=64,
        ay=-44,
        bgcolor=TOKENS["surface"],
        bordercolor=TOKENS["border"],
        borderwidth=1,
        borderpad=4,
        font={"color": TOKENS["ink"], "size": 12},
    )
    fig.update_xaxes(title="Número de clusters (K)", dtick=1)
    fig.update_yaxes(title=y_title)
    return style_figure(fig, height=340)


def build_cluster_map(customers, centroids, artifact, new_customer=None):
    feature_labels = artifact["feature_labels"]
    has_new_customer = new_customer is not None
    fig = go.Figure()

    for profile, details in artifact["profiles"].items():
        members = customers[customers["profile"] == profile]
        is_highlighted = not has_new_customer or profile == new_customer["profile"]
        fig.add_trace(
            go.Scatter(
                x=members["annual_income"],
                y=members["spending_score"],
                mode="markers",
                name=profile,
                marker={
                    "size": 10,
                    "color": details["color"],
                    "symbol": PROFILE_SYMBOLS.get(profile, "circle"),
                    "opacity": 0.9 if is_highlighted else 0.22,
                    "line": {"color": TOKENS["surface"], "width": 1.5},
                },
                customdata=np.stack([members["customer_id"], members["age"], members["gender"], members["cluster"]], axis=-1),
                hovertemplate=(
                    f"<b>{profile}</b><br>"
                    "Cliente #%{customdata[0]} · %{customdata[2]}, %{customdata[1]} años<br>"
                    f"{feature_labels['annual_income']}: %{{x}}<br>"
                    f"{feature_labels['spending_score']}: %{{y}}<br>"
                    "Cluster: C%{customdata[3]}<extra></extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=centroids["annual_income"],
            y=centroids["spending_score"],
            mode="markers",
            name="Centroides",
            marker={"symbol": "x-thin", "size": 15, "line": {"color": TOKENS["ink"], "width": 2.5}},
            customdata=centroids["cluster"],
            hovertemplate="<b>Centroide C%{customdata}</b><br>Ingreso: %{x:.1f} k$<br>Gasto: %{y:.1f}<extra></extra>",
        )
    )
    # Direct labels on a surface-colored plate so they stay readable over the points
    for centroid in centroids.itertuples():
        fig.add_annotation(
            x=centroid.annual_income,
            y=centroid.spending_score,
            text=f"<b>C{centroid.cluster}</b> · {short_profile_name(centroid.profile)}",
            showarrow=False,
            yshift=20,
            bgcolor="rgba(252, 252, 251, 0.94)",
            bordercolor=TOKENS["border"],
            borderwidth=1,
            borderpad=3,
            font={"color": TOKENS["ink"], "size": 12},
        )

    if has_new_customer:
        fig.add_trace(
            go.Scatter(
                x=[new_customer["annual_income"]],
                y=[new_customer["spending_score"]],
                mode="markers+text",
                name="Nuevo cliente",
                marker={"symbol": "star", "size": 24, "color": TOKENS["ink"], "line": {"color": TOKENS["surface"], "width": 2}},
                text=["Nuevo cliente"],
                textposition="bottom center",
                textfont={"color": TOKENS["ink"], "size": 13},
                hovertemplate=(
                    "<b>Nuevo cliente</b><br>Ingreso: %{x} k$<br>Gasto: %{y}<br>"
                    f"Perfil: {new_customer['profile']}<extra></extra>"
                ),
            )
        )

    fig.update_xaxes(title=feature_labels["annual_income"])
    fig.update_yaxes(title=feature_labels["spending_score"], range=[-5, 108])
    return style_figure(fig, height=520)


# ---------------------------------------------------------------------------- views


def open_classifier_tab():
    st.session_state[TABS_KEY] = TABS[1]


def render_page_header(artifact):
    title_column, action_column = st.columns([2.2, 1], vertical_alignment="bottom", gap="small")
    with title_column:
        st.html(
            '<div class="topline"><span class="eyebrow">Minería de datos · Proyecto final</span>'
            f'{badge(f"Modelo validado · K = {artifact["optimal_k"]}", "good", "check")}</div>'
            '<h1 class="page-title">Segmentación de clientes con <span class="nowrap">K-Means</span></h1>'
            '<p class="page-lead">Cinco segmentos construidos a partir del ingreso anual y el puntaje de gasto de '
            "200 clientes del centro comercial. Compara los segmentos, clasifica clientes nuevos y revisa cómo se eligió K.</p>"
        )
    with action_column:
        st.button(
            "Clasificar un cliente",
            type="primary",
            icon=":material/person_search:",
            on_click=open_classifier_tab,
            width="stretch",
        )


def render_segments_view(customers, centroids, artifact):
    summary = build_profile_summary(customers)
    map_column, list_column = st.columns([2.2, 1], gap="large")
    with map_column:
        with st.container(border=True):
            panel_header(
                "Mapa de segmentos",
                "Cada punto es un cliente: el color y la forma indican su segmento y las ✕ marcan los centroides. "
                "Pasa el cursor sobre un punto para ver su detalle.",
            )
            st.plotly_chart(build_cluster_map(customers, centroids, artifact), theme=None, config={"displaylogo": False})
    with list_column:
        with st.container(border=True):
            panel_header("Tamaño de los segmentos", "Clientes por segmento y valores medios, ordenados por gasto.")
            st.html(segment_list_html(summary, artifact))

    section_header(
        "Perfiles y estrategias",
        "Qué caracteriza a cada segmento y qué acción de marketing se recomienda para activarlo o retenerlo.",
    )
    st.html(profile_directory_html(summary, artifact))

    with st.expander("Tabla comparativa de segmentos"):
        st.dataframe(
            summary[["cluster", "profile", "customers_count", "customers_share", "income_mean", "spending_mean", "age_mean", "female_share"]],
            hide_index=True,
            column_config={
                "cluster": st.column_config.NumberColumn("Cluster", width="small"),
                "profile": st.column_config.TextColumn("Perfil"),
                "customers_count": st.column_config.NumberColumn("Clientes", width="small"),
                "customers_share": st.column_config.ProgressColumn("% del total", format="%.1f %%", min_value=0, max_value=100),
                "income_mean": st.column_config.NumberColumn("Ingreso medio (k$)", format="%.1f"),
                "spending_mean": st.column_config.NumberColumn("Gasto medio", format="%.1f"),
                "age_mean": st.column_config.NumberColumn("Edad media", format="%.1f"),
                "female_share": st.column_config.NumberColumn("% mujeres", format="%.0f %%"),
            },
        )
    st.download_button(
        "Descargar clientes segmentados (CSV)",
        data=customers.rename(columns={value: key for key, value in COLUMN_NAMES.items()})
        .rename(columns={"cluster": "Cluster", "profile": "Perfil"})
        .to_csv(index=False)
        .encode("utf-8-sig"),
        file_name="mall_customers_segmented.csv",
        mime="text/csv",
        icon=":material/download:",
    )


def render_classifier_view(customers, centroids, artifact):
    income_min, income_max = int(customers["annual_income"].min()), int(customers["annual_income"].max())
    input_column, result_column = st.columns([1, 1.6], gap="large")

    with input_column:
        with st.container(border=True):
            panel_header("Datos del cliente", "El modelo usa solo estas dos variables y las escala igual que en el entrenamiento.")
            annual_income = st.number_input(
                "Ingreso anual (k$)",
                min_value=0,
                max_value=300,
                value=90,
                step=1,
                key="new_income",
                help="Ingreso anual del cliente en miles de dólares.",
            )
            spending_score = st.slider(
                "Puntaje de gasto (1-100)",
                min_value=1,
                max_value=100,
                value=20,
                key="new_spending",
                help="Índice que asigna el centro comercial según el comportamiento de compra del cliente.",
            )
            st.html(f'<p class="field-help">Rango de ingresos observado en el entrenamiento: {income_min}–{income_max} k$.</p>')
            is_out_of_range = not income_min <= annual_income <= income_max
            if is_out_of_range:
                st.html(
                    callout_html(
                        f"El ingreso está fuera del rango de entrenamiento ({income_min}–{income_max} k$). "
                        "La asignación es una extrapolación: interprétala con cuidado."
                    )
                )

    new_customer = classify_customer(artifact, centroids, annual_income, spending_score)
    profile_details = artifact["profiles"][new_customer["profile"]]
    nearest, runner_up = new_customer["distances"].iloc[0], new_customer["distances"].iloc[1]
    if is_out_of_range:
        confidence_badge = badge("Fuera del rango de entrenamiento", "warning", "alert")
    elif runner_up["distance"] - nearest["distance"] > 0.5:
        confidence_badge = badge("Asignación clara", "good", "check")
    else:
        confidence_badge = badge(f"Cerca de la frontera con {short_profile_name(runner_up['profile'])}", "warning", "alert")

    with result_column:
        with st.container(border=True):
            st.html(
                '<div role="status" aria-live="polite">'
                '<p class="label">Perfil asignado</p>'
                f'<div class="result-title">{marker_html(new_customer["profile"], profile_details["color"], 18)}'
                f'{escape(new_customer["profile"])}</div>'
                f'<div class="result-badges">{badge(f"Cluster C{new_customer["cluster"]}", mono=True)}'
                f'{badge(f"Distancia al centroide {nearest["distance"]:.2f}", mono=True)}{confidence_badge}</div>'
                '<div class="result-grid">'
                f'<div><p class="label">Quiénes son</p><p class="profile-text">{escape(profile_details["description"])}</p></div>'
                f'<div><p class="label">Estrategia recomendada</p><p class="profile-text">{escape(profile_details["strategy"])}</p></div>'
                "</div>"
                '<div class="result-section"><p class="label">Distancia a cada centroide (escala estandarizada)</p>'
                f"{distance_list_html(new_customer['distances'], artifact)}</div>"
                "</div>"
            )

    with st.container(border=True):
        panel_header("Ubicación en el mapa de segmentos", "La ★ es el cliente ingresado; se resalta el segmento asignado.")
        st.plotly_chart(build_cluster_map(customers, centroids, artifact, new_customer), theme=None, config={"displaylogo": False})
        st.html(legend_html(artifact))


def render_model_view(artifact):
    k_metrics = pd.DataFrame(artifact["k_metrics"])
    optimal_k = artifact["optimal_k"]
    st.html(
        '<div><h2 class="section-title" style="margin-top:0">Por qué cinco segmentos</h2>'
        '<p class="section-lead">K-Means se entrenó para K = 1…10 sobre las variables estandarizadas. '
        "Tres criterios independientes coinciden en el mismo número de clusters.</p></div>"
    )
    st.html(criteria_html(artifact, k_metrics))

    elbow_column, silhouette_column = st.columns(2, gap="large")
    with elbow_column:
        with st.container(border=True):
            panel_header("Inercia por K", "La inercia siempre baja al aumentar K. El codo marca dónde deja de compensar.")
            st.plotly_chart(
                build_metric_line(k_metrics, "inertia", "Inercia (WCSS)", optimal_k, "Codo", ".1f"),
                theme=None,
                config={"displaylogo": False},
            )
    with silhouette_column:
        with st.container(border=True):
            panel_header("Coeficiente de silueta por K", "Mide cohesión y separación de los clusters. Mayor es mejor.")
            st.plotly_chart(
                build_metric_line(k_metrics, "silhouette", "Silueta promedio", optimal_k, "Máximo", ".3f"),
                theme=None,
                config={"displaylogo": False},
            )

    with st.expander("Tabla de métricas por K"):
        st.dataframe(
            k_metrics.rename(
                columns={
                    "k": "K",
                    "inertia": "Inercia",
                    "inertia_reduction_pct": "Reducción vs. K-1 (%)",
                    "silhouette": "Silueta",
                    "davies_bouldin": "Davies-Bouldin",
                }
            ),
            hide_index=True,
            column_config={
                "Inercia": st.column_config.NumberColumn(format="%.2f"),
                "Reducción vs. K-1 (%)": st.column_config.NumberColumn(format="%.1f"),
                "Silueta": st.column_config.NumberColumn(format="%.4f"),
                "Davies-Bouldin": st.column_config.NumberColumn(format="%.4f"),
            },
        )


def render_missing_files():
    st.html(
        '<section class="empty-state" role="alert"><h2>No se encontró el modelo entrenado</h2>'
        "<p>La aplicación necesita el modelo y el dataset para mostrar los segmentos:</p>"
        f"<p><code>{escape(str(MODEL_PATH.relative_to(BASE_DIR)))}</code><br>"
        f"<code>{escape(str(DATA_PATH.relative_to(BASE_DIR)))}</code></p>"
        "<p>Ejecuta completo el cuaderno <code>proyecto_final_kmeans.ipynb</code> para generarlo y vuelve a cargar la página.</p>"
        "</section>"
    )


def render_footer(artifact):
    st.html(
        '<footer class="app-footer">'
        f'<span>Modelo: StandardScaler + K-Means · <code>random_state = {artifact["random_state"]}</code> · '
        "entrenado en <code>proyecto_final_kmeans.ipynb</code></span>"
        "<span>Datos: Mall Customers (Kaggle) · ingreso anual y puntaje de gasto</span>"
        "</footer>"
    )


def main():
    st.set_page_config(page_title="Segmentación de clientes · K-Means", page_icon=":material/scatter_plot:", layout="wide")
    root_tokens = "".join(f"--{name}: {value};" for name, value in TOKENS.items())
    st.html(f"<style>:root {{{root_tokens}}}{PAGE_CSS}</style>")

    if not MODEL_PATH.exists() or not DATA_PATH.exists():
        render_missing_files()
        st.stop()

    artifact = load_artifact(MODEL_PATH)
    customers = load_segmented_customers(DATA_PATH, MODEL_PATH)
    centroids = get_centroids(artifact)

    render_page_header(artifact)
    st.html(
        kpi_strip_html(
            [
                ("Clientes analizados", f"{len(customers)}", "Dataset Mall Customers"),
                ("Segmentos", f"{artifact['optimal_k']}", "Codo, silueta y Davies-Bouldin coinciden"),
                ("Coeficiente de silueta", f"{artifact['silhouette']:.3f}", "Estructura razonable (0.51–0.70)"),
                ("Índice de Davies-Bouldin", f"{artifact['davies_bouldin']:.3f}", "Menor es mejor"),
            ]
        )
    )

    segments_tab, classifier_tab, model_tab = st.tabs(TABS, key=TABS_KEY, on_change="rerun")
    if segments_tab.open:
        with segments_tab:
            render_segments_view(customers, centroids, artifact)
    if classifier_tab.open:
        with classifier_tab:
            render_classifier_view(customers, centroids, artifact)
    if model_tab.open:
        with model_tab:
            render_model_view(artifact)

    render_footer(artifact)


main()
