"""Streamlit app: Mall Customers segmentation with K-Means.

Loads the pipeline trained in ``proyecto_final_kmeans.ipynb`` and lets the user
explore the elbow method, the generated clusters and classify a new customer.
Run with: ``streamlit run app.py``
"""
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

# Chart chrome (light chart surface) - charts always render on this surface so the validated palette holds
CHART_TOKENS = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "highlight": "#c0392b",
    "font": 'system-ui, -apple-system, "Segoe UI", sans-serif',
}

# Marker shapes are the secondary encoding for the five profiles (identity never relies on color alone)
PROFILE_SYMBOLS = {
    "Clientes de alto valor": "diamond",
    "Clientes promedio": "circle",
    "Clientes de bajo valor": "triangle-down",
    "Clientes con potencial": "square",
    "Compradores impulsivos": "triangle-up",
}

VIEWS = ["Método del codo", "Clusters", "Clasificar cliente"]

PAGE_CSS = """
<style>
:root {
  --space-xs: 8px;
  --space-sm: 16px;
  --space-md: 24px;
  --space-lg: 48px;
  --radius-sm: 6px;
  --hairline: rgba(128, 128, 128, 0.28);
}
.block-container { max-width: 1280px; padding-top: var(--space-lg); padding-bottom: var(--space-lg); }
.eyebrow {
  font-size: 13px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; opacity: 0.6;
  margin-bottom: var(--space-xs);
}
.hero-title {
  font-size: clamp(2.25rem, 5vw, 3.75rem); line-height: 1.05; font-weight: 750; letter-spacing: -0.025em;
  margin: 0 0 var(--space-sm) 0; max-width: 16ch;
}
.hero-lead { font-size: 1.1rem; line-height: 1.55; opacity: 0.75; max-width: 64ch; margin-bottom: var(--space-md); }
.kpi { border-top: 1px solid var(--hairline); padding-top: var(--space-sm); }
.kpi-label { font-size: 13px; opacity: 0.65; }
.kpi-value { font-size: clamp(1.75rem, 3.2vw, 2.5rem); font-weight: 700; line-height: 1.2; }
.kpi-note { font-size: 13px; opacity: 0.7; }
.section-title { font-size: clamp(1.5rem, 2.6vw, 2rem); font-weight: 700; letter-spacing: -0.015em; margin: var(--space-md) 0 4px 0; }
.section-lead { opacity: 0.72; max-width: 72ch; margin-bottom: var(--space-sm); }
.result {
  border: 1px solid var(--hairline); border-left: 4px solid var(--profile-color);
  border-radius: var(--radius-sm); padding: var(--space-md);
}
.result-label { font-size: 13px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.6; }
.result-title { font-size: clamp(1.6rem, 3vw, 2.25rem); font-weight: 750; letter-spacing: -0.015em; margin: 4px 0 var(--space-xs); }
.result-meta { font-size: 14px; opacity: 0.7; margin-bottom: var(--space-sm); font-variant-numeric: tabular-nums; }
.result h4 { font-size: 15px; margin: var(--space-sm) 0 4px 0; }
.strategy { border-top: 1px solid var(--hairline); padding: var(--space-sm) 0 var(--space-md); }
.strategy-title { font-weight: 650; font-size: 1.05rem; margin-bottom: 4px; }
.strategy-meta { font-weight: 400; font-size: 13px; opacity: 0.6; font-variant-numeric: tabular-nums; }
.strategy-description { opacity: 0.7; font-size: 14px; margin-bottom: var(--space-xs); }
.swatch { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 8px; vertical-align: -1px; }
.footer-note { font-size: 13px; opacity: 0.6; margin-top: var(--space-lg); border-top: 1px solid var(--hairline); padding-top: var(--space-sm); }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
"""


@st.cache_resource(show_spinner="Cargando el modelo...")
def load_artifact(model_path):
    return joblib.load(model_path)


@st.cache_data(show_spinner="Cargando y segmentando los clientes...")
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


def short_profile_name(profile):
    return profile.replace("Clientes ", "").replace("de ", "").capitalize()


def style_figure(fig, height=440):
    axis_style = {
        "gridcolor": CHART_TOKENS["grid"],
        "linecolor": CHART_TOKENS["axis"],
        "zeroline": False,
        "tickfont": {"color": CHART_TOKENS["muted"], "size": 12},
        "title_font": {"color": CHART_TOKENS["ink_secondary"], "size": 13},
        "showline": True,
    }
    fig.update_layout(
        height=height,
        paper_bgcolor=CHART_TOKENS["surface"],
        plot_bgcolor=CHART_TOKENS["surface"],
        font={"family": CHART_TOKENS["font"], "color": CHART_TOKENS["ink"]},
        margin={"l": 64, "r": 24, "t": 56, "b": 56},
        title_font={"size": 16, "color": CHART_TOKENS["ink"]},
        hoverlabel={"bgcolor": "#ffffff", "bordercolor": CHART_TOKENS["axis"], "font": {"color": CHART_TOKENS["ink"]}},
        legend={"font": {"color": CHART_TOKENS["ink_secondary"]}, "bgcolor": "rgba(0,0,0,0)"},
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    return fig


def build_metric_line(k_metrics, column, title, y_title, best_k, best_label, color, value_format):
    data = k_metrics.dropna(subset=[column])
    best_value = data.loc[data["k"] == best_k, column].item()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["k"],
            y=data[column],
            mode="lines+markers",
            line={"color": color, "width": 2},
            marker={"size": 9, "color": color, "line": {"color": CHART_TOKENS["surface"], "width": 2}},
            name=y_title,
            hovertemplate=f"K = %{{x}}<br>{y_title}: %{{y:{value_format}}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[best_k],
            y=[best_value],
            mode="markers",
            marker={"size": 20, "color": "rgba(0,0,0,0)", "line": {"color": CHART_TOKENS["highlight"], "width": 2.5}},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_annotation(
        x=best_k,
        y=best_value,
        text=f"<b>{best_label}</b><br>K = {best_k} · {best_value:{value_format}}",
        showarrow=True,
        arrowhead=0,
        arrowcolor=CHART_TOKENS["muted"],
        ax=60,
        ay=-50,
        font={"color": CHART_TOKENS["ink"], "size": 12},
        align="left",
    )
    fig.update_layout(title=title, showlegend=False)
    fig.update_xaxes(title="Número de clusters (K)", dtick=1)
    fig.update_yaxes(title=y_title)
    return style_figure(fig, height=400)


def build_cluster_map(customers, centroids, artifact, new_customer=None):
    feature_labels = artifact["feature_labels"]
    profiles = artifact["profiles"]
    has_new_customer = new_customer is not None
    fig = go.Figure()

    for profile, details in profiles.items():
        members = customers[customers["profile"] == profile]
        is_assigned = has_new_customer and profile == new_customer["profile"]
        opacity = 0.9 if not has_new_customer or is_assigned else 0.25
        fig.add_trace(
            go.Scatter(
                x=members["annual_income"],
                y=members["spending_score"],
                mode="markers",
                name=f"{profile} ({len(members)})",
                marker={
                    "size": 11,
                    "color": details["color"],
                    "symbol": PROFILE_SYMBOLS.get(profile, "circle"),
                    "opacity": opacity,
                    "line": {"color": CHART_TOKENS["surface"], "width": 1.5},
                },
                customdata=np.stack([members["customer_id"], members["age"], members["gender"], members["cluster"]], axis=-1),
                hovertemplate=(
                    f"<b>{profile}</b><br>"
                    "Cliente #%{customdata[0]} · %{customdata[2]}, %{customdata[1]} años<br>"
                    f"{feature_labels['annual_income']}: %{{x}}<br>"
                    f"{feature_labels['spending_score']}: %{{y}}<br>"
                    "Cluster: %{customdata[3]}<extra></extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=centroids["annual_income"],
            y=centroids["spending_score"],
            mode="markers",
            name="Centroides",
            marker={"symbol": "x-thin", "size": 16, "line": {"color": CHART_TOKENS["ink"], "width": 3}},
            hovertemplate="<b>Centroide C%{customdata}</b><br>Ingreso: %{x:.1f} k$<br>Gasto: %{y:.1f}<extra></extra>",
            customdata=centroids["cluster"],
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
            bgcolor="rgba(252, 252, 251, 0.92)",
            bordercolor=CHART_TOKENS["axis"],
            borderwidth=1,
            borderpad=3,
            font={"color": CHART_TOKENS["ink"], "size": 12},
        )

    if has_new_customer:
        fig.add_trace(
            go.Scatter(
                x=[new_customer["annual_income"]],
                y=[new_customer["spending_score"]],
                mode="markers+text",
                name="Nuevo cliente",
                marker={"symbol": "star", "size": 26, "color": CHART_TOKENS["ink"], "line": {"color": CHART_TOKENS["surface"], "width": 2}},
                text=["Nuevo cliente"],
                textposition="bottom center",
                textfont={"color": CHART_TOKENS["ink"], "size": 13},
                hovertemplate=(
                    "<b>Nuevo cliente</b><br>Ingreso: %{x} k$<br>Gasto: %{y}<br>"
                    f"Perfil: {new_customer['profile']}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Mapa de clusters: ingreso anual vs. puntaje de gasto",
        legend={"orientation": "h", "yanchor": "top", "y": -0.16, "xanchor": "left", "x": 0},
    )
    fig.update_xaxes(title=feature_labels["annual_income"])
    fig.update_yaxes(title=feature_labels["spending_score"], range=[-5, 108])
    return style_figure(fig, height=600)


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


def render_kpi(column, label, value, note):
    column.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def render_section_header(title, lead):
    st.markdown(f'<h2 class="section-title">{title}</h2><p class="section-lead">{lead}</p>', unsafe_allow_html=True)


def render_elbow_view(artifact):
    k_metrics = pd.DataFrame(artifact["k_metrics"])
    optimal_k = artifact["optimal_k"]
    render_section_header(
        "Método del codo y coeficiente de silueta",
        "Se entrenó K-Means para K = 1…10 sobre el ingreso y el puntaje de gasto estandarizados. "
        "La inercia siempre baja al aumentar K. El codo marca el punto en el que agregar clusters "
        "deja de compensar, y la silueta confirma cuál K separa mejor los grupos.",
    )
    elbow_column, silhouette_column = st.columns(2, gap="large")
    elbow_column.plotly_chart(
        build_metric_line(k_metrics, "inertia", "Inercia (WCSS) por K", "Inercia", optimal_k, "Codo", "#2a78d6", ".1f"),
        theme=None,
        config={"displaylogo": False},
    )
    silhouette_column.plotly_chart(
        build_metric_line(k_metrics, "silhouette", "Coeficiente de silueta por K", "Silueta", optimal_k, "Máximo", "#2a78d6", ".3f"),
        theme=None,
        config={"displaylogo": False},
    )

    reduction_from_previous = k_metrics.loc[k_metrics["k"] == optimal_k, "inertia_reduction_pct"].item()
    reduction_to_next = k_metrics.loc[k_metrics["k"] == optimal_k + 1, "inertia_reduction_pct"].item()
    st.info(
        f"**K = {optimal_k}** es el valor elegido. Al pasar de K = {optimal_k - 1} a {optimal_k}, la inercia baja "
        f"{reduction_from_previous:.1f} %, pero de {optimal_k} a {optimal_k + 1} solo baja {reduction_to_next:.1f} % (codo). "
        f"Además, K = {optimal_k} tiene la mayor silueta ({artifact['silhouette']:.3f}) y el menor "
        f"índice de Davies-Bouldin ({artifact['davies_bouldin']:.3f}).",
        icon=":material/check_circle:",
    )

    with st.expander("Ver tabla de métricas por K"):
        metrics_table = k_metrics.rename(
            columns={
                "k": "K",
                "inertia": "Inercia",
                "inertia_reduction_pct": "Reducción vs. K-1 (%)",
                "silhouette": "Silueta",
                "davies_bouldin": "Davies-Bouldin",
            }
        )
        st.dataframe(
            metrics_table,
            hide_index=True,
            column_config={
                "Inercia": st.column_config.NumberColumn(format="%.2f"),
                "Reducción vs. K-1 (%)": st.column_config.NumberColumn(format="%.1f"),
                "Silueta": st.column_config.NumberColumn(format="%.4f"),
                "Davies-Bouldin": st.column_config.NumberColumn(format="%.4f"),
            },
        )


def render_clusters_view(customers, centroids, artifact):
    render_section_header(
        f"{artifact['optimal_k']} segmentos de clientes",
        "Cada punto es un cliente. El color y la forma indican su perfil, y las ✕ marcan los centroides "
        "(el cliente típico de cada grupo). Pasa el cursor sobre un punto para ver el detalle.",
    )
    st.plotly_chart(build_cluster_map(customers, centroids, artifact), theme=None, config={"displaylogo": False})

    st.markdown("#### Resumen por perfil")
    summary = build_profile_summary(customers)
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

    st.markdown("#### Estrategias de marketing por perfil")
    strategy_columns = st.columns(2, gap="large")
    for position, row in enumerate(summary.itertuples()):
        details = artifact["profiles"][row.profile]
        strategy_columns[position % 2].markdown(
            f'<div class="strategy"><div class="strategy-title"><span class="swatch" style="background:{details["color"]}"></span>'
            f'{row.profile} <span class="strategy-meta">· C{row.cluster} · {row.customers_count} clientes</span></div>'
            f'<div class="strategy-description">{details["description"]}</div>'
            f'<div>{details["strategy"]}</div></div>',
            unsafe_allow_html=True,
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


def render_classifier_view(customers, centroids, artifact):
    render_section_header(
        "Clasificar un cliente nuevo",
        "Ingresa el ingreso anual y el puntaje de gasto. El modelo escala los valores con el mismo "
        "StandardScaler del entrenamiento y asigna el cliente al centroide más cercano.",
    )
    income_min, income_max = int(customers["annual_income"].min()), int(customers["annual_income"].max())

    input_column, result_column = st.columns([1, 1.4], gap="large")
    with input_column:
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
        st.caption(f"Rango de ingresos observado en el entrenamiento: {income_min}–{income_max} k$.")
        is_out_of_range = not income_min <= annual_income <= income_max
        if is_out_of_range:
            st.warning(
                f"El ingreso ingresado está fuera del rango con el que se entrenó el modelo ({income_min}–{income_max} k$). "
                "La asignación es una extrapolación: interprétala con cuidado.",
                icon=":material/warning:",
            )

    new_customer = classify_customer(artifact, centroids, annual_income, spending_score)
    profile_details = artifact["profiles"][new_customer["profile"]]
    nearest, runner_up = new_customer["distances"].iloc[0], new_customer["distances"].iloc[1]
    if is_out_of_range:
        confidence_note = "Fuera del rango de entrenamiento."
    elif runner_up["distance"] - nearest["distance"] > 0.5:
        confidence_note = "Está claramente dentro del segmento."
    else:
        confidence_note = f"Está cerca de la frontera con «{runner_up['profile']}»."

    with result_column:
        st.markdown(
            f"""
<div class="result" style="--profile-color: {profile_details['color']}" role="status" aria-live="polite">
  <div class="result-label">Perfil asignado</div>
  <div class="result-title"><span class="swatch" style="background:{profile_details['color']}"></span>{new_customer['profile']}</div>
  <div class="result-meta">Cluster {new_customer['cluster']} · distancia al centroide {nearest['distance']:.2f} (z) · {confidence_note}</div>
  <div>{profile_details['description']}</div>
  <h4>Estrategia recomendada</h4>
  <div>{profile_details['strategy']}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.plotly_chart(build_cluster_map(customers, centroids, artifact, new_customer), theme=None, config={"displaylogo": False})

    with st.expander("Ver distancia a cada centroide"):
        st.dataframe(
            new_customer["distances"],
            hide_index=True,
            column_config={
                "cluster": st.column_config.NumberColumn("Cluster", width="small"),
                "profile": st.column_config.TextColumn("Perfil"),
                "distance": st.column_config.NumberColumn("Distancia euclidiana (escala z)", format="%.3f"),
            },
        )


def main():
    st.set_page_config(page_title="Segmentación de clientes · K-Means", page_icon=":material/scatter_plot:", layout="wide")
    st.markdown(PAGE_CSS, unsafe_allow_html=True)

    if not MODEL_PATH.exists() or not DATA_PATH.exists():
        st.error(
            "No se encontró el modelo entrenado o el dataset.\n\n"
            f"- Modelo esperado: `{MODEL_PATH.relative_to(BASE_DIR)}`\n"
            f"- Dataset esperado: `{DATA_PATH.relative_to(BASE_DIR)}`\n\n"
            "Ejecuta completo el cuaderno `proyecto_final_kmeans.ipynb` para generar el modelo y vuelve a cargar la página.",
            icon=":material/error:",
        )
        st.stop()

    artifact = load_artifact(MODEL_PATH)
    customers = load_segmented_customers(DATA_PATH, MODEL_PATH)
    centroids = get_centroids(artifact)

    st.markdown(
        """
<header>
  <div class="eyebrow">Minería de datos · Proyecto final</div>
  <h1 class="hero-title">Segmentación de clientes con K-Means</h1>
  <p class="hero-lead">Agrupa a los clientes del centro comercial según su ingreso anual y su puntaje de gasto
  para diseñar estrategias de marketing por segmento y clasificar automáticamente a los clientes nuevos.</p>
</header>
""",
        unsafe_allow_html=True,
    )

    kpi_columns = st.columns(4, gap="large")
    render_kpi(kpi_columns[0], "Clientes analizados", f"{len(customers)}", "Dataset Mall Customers")
    render_kpi(kpi_columns[1], "Segmentos (K)", f"{artifact['optimal_k']}", "Codo, silueta y Davies-Bouldin coinciden")
    render_kpi(kpi_columns[2], "Coeficiente de silueta", f"{artifact['silhouette']:.3f}", "Estructura razonable (0.51–0.70)")
    render_kpi(kpi_columns[3], "Davies-Bouldin", f"{artifact['davies_bouldin']:.3f}", "Menor es mejor")

    st.write("")
    view = st.segmented_control("Sección", VIEWS, default=VIEWS[0], key="view", label_visibility="collapsed")
    if view is None:
        view = VIEWS[0]

    if view == "Método del codo":
        render_elbow_view(artifact)
    elif view == "Clusters":
        render_clusters_view(customers, centroids, artifact)
    else:
        render_classifier_view(customers, centroids, artifact)

    st.markdown(
        '<footer class="footer-note">Modelo: StandardScaler + K-Means (random_state = '
        f'{artifact["random_state"]}) entrenado en <code>proyecto_final_kmeans.ipynb</code>. '
        "Variables: ingreso anual (k$) y puntaje de gasto (1-100).</footer>",
        unsafe_allow_html=True,
    )


main()
