import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px


def main():
    st.set_page_config(page_title="Globant", layout="wide")

    # Cargar datos
    @st.cache_data
    def load_data(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)

        # Convertir fecha
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # Crear columnas temporales si hay fecha
        if "Date" in df.columns:
            df["Week"] = df["Date"].dt.isocalendar().week
            df["DayOfMonth"] = df["Date"].dt.day
            df["DayOfWeek"] = df["Date"].dt.day_name()
        else:
            # Fallback por si acaso
            if "Week" not in df.columns:
                df["Week"] = np.nan
            if "DayOfMonth" not in df.columns:
                df["DayOfMonth"] = np.nan
            if "DayOfWeek" not in df.columns:
                df["DayOfWeek"] = np.nan
        return df

    try:
        # Ajusta la ruta si es necesario
        df = load_data("data_globant_clean.csv")
    except FileNotFoundError:
        st.error(
            "No se encontró el archivo `data_globant_clean.csv`.\n\n"
            "💡 Colócalo en la misma carpeta que este script o cambia la ruta en `load_data()`."
        )
        return

    if "Engagement" not in df.columns:
        st.error("La columna `Engagement` no existe en el CSV. Revisa el nombre exacto.")
        st.stop()

    # Interfaz
    st.title("Engagement Globant")
    st.markdown(
        "Explora el engagement a lo largo del tiempo filtrando los datos por proyecto, estudio, equipo, "
        "posición, seniority y locación."
    )

    st.sidebar.header("Filtros")

    filters = {
        "Project": "Proyecto",
        "Studio": "Estudio",
        "Team Name": "Equipo",
        "Position": "Posición",
        "Seniority": "Seniority",
        "Location": "Locación",
    }

    df_filtered = df.copy()

    for col, label in filters.items():
        if col in df.columns:
            opciones_col = sorted(df[col].dropna().unique().tolist())
            selected = st.sidebar.multiselect(
                f"{label}:",
                options=["Todos"] + opciones_col,
                default=["Todos"],
            )
            if "Todos" not in selected:
                df_filtered = df_filtered[df_filtered[col].isin(selected)]

    # ===== Tipo de agregación temporal =====
    agg_type = st.sidebar.selectbox(
        "Tipo de agregación temporal:",
        ["Promedio semanal", "Promedio por día de la semana", "Promedio por día del mes"],
    )

    # ===== Tratamiento de ceros =====
    ignore_zero = st.sidebar.checkbox(
        "Excluir engagement <= 0 del promedio",
        value=True,
        help="Los valores <= 0 no cuentan para el promedio, pero sí para el color de la línea.",
    )

    # Preprocesamiento
    if df_filtered.empty:
        st.warning("No hay datos que coincidan con los filtros seleccionados.")
        st.stop()

    df_vis = df_filtered.copy()

    # Marcamos dónde hay 0 o menos
    df_vis["IsZero"] = (df_vis["Engagement"] <= 0).astype(int)

    if ignore_zero:
        df_vis.loc[df_vis["Engagement"] <= 0, "Engagement"] = np.nan

    # Agregación
    def aggregate(df_in: pd.DataFrame, group_col: str) -> pd.DataFrame:
        if group_col not in df_in.columns:
            return pd.DataFrame()
        grouped = df_in.groupby(group_col).agg(
            Engagement_mean=("Engagement", "mean"),
            CountNonNa=("Engagement", "count"),   # registros que sí aportan promedio
            ZeroCount=("IsZero", "sum"),          # cuántos son 0 o menos
        )
        grouped = grouped.reset_index()
        return grouped

    if agg_type == "Promedio semanal":
        group_col = "Week"
        x_title = "Semana del año"
        df_plot = aggregate(df_vis, group_col)

    elif agg_type == "Promedio por día de la semana":
        group_col = "DayOfWeek"
        x_title = "Día de la semana"
        df_plot = aggregate(df_vis, group_col)
        if not df_plot.empty:
            order = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            df_plot[group_col] = pd.Categorical(
                df_plot[group_col], categories=order, ordered=True
            )
            df_plot = df_plot.sort_values(group_col)

    else:  # "Promedio por día del mes"
        group_col = "DayOfMonth"
        x_title = "Día del mes"
        df_plot = aggregate(df_vis, group_col)

    if df_plot.empty:
        st.warning("No hay datos agregados para la combinación de filtros y tipo de agregación.")
        st.stop()

    # Color dinámico
    df_plot["TotalRegistros"] = df_plot["ZeroCount"] + df_plot["CountNonNa"]
    df_plot["PercentZero"] = np.where(
        df_plot["TotalRegistros"] > 0,
        df_plot["ZeroCount"] / df_plot["TotalRegistros"],
        0.0,
    )

    def get_color(p: float) -> str:
        if p < 0.05:
            return "green"
        elif p < 0.15:
            return "yellow"
        else:
            return "red"

    df_plot["Color"] = df_plot["PercentZero"].apply(get_color)

    # Gráfica
    st.subheader("Evolución del engagement")

    fig = px.line(
        df_plot,
        x=group_col,
        y="Engagement_mean",
        markers=True,
        title="Engagement promedio según selección",
    )

    # Línea neutra y puntos coloreados según % de ceros
    fig.update_traces(
        line=dict(color="lightgray", width=2),
        marker=dict(size=10),
    )

    # Aplicar colores punto a punto
    fig.data[0].marker.color = df_plot["Color"]

    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="Engagement promedio",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Datos agregados")
    st.dataframe(df_plot)


if __name__ == "__main__":
    main()