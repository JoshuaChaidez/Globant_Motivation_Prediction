import os
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(__file__)
DATA_NN_PATH = os.path.join(BASE_DIR, "data_globant_cnn.csv")
MODEL_NN_PATH = os.path.join(BASE_DIR, "my_model.keras")

CATEGORICAL_NN_COLS: List[str] = [
    "Position",
    "Location",
    "Studio",
    "Client Tag",
    "Project Tag",
    "Team Name",
]

# TensorFlow / Keras
try:
    from tensorflow import keras  # type: ignore
    TF_AVAILABLE = True
    TF_ERROR = None
except Exception as e:
    TF_AVAILABLE = False
    TF_ERROR = str(e)


@st.cache_data(show_spinner=True)
def load_nn_data(path: str) -> pd.DataFrame:
    """
    Carga el CSV para la red neuronal y garantiza columnas de fecha básicas.
    """
    df = pd.read_csv(path)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    if "Date" in df.columns:
        if "Month" not in df.columns:
            df["Month"] = df["Date"].dt.month
        if "Day" not in df.columns:
            df["Day"] = df["Date"].dt.day

    return df


def assign_15day_blocks_nn(group: pd.DataFrame) -> pd.DataFrame:
    """
    Asigna bloques de 15 filas por empleado para la NN.
    """
    group = group.sort_values("Date").reset_index(drop=True)
    group["15_dias"] = (group.index // 15) + 1
    return group


@st.cache_resource(show_spinner=True)
def build_nn_pipeline(data_path: str, model_path: str):
    """
    Construye el pipeline de la red neuronal:

    - Carga datos (data_globant_cnn.csv)
    - Crea 15_dias por EmployeeID
    - One-hot a columnas categóricas
    - Elimina la etiqueta del set de features
    - Normaliza con MinMaxScaler
    - Carga el modelo Keras
    """
    if not TF_AVAILABLE:
        raise RuntimeError(
            "TensorFlow/Keras no está disponible en este entorno. "
            f"Detalle: {TF_ERROR}"
        )

    df_raw = load_nn_data(data_path).copy()

    required = {
        "Date",
        "Position",
        "Seniority",
        "Location",
        "Studio",
        "Client Tag",
        "Project Tag",
        "Team Name",
        "EmployeeID",
        "Engagement_D",
        "Month",
        "Day",
    }
    missing = required - set(df_raw.columns)
    if missing:
        raise ValueError(f"Faltan columnas necesarias en el CSV de NN: {missing}")

    # Crear bloques de 15 días por empleado
    df_blocks = (
        df_raw.groupby("EmployeeID", group_keys=False)
        .apply(assign_15day_blocks_nn)
        .reset_index(drop=True)
    )

    df_display = df_blocks.copy()

    # Preparar datos para el modelo
    df_model = df_blocks.drop(columns=["Date"])

    # One-hot encoding de categóricas
    df_model = pd.get_dummies(df_model, columns=CATEGORICAL_NN_COLS, drop_first=False)

    # Eliminar etiquetas de las features
    drop_cols = [c for c in ["Engagement_D", "Engagement_D_num"] if c in df_model.columns]
    features_df = df_model.drop(columns=drop_cols)

    feature_cols = list(features_df.columns)
    X = features_df.astype(float).values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Cargar modelo
    model = keras.models.load_model(model_path)

    # Validar dimensiones
    model_input_dim = model.input_shape[-1]
    if X_scaled.shape[1] != model_input_dim:
        raise ValueError(
            f"Incompatibilidad entre datos y modelo NN: "
            f"{X_scaled.shape[1]} features vs {model_input_dim} esperadas por el modelo."
        )

    # Etiquetas de salida según neuronas de la última capa
    n_outputs = model.output_shape[-1]
    if n_outputs == 3:
        class_labels = ["Bajo", "Medio", "Alto"]
    else:
        class_labels = [f"Clase {i+1}" for i in range(n_outputs)]

    return df_display, feature_cols, scaler, X_scaled, model, class_labels


def predict_from_index_nn(idx: int, X_scaled: np.ndarray, model) -> np.ndarray:
    """
    Predice usando una fila existente en X_scaled (modo 'Buscar empleado').
    """
    if idx < 0 or idx >= X_scaled.shape[0]:
        raise IndexError("Índice fuera de rango para X_scaled.")
    x = X_scaled[idx : idx + 1]
    probs = model.predict(x, verbose=0)[0]
    return probs


def main():
    st.set_page_config(page_title="Globant – Engagement", layout="wide")

    # =========================
    # Cargar datos principales (exploración + Markov)
    # =========================
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
        st.error("No se encontró el archivo `data_globant_clean.csv`.\n\n")
        return

    if "Engagement" not in df.columns:
        st.error("La columna `Engagement` no existe en el CSV. Revisa el nombre exacto.")
        st.stop()

    # =========================
    # Interfaz principal
    # =========================
    st.title("Engagement Globant")

    st.sidebar.header("Filtros")

    filters = {
        "Studio": "Estudio",
        "Team Name": "Equipo",
        "Position": "Posición",
        "Seniority": "Seniority",
        "Location": "Locación",
    }

    df_filtered = df.copy()
    for col, label in filters.items():
        if col in df.columns:
            options = ["(Todos)"] + sorted(df[col].dropna().unique().tolist())
            selected = st.sidebar.selectbox(label, options, key=f"filter_{col}")
            if selected != "(Todos)":
                df_filtered = df_filtered[df_filtered[col].isin([selected])]
        else:
            st.sidebar.warning(f"La columna `{col}` no existe en el CSV.")

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
        valid = df_in.dropna(subset=["Engagement"]).copy()
        if valid.empty:
            return pd.DataFrame(columns=[group_col, "Engagement_mean", "ZeroCount", "CountNonNa"])

        grouped = (
            valid.groupby(group_col)
            .agg(Engagement_mean=("Engagement", "mean"), ZeroCount=("IsZero", "sum"))
            .reset_index()
        )

        total_counts = df_in.groupby(group_col)["Engagement"].count().reset_index()
        total_counts = total_counts.rename(columns={"Engagement": "CountNonNa"})

        grouped = pd.merge(grouped, total_counts, on=group_col, how="left")
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

    # Gráfica principal
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
    # Aplica color punto a punto
    if "PercentZero" in df_plot.columns:
        colors = [get_color(p) for p in df_plot["PercentZero"]]
        fig.update_traces(marker=dict(color=colors))

    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="Engagement promedio",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # Sección: Predicción con Red Neuronal
    # =========================
    st.markdown("---")
    st.subheader("Red Neuronal MLP")

    if not TF_AVAILABLE:
        st.error(
            "TensorFlow/Keras no está disponible, no se puede cargar el modelo de NN.\n\n"
            f"Detalle técnico: {TF_ERROR}"
        )
    else:
        try:
            df_nn_display, feature_cols, scaler, X_scaled, model, class_labels = build_nn_pipeline(
                DATA_NN_PATH, MODEL_NN_PATH
            )
        except Exception as e:
            st.error(f"No se pudo inicializar el pipeline de la red neuronal: {e}")
        else:
            if df_nn_display.empty:
                st.info("No hay datos disponibles para realizar predicciones con la NN.")
            else:
                if "EmployeeID" not in df_nn_display.columns:
                    st.error("El dataset de NN no tiene columna 'EmployeeID'.")
                else:
                    emp_ids = sorted(df_nn_display["EmployeeID"].unique().tolist())
                    selected_emp = st.selectbox("EmployeeID (NN)", emp_ids)

                    df_emp = df_nn_display[df_nn_display["EmployeeID"] == selected_emp].copy()
                    df_emp = df_emp.sort_values("Date")

                    if df_emp.empty:
                        st.warning("No hay registros para este empleado en el dataset de NN.")
                    else:
                        date_options = df_emp["Date"].dt.strftime("%Y-%m-%d").tolist()
                        default_idx_nn = len(date_options) - 1 if date_options else 0

                        selected_date_str_nn = st.selectbox(
                            "Selecciona el registro por fecha",
                            options=date_options,
                            index=default_idx_nn,
                        )

                        mask_nn = df_emp["Date"].dt.strftime("%Y-%m-%d") == selected_date_str_nn
                        if not mask_nn.any():
                            st.warning("No se encontró el registro seleccionado para la NN.")
                        else:
                            row_emp_nn = df_emp[mask_nn].iloc[-1]
                            idx_global_nn = row_emp_nn.name

                            st.markdown("**Características del registro seleccionado:**")

                            carac_dict = {
                                "Position": row_emp_nn["Position"],
                                "Seniority": row_emp_nn["Seniority"],
                                "Location": row_emp_nn["Location"],
                                "Studio": row_emp_nn["Studio"],
                                "Client Tag": row_emp_nn["Client Tag"],
                                "Project Tag": row_emp_nn["Project Tag"],
                                "Team Name": row_emp_nn["Team Name"],
                                "Month": row_emp_nn["Month"],
                                "Day": row_emp_nn["Day"],
                                "15_dias": row_emp_nn.get("15_dias", np.nan),
                                "Engagement_D (real)": row_emp_nn.get("Engagement_D", None),
                            }

                            df_carac = pd.DataFrame.from_dict(carac_dict, orient="index", columns=["Descripción"])

                            # df_carac display sin encabezados extra
                            st.table(df_carac)

                            if st.button("**RUN**"):
                                try:
                                    probs = predict_from_index_nn(idx_global_nn, X_scaled, model)
                                except Exception as e:
                                    st.error(f"Error al predecir con la NN: {e}")
                                else:
                                    probs = np.array(probs)
                                    if probs.ndim != 1:
                                        st.error("La salida del modelo NN no tiene la forma esperada.")
                                    else:
                                        k = min(len(probs), len(class_labels))
                                        probs = probs[:k]
                                        idx_max = int(np.argmax(probs))
                                        pred_label = class_labels[idx_max]

                                        st.markdown("---")
                                        st.markdown(
                                            f"### Predicción NN: **{pred_label}** "
                                        )

                                        df_probs = pd.DataFrame(
                                            {"Clase": class_labels[:k], "Probabilidad": probs}
                                        )

                                        fig_bar = px.bar(
                                            df_probs,
                                            x="Clase",
                                            y="Probabilidad",
                                            range_y=[0, 1],
                                            title="Distribución de probabilidad por clase (NN)",
                                        )
                                        #fig_bar.update_yaxes(title="Probabilidad")
                                        #st.plotly_chart(fig_bar, use_container_width=True)

                                        st.dataframe(df_probs.set_index("Clase"))

    # =========================
    # Sección: Predicción (Cadena de Markov)
    # =========================
    st.markdown("---")
    st.subheader("Cadena de Markov")

    # Copia de datos solo con columnas necesarias
    df_markov = df.copy()

    # Aseguramos columna de estados discretos
    STATE_COL = "Engagement_bin"
    if STATE_COL not in df_markov.columns:
        # Si no existe, creamos 5 estados a partir de la columna Engagement
        n_states = 5
        df_markov[STATE_COL] = pd.cut(
            df_markov["Engagement"],
            bins=n_states,
            labels=[f"{i+1}" for i in range(n_states)],
            include_lowest=True,
        )

    # Quitamos filas sin estado
    df_markov = df_markov.dropna(subset=[STATE_COL])
    df_markov[STATE_COL] = df_markov[STATE_COL].astype(str)

    @st.cache_data
    def compute_transition_matrix(df_in: pd.DataFrame, state_col: str, id_col: str = "Name"):
        # Orden temporal
        sort_cols = []
        for col in ["Date", "Week", "DayOfMonth"]:
            if col in df_in.columns:
                sort_cols.append(col)

        if not sort_cols:
            # Si no hay columnas temporales, usamos el índice como fallback
            df_local = df_in.reset_index().rename(columns={"index": "_Order"})
            sort_cols_local = ["_Order"]
        else:
            df_local = df_in.copy()
            sort_cols_local = sort_cols

        df_sorted = df_local[[id_col, state_col] + sort_cols_local].dropna(subset=[state_col])
        df_sorted = df_sorted.sort_values([id_col] + sort_cols_local)

        states = np.sort(df_sorted[state_col].unique())
        n = len(states)
        state_to_idx = {s: i for i, s in enumerate(states)}

        counts = np.zeros((n, n), dtype=float)

        for _, group in df_sorted.groupby(id_col):
            s = group[state_col].values
            for i in range(len(s) - 1):
                a = state_to_idx[s[i]]
                b = state_to_idx[s[i + 1]]
                counts[a, b] += 1.0

        row_sums = counts.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            probs = np.where(row_sums > 0, counts / row_sums, 0.0)

        return states, state_to_idx, counts, probs

    states, state_to_idx, counts_mat, P = compute_transition_matrix(df_markov, STATE_COL)

    if len(states) == 0:
        st.info("No hay suficientes datos para construir la cadena de Markov.")
    else:
        # Selector de modo
        modo = st.radio("Selección para predicción", ["Empleado", "Libre"], horizontal=True)

        # Horizonte de predicción
        n_dias = st.slider("Días en el futuro a predecir", min_value=1, max_value=21, value=7)

        # Matriz P^n
        from numpy.linalg import matrix_power

        Pn = matrix_power(P, n_dias)

        # Variable donde guardaremos las probabilidades a mostrar
        probs_vector = None
        estado_inicial = None

        if modo == "Libre":
            estado_inicial = st.selectbox("Selecciona el estado inicial", list(states))
            idx = state_to_idx.get(estado_inicial, None)
            if idx is not None:
                probs_vector = Pn[idx, :]
            else:
                st.warning("El estado seleccionado no existe en la matriz de transición.")

        else:  # Empleado
            all_names = sorted(df_markov["Name"].dropna().unique())
            search_text = st.text_input("Buscar empleado:")

            if search_text:
                filtered_names = [n for n in all_names if search_text.lower() in n.lower()]
            else:
                filtered_names = all_names

            if not filtered_names:
                st.info("No se encontraron empleados con ese texto de búsqueda.")
            else:
                selected_name = st.selectbox("Selecciona el empleado", filtered_names)

                emp_df = df_markov[df_markov["Name"] == selected_name].copy()
                # Ordenar por fecha (o columnas temporales disponibles)
                if "Date" in emp_df.columns:
                    emp_df = emp_df.sort_values("Date")
                elif "Week" in emp_df.columns:
                    emp_df = emp_df.sort_values("Week")
                elif "DayOfMonth" in emp_df.columns:
                    emp_df = emp_df.sort_values("DayOfMonth")

                if emp_df.empty:
                    st.info("No hay datos para este empleado.")
                else:
                    ultimos = emp_df.tail(10)

                    st.markdown(f"**Últimos {len(ultimos)} registros de engagement de {selected_name}**")
                    if "Date" in ultimos.columns:
                        fig_emp = px.line(
                            ultimos,
                            x="Date",
                            y="Engagement",
                            markers=True,
                            title=f"Engagement Histórico: {selected_name}",
                        )
                    else:
                        fig_emp = px.line(
                            ultimos.reset_index(),
                            x=ultimos.reset_index().index,
                            y="Engagement",
                            markers=True,
                            title=f"Engagement Histórico: {selected_name}",
                        )
                    st.plotly_chart(fig_emp, use_container_width=True)

                    # Estado inicial para Markov: el último estado observado
                    last_state = ultimos[STATE_COL].iloc[-1]
                    estado_inicial = last_state
                    idx = state_to_idx.get(last_state, None)
                    if idx is not None:
                        probs_vector = Pn[idx, :]
                    else:
                        st.warning("El último estado del empleado no está en la matriz de transición.")

        if probs_vector is not None:
            # Mostrar resultados de la distribución a n días
            df_result = pd.DataFrame(
                {"Estado": states, "Probabilidad": probs_vector}
            ).sort_values("Probabilidad", ascending=False)

            # Gráfica de barras del top-k
            top_k = min(5, len(df_result))
            fig_bar = px.bar(
                df_result.head(top_k),
                x="Estado",
                y="Probabilidad",
                text="Probabilidad",
                title=f"Top {top_k} estados más probables en {n_dias} días",
            )
            fig_bar.update_traces(texttemplate="%{y:.2%}", textposition="outside")
            fig_bar.update_yaxes(title="Probabilidad")
            # Color de barras a rojo
            fig_bar.update_traces(marker_color="red")
            st.plotly_chart(fig_bar, use_container_width=True)


if __name__ == "__main__":
    main()