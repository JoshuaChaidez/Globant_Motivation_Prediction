import os
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

# -------------------------------------------------
# Configuración general de la página
# -------------------------------------------------
st.set_page_config(page_title="Globant – Engagement", layout="wide")

# -------------------------------------------------
# Rutas de archivos
# -------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_MAIN_PATH = os.path.join(BASE_DIR, "data_globant_clean.csv")  # exploración + Markov
DATA_NN_PATH = os.path.join(BASE_DIR, "data_globant_cnn.csv")      # datos NN
MODEL_NN_PATH = os.path.join(BASE_DIR, "my_model.keras")           # modelo NN
DICC_PATH = os.path.join(BASE_DIR, "diccionario_final.csv")              # EmployeeID ↔ Name

# Columnas categóricas para la NN
CATEGORICAL_NN_COLS: List[str] = [
    "Position",
    "Location",
    "Studio",
    "Client Tag",
    "Project Tag",
    "Team Name",
]

# -------------------------------------------------
# TensorFlow / Keras
# -------------------------------------------------
try:
    from tensorflow import keras  # type: ignore
    TF_AVAILABLE = True
    TF_ERROR = None
except Exception as e:
    TF_AVAILABLE = False
    TF_ERROR = str(e)


# =================================================
# 1. UTILIDADES: Carga de datos principal
# =================================================
@st.cache_data(show_spinner=True)
def load_main_data(path: str) -> pd.DataFrame:
    """
    Carga el CSV principal (exploración + Markov) y crea columnas temporales.
    """
    df = pd.read_csv(path)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        df["Week"] = df["Date"].dt.isocalendar().week
        df["DayOfMonth"] = df["Date"].dt.day
        df["DayOfWeek"] = df["Date"].dt.day_name()
    else:
        # Fallback, por si no hubiera Date
        if "Week" not in df.columns:
            df["Week"] = np.nan
        if "DayOfMonth" not in df.columns:
            df["DayOfMonth"] = np.nan
        if "DayOfWeek" not in df.columns:
            df["DayOfWeek"] = np.nan

    return df


# =================================================
# 2. UTILIDADES: Carga y pipeline de la NN
# =================================================
@st.cache_data(show_spinner=True)
def load_nn_data(path: str) -> pd.DataFrame:
    """
    Carga el CSV para la red neuronal y garantiza columnas de fecha básicas.
    """
    df = pd.read_csv(path)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        if "Month" not in df.columns:
            df["Month"] = df["Date"].dt.month
        if "Day" not in df.columns:
            df["Day"] = df["Date"].dt.day

    return df


@st.cache_data(show_spinner=True)
def load_diccionario(path: str) -> pd.DataFrame:
    """
    Carga el diccionario EmployeeID ↔ Name.
    Se espera que contenga al menos columnas: EmployeeID, Name
    """
    df = pd.read_csv(path)
    return df


def assign_15day_blocks_nn(group: pd.DataFrame) -> pd.DataFrame:
    """
    Asigna bloques de 15 filas por empleado (para la NN):
    0–14 -> 1, 15–29 -> 2, ...
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

    Devuelve:
      df_display: DataFrame con datos + 15_dias (para UI)
      feature_cols: lista de columnas de entrada
      scaler: MinMaxScaler
      X_scaled: np.ndarray (n_samples, n_features)
      model: modelo keras cargado
      class_labels: etiquetas de salida (Bajo, Medio, Alto, etc.)
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
        class_labels = [f"{i+1}" for i in range(n_outputs)]

    return df_display, feature_cols, scaler, X_scaled, model, class_labels


def predict_from_index_nn(idx: int, X_scaled: np.ndarray, model) -> np.ndarray:
    """
    Predice usando una fila existente en X_scaled (modo 'Empleado').
    """
    if idx < 0 or idx >= X_scaled.shape[0]:
        raise IndexError("Índice fuera de rango para X_scaled.")
    x = X_scaled[idx : idx + 1]
    probs = model.predict(x, verbose=0)[0]
    return probs


def predict_from_manual_nn(
    row_dict: dict,
    feature_cols: List[str],
    scaler: MinMaxScaler,
    model,
) -> np.ndarray:
    """
    Aplica las mismas transformaciones a una sola fila manual (modo 'Libre').
    """
    single_df = pd.DataFrame([row_dict])

    # One-hot de categóricas
    df_model = pd.get_dummies(single_df, columns=CATEGORICAL_NN_COLS, drop_first=False)

    # Eliminar etiqueta si está
    drop_cols = [c for c in ["Engagement_D", "Engagement_D_num"] if c in df_model.columns]
    features_df = df_model.drop(columns=drop_cols)

    # Asegurar todas las columnas esperadas por el modelo
    for col in feature_cols:
        if col not in features_df.columns:
            features_df[col] = 0.0

    features_df = features_df[feature_cols]

    x = features_df.astype(float).values
    x_scaled = scaler.transform(x)
    probs = model.predict(x_scaled, verbose=0)[0]
    return probs


# =================================================
# 3. MAIN APP
# =================================================
def main():
    st.title("Engagement Globant")

    # ---------------------------------------------
    # Cargar datos principales (exploración + Markov)
    # ---------------------------------------------
    try:
        df = load_main_data(DATA_MAIN_PATH)
    except FileNotFoundError:
        st.error(f"No se encontró el archivo `{DATA_MAIN_PATH}`.")
        return

    if "Engagement" not in df.columns:
        st.error("La columna `Engagement` no existe en el CSV principal.")
        return

    # ---------------------------------------------
    # Sidebar de filtros
    # ---------------------------------------------
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
                df_filtered = df_filtered[df_filtered[col] == selected]
        else:
            st.sidebar.warning(f"La columna `{col}` no existe en el CSV.")

    agg_type = st.sidebar.selectbox(
        "Tipo de agregación temporal:",
        ["Promedio semanal", "Promedio por día de la semana", "Promedio por día del mes"],
    )

    ignore_zero = st.sidebar.checkbox(
        "Excluir engagement <= 0 del promedio",
        value=True,
        help="Los valores <= 0 no cuentan para el promedio, pero sí colorean los puntos.",
    )

    if df_filtered.empty:
        st.warning("No hay datos que coincidan con los filtros seleccionados.")
        return

    df_vis = df_filtered.copy()
    df_vis["IsZero"] = (df_vis["Engagement"] <= 0).astype(int)

    if ignore_zero:
        df_vis.loc[df_vis["Engagement"] <= 0, "Engagement"] = np.nan

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
            order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            df_plot[group_col] = pd.Categorical(df_plot[group_col], categories=order, ordered=True)
            df_plot = df_plot.sort_values(group_col)
    else:
        group_col = "DayOfMonth"
        x_title = "Día del mes"
        df_plot = aggregate(df_vis, group_col)

    if df_plot.empty:
        st.warning("No hay datos agregados para la combinación de filtros y agregación.")
        return

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

    st.subheader("Evolución del engagement (filtrado)")

    fig = px.line(
        df_plot,
        x=group_col,
        y="Engagement_mean",
        markers=True,
        title="Engagement promedio",
    )
    fig.update_traces(line=dict(color="lightgray", width=2), marker=dict(size=10))

    colors = [get_color(p) for p in df_plot["PercentZero"]]
    fig.update_traces(marker=dict(color=colors))

    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="Engagement promedio",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # =================================================
    # Sección: Predicción con Red Neuronal
    # =================================================
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
                st.info("No hay datos para la NN.")
            else:
                # Intentar cargar diccionario
                try:
                    df_dicc = load_diccionario(DICC_PATH)
                except Exception as e:
                    st.error(f"No se pudo cargar `diccionario.csv`: {e}")
                    df_dicc = None

                if df_dicc is not None:
                    if not {"EmployeeID", "Name"}.issubset(df_dicc.columns):
                        st.error(
                            "El `diccionario.csv` debe tener columnas 'EmployeeID' y 'Name'. "
                            "Ajusta load_diccionario si los nombres son otros."
                        )
                        df_dicc = None

                if df_dicc is not None:
                    df_nn_merged = df_nn_display.merge(
                        df_dicc[["EmployeeID", "Name"]],
                        on="EmployeeID",
                        how="left",
                    )
                else:
                    df_nn_merged = df_nn_display.copy()
                    df_nn_merged["Name"] = df_nn_merged["EmployeeID"].astype(str)

                st.write(
                    "Selecciona modo de predicción NN: "
                    "**Empleado** (por nombre) o **Libre** (parámetros manuales)."
                )

                modo_nn = st.radio(
                    "Modo NN",
                    ["Empleado", "Libre"],
                    index=0,
                    horizontal=True,
                )

                probs_nn = None

                # ---------- MODO EMPLEADO ----------
                if modo_nn == "Empleado":
                    if "Name" not in df_nn_merged.columns:
                        st.error("No se encontró columna 'Name' tras el merge con el diccionario.")
                    else:
                        names = sorted(df_nn_merged["Name"].dropna().unique().tolist())
                        selected_name = st.selectbox("Empleado (por nombre)", names)

                        df_emp = df_nn_merged[df_nn_merged["Name"] == selected_name].copy()
                        df_emp = df_emp.sort_values("Date")

                        if df_emp.empty:
                            st.warning("No hay registros para este empleado en el dataset de NN.")
                        else:
                            date_options = df_emp["Date"].dt.strftime("%Y-%m-%d").tolist()
                            default_idx_nn = len(date_options) - 1 if date_options else 0

                            selected_date_str_nn = st.selectbox(
                                "Selecciona el registro",
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
                                    "Name": row_emp_nn.get("Name", None),
                                    "EmployeeID": row_emp_nn["EmployeeID"],
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

                                df_carac = pd.DataFrame.from_dict(
                                    carac_dict, orient="index", columns=["Valor"]
                                )
                                st.table(df_carac)

                                if st.button("RUN"):
                                    try:
                                        probs_nn = predict_from_index_nn(idx_global_nn, X_scaled, model)
                                    except Exception as e:
                                        st.error(f"Error al predecir con la NN: {e}")

                # ---------- MODO LIBRE ----------
                else:
                    pos = st.selectbox(
                        "Position",
                        sorted(df_nn_display["Position"].dropna().unique().tolist()),
                    )
                    loc = st.selectbox(
                        "Location",
                        sorted(df_nn_display["Location"].dropna().unique().tolist()),
                    )
                    studio = st.selectbox(
                        "Studio",
                        sorted(df_nn_display["Studio"].dropna().unique().tolist()),
                    )
                    client = st.selectbox(
                        "Client Tag",
                        sorted(df_nn_display["Client Tag"].dropna().unique().tolist()),
                    )
                    project = st.selectbox(
                        "Project Tag",
                        sorted(df_nn_display["Project Tag"].dropna().unique().tolist()),
                    )
                    team = st.selectbox(
                        "Team Name",
                        sorted(df_nn_display["Team Name"].dropna().unique().tolist()),
                    )

                    seniority_val = st.number_input(
                        "Seniority",
                        value=int(df_nn_display["Seniority"].median()),
                        step=1,
                    )

                    month_val = st.slider(
                        "Month",
                        1,
                        12,
                        int(df_nn_display["Month"].median()),
                    )
                    day_val = st.slider(
                        "Day",
                        1,
                        31,
                        int(df_nn_display["Day"].median()),
                    )


                    max_block = int(df_nn_display.get("15_dias", pd.Series([1])).max())
                    block_15 = st.slider("Bloque de 15 días (15_dias)", 1, max_block, 1)


                    if st.button("RUN"):
                        row_dict = {
                            "Position": pos,
                            "Location": loc,
                            "Studio": studio,
                            "Client Tag": client,
                            "Project Tag": project,
                            "Team Name": team,
                            "Seniority": int(seniority_val),
                            "Month": int(month_val),
                            "Day": int(day_val),
                            "EmployeeID": int(emp_manual),
                            "15_dias": int(block_15),
                            "Engagement_D": eng_text,
                        }
                        try:
                            probs_nn = predict_from_manual_nn(row_dict, feature_cols, scaler, model)
                        except Exception as e:
                            st.error(f"Error al predecir con datos manuales (NN): {e}")

                # ---------- Mostrar resultado NN ----------
                if probs_nn is not None:
                    probs_nn = np.array(probs_nn)
                    if probs_nn.ndim != 1:
                        st.error("La salida del modelo NN no tiene la forma esperada.")
                    else:
                        k = min(len(probs_nn), len(class_labels))
                        probs_nn = probs_nn[:k]
                        idx_max = int(np.argmax(probs_nn))
                        pred_label = class_labels[idx_max]

                        st.markdown("---")
                        st.markdown(
                            f"## Predicción NN: **{pred_label}**"
                        )



    # =================================================
    # Sección: Predicción con Cadena de Markov
    # =================================================
    st.markdown("---")
    st.subheader("Cadena de Markov")

    df_markov = df.copy()

    STATE_COL = "Engagement_bin"
    if STATE_COL not in df_markov.columns:
        n_states = 5
        df_markov[STATE_COL] = pd.cut(
            df_markov["Engagement"],
            bins=n_states,
            labels=[f"{i+1}" for i in range(n_states)],
            include_lowest=True,
        )

    df_markov = df_markov.dropna(subset=[STATE_COL])
    df_markov[STATE_COL] = df_markov[STATE_COL].astype(str)

    @st.cache_data(show_spinner=False)
    def compute_transition_matrix(df_in: pd.DataFrame, state_col: str, id_col: str = "Name"):
        sort_cols = []
        for col in ["Date", "Week", "DayOfMonth"]:
            if col in df_in.columns:
                sort_cols.append(col)

        if not sort_cols:
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
        return

    modo = st.radio("Selección para Markov", ["Empleado", "Libre"], horizontal=True)
    n_dias = st.slider("Días a futuro a predecir", min_value=1, max_value=21, value=7)

    from numpy.linalg import matrix_power
    Pn = matrix_power(P, n_dias)

    probs_vector = None
    estado_inicial = None

    if modo == "Libre":
        estado_inicial = st.selectbox("Selecciona el estado inicial", list(states))
        idx_m = state_to_idx.get(estado_inicial, None)
        if idx_m is not None:
            probs_vector = Pn[idx_m, :]
        else:
            st.warning("El estado seleccionado no existe en la matriz de transición.")
    else:
        if "Name" not in df_markov.columns:
            st.error("No existe columna 'Name' en el dataset para usar modo Empleado en Markov.")
            return

        all_names = sorted(df_markov["Name"].dropna().unique())
        search_text = st.text_input("Buscar empleado")

        if search_text:
            filtered_names = [n for n in all_names if search_text.lower() in n.lower()]
        else:
            filtered_names = all_names

        if not filtered_names:
            st.info("No se encontraron empleados con ese texto de búsqueda.")
            return

        selected_name_m = st.selectbox("Selecciona el empleado", filtered_names)

        emp_df = df_markov[df_markov["Name"] == selected_name_m].copy()
        if "Date" in emp_df.columns:
            emp_df = emp_df.sort_values("Date")
        elif "Week" in emp_df.columns:
            emp_df = emp_df.sort_values("Week")
        elif "DayOfMonth" in emp_df.columns:
            emp_df = emp_df.sort_values("DayOfMonth")

        if emp_df.empty:
            st.info("No hay datos para este empleado.")
            return

        ultimos = emp_df.tail(10)

        st.markdown(f"**Últimos {len(ultimos)} registros de engagement de {selected_name_m}**")
        if "Date" in ultimos.columns:
            fig_emp = px.line(
                ultimos,
                x="Date",
                y="Engagement",
                markers=True,
                title=f"Engagement Histórico: {selected_name_m}",
            )
        else:
            fig_emp = px.line(
                ultimos.reset_index(),
                x=ultimos.reset_index().index,
                y="Engagement",
                markers=True,
                title=f"Engagement Histórico: {selected_name_m}",
            )
        st.plotly_chart(fig_emp, use_container_width=True)

        last_state = ultimos[STATE_COL].iloc[-1]
        estado_inicial = last_state
        idx_m = state_to_idx.get(last_state, None)
        if idx_m is not None:
            probs_vector = Pn[idx_m, :]
        else:
            st.warning("El último estado del empleado no está en la matriz de transición.")

    if probs_vector is not None:
        df_result = pd.DataFrame(
            {"Estado": states, "Probabilidad": probs_vector}
        ).sort_values("Probabilidad", ascending=False)

        top_k = min(5, len(df_result))
        fig_bar_m = px.bar(
            df_result.head(top_k),
            x="Estado",
            y="Probabilidad",
            text="Probabilidad",
            title=f"Distribucion en {n_dias} días",
        )
        fig_bar_m.update_traces(texttemplate="%{y:.2%}", textposition="outside")
        fig_bar_m.update_yaxes(title="Probabilidad")
        st.plotly_chart(fig_bar_m, use_container_width=True)


if __name__ == "__main__":
    main()