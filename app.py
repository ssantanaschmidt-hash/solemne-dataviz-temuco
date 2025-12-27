import json
from typing import Dict, Any, Optional

import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

CKAN_BASE = "https://datos.gob.cl/api/3/action"


# ---------- Helpers API ----------
def ckan_get(action: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Dict[str, Any]:
    url = f"{CKAN_BASE}/{action}"
    try:
        r = requests.get(url, params=params or {}, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if not data.get("success", False):
            raise RuntimeError(data.get("error", "Respuesta CKAN sin success=True"))
        return data["result"]
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error de conexión a la API: {e}")
    except json.JSONDecodeError:
        raise RuntimeError("La respuesta no es JSON válido (posible error del servidor).")


@st.cache_data(show_spinner=False, ttl=600)
def search_packages(query: str, rows: int = 20) -> Dict[str, Any]:
    return ckan_get("package_search", {"q": query, "rows": rows})


@st.cache_data(show_spinner=False, ttl=600)
def package_show(package_id: str) -> Dict[str, Any]:
    return ckan_get("package_show", {"id": package_id})


def _try_read_csv(url: str) -> pd.DataFrame:
    """
    Lee CSV intentando primero ';' (muy común en Chile) y luego ','.
    Usa header=None para datasets que vienen sin encabezado (como DGAC).
    """
    try:
        return pd.read_csv(url, sep=";", engine="python", header=None)
    except Exception:
        pass

    try:
        return pd.read_csv(url, header=None)
    except Exception:
        pass

    return pd.read_csv(url, sep=";", engine="python", header=None, on_bad_lines="skip")


def _autocast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            s = out[col].astype(str).str.replace(",", ".", regex=False)
            converted = pd.to_numeric(s, errors="coerce")
            if (converted.notna().mean() if len(converted) else 0) >= 0.4:
                out[col] = converted
    return out


def _maybe_fix_dgac_temperaturas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dataset DGAC 'Temperaturas por estación' (9 columnas, sin encabezados):
    0 codigo_estacion | 1 estacion | 2 latitud | 3 altitud | 4 anio | 5 mes | 6 dia | 7 tmin | 8 tmax
    """
    if df.shape[1] == 9:
        df = df.copy()
        df.columns = [
            "codigo_estacion",
            "estacion",
            "latitud",
            "altitud",
            "anio",
            "mes",
            "dia",
            "tmin",
            "tmax",
        ]
        df = _autocast_numeric(df)

        # fecha
        try:
            df["fecha"] = pd.to_datetime(
                df[["anio", "mes", "dia"]].astype(int).astype(str).agg("-".join, axis=1),
                errors="coerce"
            )
        except Exception:
            df["fecha"] = pd.NaT

        return df

    return _autocast_numeric(df)


def safe_read_table_from_resource(resource: Dict[str, Any], max_rows: int = 200_000) -> pd.DataFrame:
    if resource.get("datastore_active") and resource.get("id"):
        rid = resource["id"]
        res = ckan_get("datastore_search", {"resource_id": rid, "limit": min(max_rows, 10000)})
        df = pd.DataFrame(res.get("records", []))
        return _autocast_numeric(df)

    url = resource.get("url")
    fmt = (resource.get("format") or "").strip().lower()
    if not url:
        raise RuntimeError("El recurso no tiene URL para descargar.")

    if fmt == "csv" or url.lower().endswith(".csv"):
        df = _try_read_csv(url)
        return _maybe_fix_dgac_temperaturas(df)

    if fmt == "json" or url.lower().endswith(".json"):
        df = pd.read_json(url)
        return _autocast_numeric(df)

    # fallback
    try:
        df = _try_read_csv(url)
        return _maybe_fix_dgac_temperaturas(df)
    except Exception:
        df = pd.read_json(url)
        return _autocast_numeric(df)


# ---------- UI ----------
st.set_page_config(page_title="DataViz Solemne II - datos.gob.cl", layout="wide")
st.title("Proyecto DataViz (Solemne II) – API datos.gob.cl (CKAN)")

st.write(
    "App que consume la API REST pública de datos.gob.cl (CKAN), permite buscar datasets, cargar recursos "
    "y analizarlos con pandas + matplotlib en una interfaz Streamlit."
)

with st.sidebar:
    st.header("1) Buscar dataset")
    q = st.text_input("Palabra clave", value="temperatura")
    rows = st.slider("Cantidad de resultados", min_value=5, max_value=50, value=20, step=5)
    do_search = st.button("Buscar", use_container_width=True)

if do_search:
    st.session_state["search_result"] = search_packages(q, rows)

result = st.session_state.get("search_result")
if not result:
    st.info("Escribe una palabra clave y presiona **Buscar**.")
    st.stop()

packages = result.get("results", [])
if not packages:
    st.warning("No se encontraron resultados.")
    st.stop()

st.subheader("Resultados encontrados")
df_list = pd.DataFrame([{
    "Título": p.get("title"),
    "Nombre (name)": p.get("name"),
    "Organización": (p.get("organization") or {}).get("title"),
    "Última modificación": p.get("metadata_modified"),
    "Recursos": p.get("num_resources", 0),
} for p in packages])
st.dataframe(df_list, use_container_width=True, hide_index=True)

st.divider()
st.subheader("2) Elegir dataset y recurso")

names = [p.get("name") for p in packages]
dataset_name = st.selectbox("Dataset (name)", options=names, index=0)

pkg = package_show(dataset_name)
resources = pkg.get("resources", [])
if not resources:
    st.warning("Este dataset no tiene recursos.")
    st.stop()

res_options = [{
    "name": r.get("name") or "(sin nombre)",
    "format": (r.get("format") or "").upper(),
    "datastore_active": bool(r.get("datastore_active")),
    "url": r.get("url"),
    "resource": r
} for r in resources]

st.dataframe(pd.DataFrame([{
    "Recurso": x["name"],
    "Formato": x["format"],
    "Datastore": x["datastore_active"],
    "URL": x["url"],
} for x in res_options]), use_container_width=True, hide_index=True)

choice = st.selectbox(
    "Selecciona un recurso por nombre",
    options=list(range(len(res_options))),
    format_func=lambda i: f'{res_options[i]["name"]} ({res_options[i]["format"] or "?"})'
)
resource = res_options[choice]["resource"]

st.divider()
st.subheader("3) Cargar, filtrar y analizar")

with st.spinner("Cargando datos..."):
    try:
        df = safe_read_table_from_resource(resource)
    except Exception as e:
        st.error(f"No se pudo cargar el recurso. Detalle: {e}")
        st.stop()

st.success(f"Datos cargados ✅  Filas: {len(df):,} | Columnas: {len(df.columns)}")

# ---------- Filtros ----------
st.subheader("Filtros")

df_filtered = df.copy()

# Filtro estación
if "estacion" in df_filtered.columns:
    search_txt = st.text_input("Buscar estación (ej: Temuco)", value="Temuco")

    estaciones = df_filtered["estacion"].astype(str).dropna().unique().tolist()
    estaciones.sort()

    if search_txt.strip():
        estaciones_match = [e for e in estaciones if search_txt.lower() in e.lower()]
    else:
        estaciones_match = estaciones

    if estaciones_match:
        selected_estaciones = st.multiselect(
            "Selecciona estación(es)",
            options=estaciones_match,
            default=estaciones_match[:1]
        )
        if selected_estaciones:
            df_filtered = df_filtered[df_filtered["estacion"].astype(str).isin(selected_estaciones)]
    else:
        st.warning("No se encontraron estaciones con ese texto.")
else:
    st.info("No existe columna 'estacion' para filtrar.")

# Filtro fechas (si existe fecha)
if "fecha" in df_filtered.columns:
    df_filtered["fecha"] = pd.to_datetime(df_filtered["fecha"], errors="coerce")
    df_filtered = df_filtered[df_filtered["fecha"].notna()].copy()

    if not df_filtered.empty:
        min_date = df_filtered["fecha"].min().date()
        max_date = df_filtered["fecha"].max().date()

        col1, col2 = st.columns(2)
        with col1:
            date_from = st.date_input("Desde", value=min_date, min_value=min_date, max_value=max_date)
        with col2:
            date_to = st.date_input("Hasta", value=max_date, min_value=min_date, max_value=max_date)

        if date_from > date_to:
            st.error("La fecha 'Desde' no puede ser mayor que 'Hasta'.")
        else:
            df_filtered = df_filtered[
                (df_filtered["fecha"].dt.date >= date_from) &
                (df_filtered["fecha"].dt.date <= date_to)
            ].copy()
else:
    st.info("No existe columna 'fecha' para filtrar por rango.")

st.write("Filas después de filtros:", f"{len(df_filtered):,}")

# Preview + tipos + describe
st.write("Vista previa:")
st.dataframe(df_filtered.head(30), use_container_width=True)

st.write("Tipos de datos detectados:")
st.dataframe(pd.DataFrame({"columna": df_filtered.columns, "dtype": [str(t) for t in df_filtered.dtypes]}),
             use_container_width=True)

st.write("Resumen (describe) para columnas numéricas:")
num_cols = df_filtered.select_dtypes(include="number").columns.tolist()
if num_cols:
    st.dataframe(df_filtered[num_cols].describe().T, use_container_width=True)
else:
    st.info("No se detectaron columnas numéricas para estadísticos.")

st.divider()
st.subheader("4) Visualización (matplotlib)")

if df_filtered.empty:
    st.warning("No hay datos luego de aplicar filtros. Cambia estación o rango de fechas.")
    st.stop()

cols = df_filtered.columns.tolist()

# Defaults para gráfico
default_x = cols.index("fecha") if "fecha" in cols else 0
col_x = st.selectbox("Eje X (categoría o fecha)", options=cols, index=default_x)

if num_cols:
    if "tmax" in num_cols:
        default_y = num_cols.index("tmax")
    elif "tmin" in num_cols:
        default_y = num_cols.index("tmin")
    else:
        default_y = 0

    col_y = st.selectbox("Eje Y (numérica)", options=num_cols, index=default_y)
else:
    st.error("No hay columnas numéricas para graficar.")
    st.stop()

plot_mode = st.radio("Tipo de gráfico", options=["Línea", "Barras"], horizontal=True)

# Agregación (para que se vea más limpio)
agg_mode = st.selectbox("Agregación de la serie (para mejor visualización)", ["Diario", "Semanal", "Mensual"], index=0)

limit = st.slider("Máximo de filas a graficar (rendimiento)", 100, 50000, 5000, 100)
df_plot = df_filtered.copy()

# Ordenar por fecha si corresponde
if col_x == "fecha":
    df_plot = df_plot.sort_values("fecha")

# Agregar si es fecha + agregación
if col_x == "fecha" and agg_mode != "Diario":
    df_plot = df_plot[["fecha", col_y]].dropna().copy()
    df_plot = df_plot.set_index("fecha")

    if agg_mode == "Semanal":
        df_plot = df_plot.resample("W").mean()
    elif agg_mode == "Mensual":
        df_plot = df_plot.resample("M").mean()

    df_plot = df_plot.reset_index()

df_plot = df_plot.head(limit)

fig = plt.figure()
ax = fig.add_subplot(111)

try:
    if plot_mode == "Línea":
        ax.plot(df_plot[col_x], df_plot[col_y])
    else:
        ax.bar(df_plot[col_x], df_plot[col_y])

    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    ax.set_title(f"Serie de {col_y} ({agg_mode})")

    # Mejorar legibilidad del eje X si es fecha
    if col_x == "fecha":
        fig.autofmt_xdate()

    st.pyplot(fig, clear_figure=True)
except Exception as e:
    st.error(f"No se pudo graficar: {e}")

st.divider()
st.subheader("5) Conclusiones (guía)")
st.write(
    "- ¿Qué patrón se observa en la temperatura máxima/mínima?\n"
    "- ¿Cambian los valores al agrupar semanal o mensual?\n"
    "- ¿Hay días o semanas con valores anómalos?\n"
    "- ¿Qué conclusión se obtiene para Temuco en el período filtrado?"
)
