"""
MARS EXPLORER | NASA Hackathon 2026 | Gale Crater Mission Dashboard
Run:  python -m streamlit run app.py

One file that combines the elevation, slope, hazard, 3D and route previews,
the MOLA metadata check and the region test into a single dashboard.
"""
import base64
import csv
import heapq
import io
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from pds4_tools import pds4_read

try:
    from scipy.ndimage import distance_transform_edt
except Exception:
    distance_transform_edt = None

st.set_page_config(
    page_title="MARS EXPLORER | Mission Dashboard",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# PATHS AND CONSTANTS
# =========================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
IMAGERY_DIR = DATA_DIR / "imagery"
GEOLOGY_DIR = DATA_DIR / "geology"
THERMAL_DIR = DATA_DIR / "thermal"
MOLA_XML = DATA_DIR / "elevation" / "mola" / "megt00n090hb.xml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

GALE_LAT, GALE_LON = -5.4, 137.8
PPD = 128                      # MOLA pixels per degree
TILE_LAT_MAX = 0.0             # tile megt00n090hb starts at the equator
TILE_LON_MIN = 90.0            # and at 90 degrees east
MARS_RADIUS_M = 3_396_190.0

# Two study areas. "stride" is how coarsely the route search samples the grid.
AREAS = {
    "Close-up (about 150 km wide)": {"kind": "window", "half": 160, "stride": 1},
    "Wide region (about 940 km wide)": {
        "kind": "box", "lat": (-17.3, -1.5), "lon": (129.9, 145.7), "stride": 4},
}

HAZARD_LABELS = {0: "Low (under 5°)", 1: "Moderate (5-15°)", 2: "High (15-30°)", 3: "Very high (30° and up)"}
HAZARD_SHORT = {0: "Low", 1: "Moderate", 2: "High", 3: "Very high"}
HAZARD_COLORS = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
HAZARD_COLORSCALE = [
    [0.00, HAZARD_COLORS[0]], [0.25, HAZARD_COLORS[0]],
    [0.25, HAZARD_COLORS[1]], [0.50, HAZARD_COLORS[1]],
    [0.50, HAZARD_COLORS[2]], [0.75, HAZARD_COLORS[2]],
    [0.75, HAZARD_COLORS[3]], [1.00, HAZARD_COLORS[3]],
]
MARS_SCALE = [[0.0, "#21130F"], [0.25, "#71321F"], [0.50, "#B94E2C"], [0.75, "#E18A48"], [1.0, "#F5D6A5"]]

# name: (start lat, start lon, goal lat, goal lon)
PRESETS = {
    "Landing site to Mount Sharp base": (-4.59, 137.44, -5.05, 137.80),
    "Landing site to North crater wall": (-4.59, 137.44, -4.30, 137.50),
    "Long traverse (wide region only)": (-3.5, 132.0, -8.0, 140.0),
}

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#F1E3D4"),
    margin=dict(l=10, r=10, t=50, b=10),
)

# =========================================================
# STYLE
# =========================================================
BACKGROUND_IMAGE = BASE_DIR / "assets" / "mars.jpg"
BACKGROUND_CSS = "none"
if BACKGROUND_IMAGE.exists():
    _b64 = base64.b64encode(BACKGROUND_IMAGE.read_bytes()).decode("ascii")
    BACKGROUND_CSS = f"url(data:image/jpeg;base64,{_b64})"

CSS = """
<style>
:root{color-scheme:dark}
#MainMenu,footer,header{visibility:hidden}
.stApp{background:#160F0D;color:#F2E2CF}
.stApp::before{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;
 background-image:linear-gradient(105deg,rgba(18,12,11,.72) 0%,rgba(35,17,13,.60) 48%,rgba(57,25,16,.38) 100%),__BG__;
 background-size:cover;background-position:center;background-repeat:no-repeat}
[data-testid="stAppViewContainer"],[data-testid="stHeader"]{background:transparent}
[data-testid="stMain"]{position:relative;z-index:1}
[data-testid="stMainBlockContainer"]{max-width:1600px;padding:1.8rem 1rem 2.5rem}
[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li{line-height:1.62}
[data-testid="stSidebar"]{background:linear-gradient(180deg,rgba(39,25,20,.98),rgba(22,15,13,.98));
 border-right:1px solid rgba(217,163,91,.25);box-shadow:8px 0 30px rgba(0,0,0,.25)}
[data-testid="stSidebar"] *{color:#F0E1D2}
[data-testid="stSidebar"] [data-testid="stRadio"] label{border-radius:11px;padding:9px 11px;margin:1px 0}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{background:rgba(216,91,50,.14)}
.hero{background:linear-gradient(112deg,rgba(31,19,15,.96),rgba(91,39,25,.88),rgba(160,70,39,.52));
 border:1px solid rgba(240,138,75,.42);border-radius:22px;padding:34px 38px;margin-bottom:22px;
 box-shadow:0 18px 55px rgba(0,0,0,.32);backdrop-filter:blur(16px)}
.hero h1{margin:0;font-size:clamp(30px,4vw,46px);font-weight:850;color:#FFF3E7;letter-spacing:1.1px}
.hero p{margin:9px 0 0 0;color:#E9CDB7;font-size:16px}
.badge{display:inline-block;padding:5px 12px;margin:14px 8px 0 0;border-radius:999px;font-size:12px;font-weight:650;
 background:rgba(242,226,207,.09);border:1px solid rgba(242,226,207,.22);color:#F7E8D8}
.kpi{background:linear-gradient(145deg,rgba(58,36,27,.95),rgba(29,19,16,.95));
 border:1px solid rgba(217,163,91,.24);border-radius:17px;padding:18px 20px;box-shadow:0 10px 30px rgba(0,0,0,.24)}
.kpi .label{color:#C7A994;font-size:11px;text-transform:uppercase;letter-spacing:1.3px}
.kpi .value{color:#FFF3E7;font-size:26px;font-weight:850;margin-top:5px}
.kpi .sub{color:#D9A35B;font-size:12px;margin-top:3px}
.step{background:rgba(43,28,22,.93);border:1px solid rgba(217,163,91,.20);border-left:4px solid #D85B32;
 border-radius:13px;padding:15px 17px;height:100%}
.step b{color:#E8B36F}
.step span{color:#D0B7A3;font-size:13px}
.section-title{font-size:26px;font-weight:850;color:#F0A45D;margin:7px 0 3px 0}
.section-sub{color:#D0B7A3;margin-bottom:15px}
.note{background:rgba(112,65,36,.32);border:1px solid rgba(217,163,91,.30);border-radius:13px;
 padding:13px 16px;color:#F1DFCB;font-size:14px;margin:6px 0}
.legend span{display:inline-block;margin:4px 14px 4px 0;font-size:13px}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
[data-testid="stMetric"]{background:rgba(43,28,22,.91);border:1px solid rgba(217,163,91,.20);
 padding:13px 16px;border-radius:14px}
[data-testid="stExpander"]{background:rgba(34,22,18,.90);border:1px solid rgba(217,163,91,.20);border-radius:13px}
.stButton>button,[data-testid="stDownloadButton"] button{border-radius:10px;border:1px solid rgba(240,138,75,.62);
 background:linear-gradient(100deg,#A9442B,#D85B32);color:#FFF3E7;font-weight:700}
[data-testid="stPlotlyChart"]{background:rgba(25,16,13,.78);border:1px solid rgba(217,163,91,.17);
 border-radius:14px;padding:5px}
[data-testid="stVerticalBlockBorderWrapper"]{background:rgba(38,25,20,.76);
 border-color:rgba(217,163,91,.28)!important;border-radius:15px}
[data-testid="stImage"] img{border-radius:10px}
.footer{color:#C7A994;font-size:12px;text-align:center;padding:20px}
</style>
""".replace("__BG__", BACKGROUND_CSS)
st.markdown(CSS, unsafe_allow_html=True)


# =========================================================
# UI HELPERS
# =========================================================
def kpi(label, value, sub=""):
    return (f'<div class="kpi"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="sub">{sub}</div></div>')


def title(text, sub=""):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="section-sub">{sub}</div>', unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def kpi_row(items):
    for col, (label, value, sub) in zip(st.columns(len(items)), items):
        col.markdown(kpi(label, value, sub), unsafe_allow_html=True)


def legend(items):
    html = "".join(f'<span><i style="background:{c}"></i>{t}</span>' for c, t in items)
    st.markdown(f'<div class="legend">{html}</div>', unsafe_allow_html=True)


def _version_tuple():
    try:
        return tuple(int(x) for x in st.__version__.split(".")[:2])
    except Exception:
        return (1, 0)


# Newer Streamlit replaced use_container_width with width="stretch"
STRETCH = {"width": "stretch"} if _version_tuple() >= (1, 50) else {"use_container_width": True}


def show_plot(fig):
    st.plotly_chart(fig, **STRETCH)


def html_download(fig, label, filename, key):
    """Standalone interactive HTML copy of a figure (replaces the old preview scripts' write_html)."""
    st.download_button(label, fig.to_html(include_plotlyjs="cdn"), filename, "text/html", key=key)


# =========================================================
# IMAGE GRIDS
# =========================================================
def get_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.name.lower(),
    )


@st.cache_data(show_spinner=False)
def _tile_bytes(path, width=720, height=480):
    """Crop every image to the same tile size so the grid lines up."""
    try:
        from PIL import Image, ImageOps
        with Image.open(path) as image:
            image = ImageOps.fit(image.convert("RGB"), (width, height))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=88)
            return buffer.getvalue()
    except Exception:
        return path


if hasattr(st, "dialog"):
    @st.dialog("Image viewer", width="large")
    def _viewer(path):
        st.image(path, **STRETCH)
        st.caption(Path(path).name)
else:
    _viewer = None


def image_grid(images, columns=3, key="grid"):
    """Equal-size tiles in rows of `columns`, with a button to enlarge each one."""
    if not images:
        st.info("No images in this folder yet.")
        return
    for start in range(0, len(images), columns):
        cols = st.columns(columns)
        for col, path in zip(cols, images[start:start + columns]):
            with col:
                with st.container(border=True):
                    st.image(_tile_bytes(str(path)), **STRETCH)
                    st.markdown(f"**{path.stem.replace('_', ' ')}**")
                    if _viewer and st.button("Enlarge", key=f"{key}_{path.name}"):
                        _viewer(str(path))


def gallery(folder, label, compare=False):
    images = get_images(folder)
    if not images:
        st.warning(f"No image files found in `{Path(folder).relative_to(BASE_DIR)}`.")
        st.caption("Add JPG, PNG or WEBP files to that folder and reload.")
        return
    c1, c2, c3 = st.columns([3, 1, 1])
    search = c1.text_input("Search by file name", key=f"{label}_search")
    columns = c2.selectbox("Columns", [2, 3, 4], index=1, key=f"{label}_cols")
    shown = [p for p in images if search.lower() in p.name.lower()]
    c3.metric("Images", len(shown))
    image_grid(shown, columns, key=label)

    if compare and len(images) >= 2:
        with st.expander("Compare two images side by side"):
            a, b = st.columns(2)
            first = a.selectbox("Left", images, format_func=lambda p: p.name, key=f"{label}_l")
            second = b.selectbox("Right", images, index=1, format_func=lambda p: p.name, key=f"{label}_r")
            a.image(str(first), **STRETCH)
            b.image(str(second), **STRETCH)


# =========================================================
# TERRAIN DATA
# =========================================================
@st.cache_data(show_spinner="Loading NASA MOLA elevation data...")
def load_mola_data(xml_path, modified_time):
    product = pds4_read(str(xml_path), quiet=True)
    elevation = np.asarray(product[0].data, dtype=np.float32)
    elevation[~np.isfinite(elevation)] = np.nan
    return elevation


def pixel_size_m(lat_deg):
    """Ground size of one MOLA pixel in metres: (east-west, north-south)."""
    dy = math.pi * MARS_RADIUS_M / (180.0 * PPD)
    return dy * math.cos(math.radians(lat_deg)), dy


def compute_slope(terrain, dx, dy):
    valid = np.isfinite(terrain)
    if not valid.any():
        raise ValueError("No valid elevation data found.")
    if valid.all():
        filled = terrain
    elif distance_transform_edt is not None:
        idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        filled = terrain[tuple(idx)]
    else:
        filled = np.where(valid, terrain, np.nanmedian(terrain))
    gy, gx = np.gradient(filled, dy, dx)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    slope[~valid] = np.nan
    return slope


def classify_hazard(slope):
    """0 low, 1 moderate, 2 high, 3 very high, -1 unknown."""
    hazard = np.full(slope.shape, -1, dtype=np.int8)
    ok = np.isfinite(slope)
    hazard[ok & (slope < 5)] = 0
    hazard[ok & (slope >= 5) & (slope < 15)] = 1
    hazard[ok & (slope >= 15) & (slope < 30)] = 2
    hazard[ok & (slope >= 30)] = 3
    return hazard


@st.cache_data(show_spinner="Processing terrain...")
def load_analysis(area_key):
    if not MOLA_XML.exists():
        raise FileNotFoundError(f"MOLA XML file not found: {MOLA_XML}")
    elevation = load_mola_data(str(MOLA_XML), MOLA_XML.stat().st_mtime)
    area = AREAS[area_key]

    if area["kind"] == "window":
        row_c = round(-GALE_LAT * PPD)
        col_c = round((GALE_LON - TILE_LON_MIN) * PPD)
        r0, r1 = row_c - area["half"], row_c + area["half"]
        c0, c1 = col_c - area["half"], col_c + area["half"]
    else:
        (lat_lo, lat_hi), (lon_lo, lon_hi) = area["lat"], area["lon"]
        r0, r1 = int(np.floor((TILE_LAT_MAX - lat_hi) * PPD)), int(np.ceil((TILE_LAT_MAX - lat_lo) * PPD))
        c0, c1 = int(np.floor((lon_lo - TILE_LON_MIN) * PPD)), int(np.ceil((lon_hi - TILE_LON_MIN) * PPD))

    r0, c0 = max(0, r0), max(0, c0)
    r1, c1 = min(elevation.shape[0], r1), min(elevation.shape[1], c1)
    terrain = elevation[r0:r1, c0:c1].copy()
    lats = -(np.arange(r0, r1) + 0.5) / PPD            # row 0 is the northern edge
    lons = TILE_LON_MIN + (np.arange(c0, c1) + 0.5) / PPD
    center_lat = float(lats.mean())
    dx, dy = pixel_size_m(center_lat)
    slope = compute_slope(terrain, dx, dy)
    return {"terrain": terrain, "lats": lats, "lons": lons, "slope": slope,
            "hazard": classify_hazard(slope), "dx": dx, "dy": dy,
            "center_lat": center_lat, "center_lon": float(lons.mean())}


def require_analysis(area_key):
    try:
        return load_analysis(area_key)
    except Exception as error:
        st.error(f"Terrain loading failed: {error}")
        st.caption(f"Expected the MOLA files in `{MOLA_XML.parent}`")
        st.stop()


def sub(a, s):
    """Every s-th cell; pixel size grows by s."""
    if s == 1:
        return a
    return {"terrain": a["terrain"][::s, ::s], "lats": a["lats"][::s], "lons": a["lons"][::s],
            "slope": a["slope"][::s, ::s], "hazard": a["hazard"][::s, ::s],
            "dx": a["dx"] * s, "dy": a["dy"] * s,
            "center_lat": a["center_lat"], "center_lon": a["center_lon"]}


def disp(a, limit=450):
    """Thinned copy used for drawing, so big areas stay fast."""
    return sub(a, max(1, max(a["terrain"].shape) // limit))


def terrain_statistics(terrain):
    valid = terrain[np.isfinite(terrain)]
    if valid.size == 0:
        return dict.fromkeys(["minimum", "maximum", "mean"], float("nan"))
    return {"minimum": float(valid.min()), "maximum": float(valid.max()), "mean": float(valid.mean())}


def hazard_table(hazard):
    total = max(1, np.count_nonzero(hazard >= 0))
    return [{"Class": HAZARD_LABELS[i], "Pixels": f"{np.count_nonzero(hazard == i):,}",
             "Share": f"{np.count_nonzero(hazard == i) / total * 100:.2f}%"} for i in range(4)]


def hazard_share(hazard):
    total = max(1, np.count_nonzero(hazard >= 0))
    return {i: np.count_nonzero(hazard == i) / total * 100 for i in range(4)}


def read_mola_metadata():
    """Coordinate tags from the PDS4 label (what inspect_mola_metadata.py printed)."""
    wanted = {"west_bounding_coordinate", "east_bounding_coordinate", "north_bounding_coordinate",
              "south_bounding_coordinate", "pixel_resolution_x", "pixel_resolution_y",
              "minimum_latitude", "maximum_latitude"}
    rows = []
    try:
        for element in ET.parse(MOLA_XML).getroot().iter():
            tag = element.tag.split("}")[-1]
            if tag in wanted and element.text:
                rows.append({"Field": tag, "Value": element.text.strip()})
    except Exception:
        pass
    return rows


# =========================================================
# MAPS AND 3D
# =========================================================
def _finish(fig, name, compact, height):
    fig.update_layout(title=name, xaxis_title="Longitude (°E)", yaxis_title="Latitude (°)",
                      height=height if not compact else 330, **PLOT_LAYOUT)
    return fig       # latitudes decrease down the rows, so north is already at the top


def elevation_map(a, compact=False):
    d = disp(a)
    fig = go.Figure(go.Heatmap(
        x=d["lons"], y=d["lats"], z=d["terrain"], colorscale=MARS_SCALE,
        colorbar=dict(title="Elevation (m)"),
        hovertemplate="Longitude: %{x:.3f}°E<br>Latitude: %{y:.3f}°<br>Elevation: %{z:.0f} m<extra></extra>"))
    if d["lons"].min() <= GALE_LON <= d["lons"].max() and d["lats"].min() <= GALE_LAT <= d["lats"].max():
        fig.add_trace(go.Scatter(x=[GALE_LON], y=[GALE_LAT], mode="markers+text", text=["Gale Crater"],
                                 textposition="top right", name="Gale Crater",
                                 marker=dict(size=11, color="white", line=dict(color="black", width=2))))
    fig.update_layout(showlegend=False)
    return _finish(fig, "Elevation (brighter means higher)", compact, 650)


def slope_map(a, compact=False):
    d = disp(a)
    fig = go.Figure(go.Heatmap(
        x=d["lons"], y=d["lats"], z=d["slope"], colorscale="Turbo", zmin=0, zmax=40,
        colorbar=dict(title="Slope (°)"),
        hovertemplate="Longitude: %{x:.3f}°E<br>Latitude: %{y:.3f}°<br>Slope: %{z:.1f}°<extra></extra>"))
    return _finish(fig, "Slope in degrees (red and orange are steep)", compact, 650)


def hazard_map(a, compact=False):
    d = disp(a)
    z = np.where(d["hazard"] >= 0, d["hazard"], np.nan).astype(float)
    fig = go.Figure(go.Heatmap(
        x=d["lons"], y=d["lats"], z=z, zmin=0, zmax=3, colorscale=HAZARD_COLORSCALE,
        colorbar=dict(title="Hazard", tickvals=[0, 1, 2, 3], ticktext=list(HAZARD_SHORT.values())),
        hovertemplate="Longitude: %{x:.3f}°E<br>Latitude: %{y:.3f}°<extra></extra>"))
    return _finish(fig, "Hazard classes from slope", compact, 650)


def terrain_3d(a, exaggeration):
    step = max(1, max(a["terrain"].shape) // 150)
    t = a["terrain"][::step, ::step]
    lon_g, lat_g = np.meshgrid(a["lons"][::step], a["lats"][::step])
    x_km = (lon_g - a["center_lon"]) * a["dx"] * PPD / 1000.0
    y_km = (lat_g - a["center_lat"]) * a["dy"] * PPD / 1000.0
    fig = go.Figure(go.Surface(
        x=x_km, y=y_km, z=t / 1000.0 * exaggeration, surfacecolor=t, colorscale=MARS_SCALE,
        colorbar=dict(title="Elevation (m)"), connectgaps=False,
        hovertemplate="Elevation: %{surfacecolor:.0f} m<extra></extra>"))
    fig.update_layout(
        title=f"3D terrain (heights stretched x{exaggeration:g})",
        scene=dict(xaxis_title="West to East (km)", yaxis_title="South to North (km)",
                   zaxis_title=f"Elevation (km x {exaggeration:g})", aspectmode="data",
                   camera=dict(eye=dict(x=1.6, y=-1.6, z=1.1))),
        height=700, **PLOT_LAYOUT)
    return fig


# =========================================================
# ROUTE PLANNING
# =========================================================
DEFAULT_CFG = {"slope_limit": 30.0, "slope_weight": 3.0, "elevation_weight": 0.15,
               "hazard_weight": 1.0, "high_penalty": 1000.0}
HAZARD_FACTOR = {0: 1.0, 1: 2.0, 2: 5.0}
NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def cell_distance_m(p, q, g):
    return math.hypot((q[0] - p[0]) * g["dy"], (q[1] - p[1]) * g["dx"])


def terrain_cost(p, q, g, cfg):
    """Step cost. Every factor is at least 1, so plain distance is a valid A* lower bound."""
    slope_factor = 1.0 + cfg["slope_weight"] * (g["slope"][q] / 30.0) ** 2
    elev_factor = 1.0 + cfg["elevation_weight"] * abs(g["terrain"][q] - g["terrain"][p])
    base = HAZARD_FACTOR.get(int(g["hazard"][q]), cfg["high_penalty"])
    hazard_factor = 1.0 + cfg["hazard_weight"] * (base - 1.0)
    return float(cell_distance_m(p, q, g) * slope_factor * elev_factor * hazard_factor)


def blocked_mask(g, cfg):
    return (~np.isfinite(g["terrain"]) | ~np.isfinite(g["slope"]) | (g["hazard"] < 0)
            | (g["slope"] >= cfg["slope_limit"]))


def nearest_index(g, lat, lon):
    return int(np.argmin(np.abs(g["lats"] - lat))), int(np.argmin(np.abs(g["lons"] - lon)))


def snap_to_free(cell, blocked, radius=12):
    """Closest passable cell within `radius` cells, or None."""
    r, c = cell
    if not blocked[r, c]:
        return cell
    best, best_d = None, None
    for rr in range(max(0, r - radius), min(blocked.shape[0], r + radius + 1)):
        for cc in range(max(0, c - radius), min(blocked.shape[1], c + radius + 1)):
            if not blocked[rr, cc]:
                d = (rr - r) ** 2 + (cc - c) ** 2
                if best is None or d < best_d:
                    best, best_d = (rr, cc), d
    return best


def astar(start, goal, g, cfg):
    """A* over the grid. Returns a list of (row, col), or [] when no route exists."""
    rows, cols = g["terrain"].shape
    blocked = blocked_mask(g, cfg)
    for r, c in (start, goal):
        if not (0 <= r < rows and 0 <= c < cols) or blocked[r, c]:
            return []
    open_set = [(0.0, 0.0, start)]
    came_from, g_score, closed = {}, {start: 0.0}, set()
    while open_set:
        _, _, cur = heapq.heappop(open_set)
        if cur in closed:
            continue
        if cur == goal:
            path = [goal]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            return path[::-1]
        closed.add(cur)
        for dr, dc in NEIGHBORS:
            nr, nc = cur[0] + dr, cur[1] + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or blocked[nr, nc] or (nr, nc) in closed:
                continue
            if dr and dc and (blocked[cur[0] + dr, cur[1]] or blocked[cur[0], cur[1] + dc]):
                continue                                    # no diagonal corner cutting
            nb = (nr, nc)
            tentative = g_score[cur] + terrain_cost(cur, nb, g, cfg)
            if tentative < g_score.get(nb, float("inf")):
                came_from[nb] = cur
                g_score[nb] = tentative
                heapq.heappush(open_set, (tentative + cell_distance_m(nb, goal, g), tentative, nb))
    return []


def straight_line(start, goal):
    n = max(abs(goal[0] - start[0]), abs(goal[1] - start[1])) + 1
    rr = np.rint(np.linspace(start[0], goal[0], n)).astype(int)
    cc = np.rint(np.linspace(start[1], goal[1], n)).astype(int)
    return [(int(r), int(c)) for r, c in zip(rr, cc)]


def track(g, cells, cfg=None, blocked=None):
    """Everything we need to know about a path, as plain lists (cheap to keep in session state)."""
    elev = [float(g["terrain"][p]) for p in cells]
    slope = [float(g["slope"][p]) for p in cells]
    seg = [cell_distance_m(p, q, g) for p, q in zip(cells[:-1], cells[1:])]
    dz = np.diff(elev)
    cost = None
    if cfg is not None:
        cost = sum(terrain_cost(p, q, g, cfg) for p, q in zip(cells[:-1], cells[1:]))
    return {
        "coords": [(float(g["lats"][r]), float(g["lons"][c])) for r, c in cells],
        "elev": elev, "slope": slope, "haz": [int(g["hazard"][p]) for p in cells],
        "seg_m": seg,
        "dist_km": [0.0] + list(np.cumsum(seg) / 1000.0),
        "distance_km": sum(seg) / 1000.0,
        "mean_slope": float(np.nanmean(slope)) if np.isfinite(slope).any() else float("nan"),
        "max_slope": float(np.nanmax(slope)) if np.isfinite(slope).any() else float("nan"),
        "ascent": float(np.nansum(np.clip(dz, 0, None))),
        "descent": float(np.nansum(np.clip(-dz, 0, None))),
        "cost": cost,
        "blocked_cells": int(sum(bool(blocked[p]) for p in cells)) if blocked is not None else 0,
    }


def compute_route(a, area_key, s_lat, s_lon, g_lat, g_lon, cfg):
    """Run the planner and store the result. Returns an error message, or None on success."""
    g = sub(a, AREAS[area_key]["stride"])
    blocked = blocked_mask(g, cfg)
    start = snap_to_free(nearest_index(g, s_lat, s_lon), blocked)
    goal = snap_to_free(nearest_index(g, g_lat, g_lon), blocked)
    if start is None or goal is None:
        return "The start or destination sits in blocked terrain. Move the point or raise the slope limit."
    if start == goal:
        return "Start and destination map to the same terrain cell."
    with st.spinner("Searching for a route..."):
        path = astar(start, goal, g, cfg)
    if not path:
        st.session_state.pop("route", None)
        return "No route found with these settings. Try other points or a higher slope limit."
    st.session_state["route"] = {
        "area": area_key, "cfg": cfg,
        "main": track(g, path, cfg),
        "base": track(g, straight_line(start, goal), None, blocked),
    }
    return None


def route_map(a, route):
    d = disp(a)
    main, base = route["main"], route["base"]
    fig = go.Figure(go.Heatmap(x=d["lons"], y=d["lats"], z=d["slope"], colorscale="Turbo", zmin=0, zmax=40,
                               opacity=0.85, colorbar=dict(title="Slope (°)")))
    fig.add_trace(go.Scatter(x=[c[1] for c in base["coords"]], y=[c[0] for c in base["coords"]], mode="lines",
                             line=dict(color="yellow", width=2, dash="dash"), name="Straight line"))
    fig.add_trace(go.Scatter(x=[c[1] for c in main["coords"]], y=[c[0] for c in main["coords"]], mode="lines",
                             line=dict(color="white", width=4), name="A* route"))
    for coord, color, symbol, label in ((main["coords"][0], "#2ECC71", "circle", "Start"),
                                        (main["coords"][-1], "#00E5FF", "star", "Destination")):
        fig.add_trace(go.Scatter(x=[coord[1]], y=[coord[0]], mode="markers", name=label,
                                 marker=dict(size=14, color=color, symbol=symbol,
                                             line=dict(color="black", width=1))))
    fig.update_layout(title="Planned route over terrain slope", xaxis_title="Longitude (°E)",
                      yaxis_title="Latitude (°)", height=650, **PLOT_LAYOUT)
    return fig


def profile_figure(main):
    fig = go.Figure(go.Scatter(x=main["dist_km"], y=main["elev"], mode="lines", fill="tozeroy",
                               line=dict(color="#F08A4B", width=3)))
    fig.update_layout(title="Elevation along the route", xaxis_title="Distance (km)",
                      yaxis_title="Elevation (m)", **PLOT_LAYOUT)
    return fig


def route_csv(main):
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["Sequence", "Latitude", "Longitude", "Elevation_m", "Slope_degrees", "Hazard_class"])
    for i, ((lat, lon), z, s, h) in enumerate(zip(main["coords"], main["elev"], main["slope"], main["haz"]), 1):
        writer.writerow([i, f"{lat:.5f}", f"{lon:.5f}", f"{z:.1f}", f"{s:.2f}", HAZARD_SHORT.get(h, "Unknown")])
    return out.getvalue()


def route_geojson(main):
    feature = {
        "type": "Feature",
        "properties": {"name": "MARS EXPLORER A* candidate route", "study_site": "Gale Crater", "algorithm": "A*",
                       "note": "Preliminary terrain-cost route; not a certified safe rover route."},
        "geometry": {"type": "LineString", "coordinates": [[lon, lat] for lat, lon in main["coords"]]},
    }
    return json.dumps({"type": "FeatureCollection", "features": [feature]}, indent=2)


EVA_SPEED_FACTOR = {0: 1.0, 1: 0.7, 2: 0.4, 3: 0.15}


def eva_hours(main, speed_kmh):
    seconds = 0.0
    for dist, haz in zip(main["seg_m"], main["haz"][1:]):
        seconds += dist / (speed_kmh / 3.6 * EVA_SPEED_FACTOR.get(haz, 0.15))
    return seconds / 3600.0


# =========================================================
# SIDEBAR
# =========================================================
PAGES = ["Start Here", "Terrain Maps", "3D Terrain", "Route Planner", "Safety & EVA",
         "Image Gallery", "Data & Method"]

with st.sidebar:
    st.markdown("## 🔴 MARS EXPLORER")
    st.caption("NASA HACKATHON 2026 | GALE CRATER")
    st.divider()
    page = st.radio("Go to", PAGES, index=0)
    st.divider()
    area_key = st.selectbox("Study area", list(AREAS), help="Close-up is detailed and quick. "
                            "Wide covers the whole Gale region from the original preview scripts.")
    st.caption(f"Crater centre: {GALE_LAT}°, {GALE_LON}°E")

if st.session_state.get("route", {}).get("area") not in (None, area_key):
    st.session_state.pop("route", None)         # a route from another area would not fit this map


# =========================================================
# PAGES
# =========================================================
def page_start():
    a = require_analysis(area_key)
    st.markdown(
        '<div class="hero"><h1>MARS EXPLORER</h1>'
        "<p>Where can a rover or an astronaut safely go in Gale Crater? "
        "This dashboard turns NASA elevation data into slope maps, hazard maps and a planned route.</p>"
        '<span class="badge">NASA MOLA data</span><span class="badge">Slope and hazard maps</span>'
        '<span class="badge">3D terrain</span><span class="badge">A* route planner</span>'
        '<span class="badge">EVA time estimate</span></div>', unsafe_allow_html=True)

    stats = terrain_statistics(a["terrain"])
    share = hazard_share(a["hazard"])
    kpi_row([
        ("Study area", f"{a['terrain'].shape[1]} x {a['terrain'].shape[0]}", f"pixels of about {a['dx']:.0f} m"),
        ("Height range", f"{stats['minimum']:.0f} to {stats['maximum']:.0f} m", f"{stats['maximum'] - stats['minimum']:.0f} m of relief"),
        ("Average slope", f"{np.nanmean(a['slope']):.1f}°", f"steepest {np.nanmax(a['slope']):.1f}°"),
        ("Easy ground", f"{share[0] + share[1]:.0f}%", "slope under 15°"),
    ])

    st.write("")
    title("How to use this dashboard")
    steps = [("1. Look", "Terrain Maps shows height, slope and hazard classes."),
             ("2. Explore", "3D Terrain lets you rotate the crater and Mount Sharp."),
             ("3. Plan", "Route Planner finds a low-risk path between two points."),
             ("4. Check", "Safety & EVA estimates walking time and runs go or no-go checks.")]
    for col, (head, text) in zip(st.columns(4), steps):
        col.markdown(f'<div class="step"><b>{head}</b><br><span>{text}</span></div>', unsafe_allow_html=True)

    st.write("")
    title("Maps at a glance")
    top = st.columns(2)
    bottom = st.columns(2)
    with top[0]:
        show_plot(elevation_map(a, compact=True))
    with top[1]:
        show_plot(slope_map(a, compact=True))
    with bottom[0]:
        show_plot(hazard_map(a, compact=True))
    with bottom[1]:
        with st.container(border=True):
            st.markdown("**Reading the colours**")
            legend([(HAZARD_COLORS[i], HAZARD_LABELS[i]) for i in range(4)])
            st.caption("Hazard is based on slope only. A steeper slope means a harder and riskier drive or walk.")
            st.markdown("**Want a route?** Open Route Planner and press Quick demo.")

    images = get_images(IMAGERY_DIR)[:4]
    if images:
        st.write("")
        title("Surface imagery")
        image_grid(images, columns=4, key="start")

    with st.expander("Plain-language glossary"):
        st.markdown(
            "- **MOLA**: the laser altimeter on a NASA orbiter that measured Mars heights.\n"
            "- **Slope**: how steep the ground is, in degrees. Flat is 0°, a 45° hill is very steep.\n"
            "- **Hazard class**: slope grouped into low, moderate, high and very high.\n"
            "- **A\\***: a search method that finds the cheapest path across a grid.\n"
            "- **EVA**: extravehicular activity, meaning a walk outside the habitat.")


def page_maps():
    a = require_analysis(area_key)
    title("Terrain Maps", "Three views of the same ground. Hover over any map for exact values.")
    tab1, tab2, tab3 = st.tabs(["Elevation", "Slope", "Hazard classes"])
    with tab1:
        note("Brighter colours are higher ground. The white dot marks Gale Crater's centre.")
        fig = elevation_map(a)
        show_plot(fig)
        html_download(fig, "Download interactive map (HTML)", "gale_elevation.html", "dl_elev")
    with tab2:
        note("Slope is how steeply the ground tilts. Blue and green are gentle, orange and red are steep.")
        fig = slope_map(a)
        show_plot(fig)
        s = a["slope"][np.isfinite(a["slope"])]
        kpi_row([("Minimum", f"{s.min():.1f}°", ""), ("Average", f"{s.mean():.1f}°", ""), ("Maximum", f"{s.max():.1f}°", "")])
        html_download(fig, "Download interactive map (HTML)", "gale_slope.html", "dl_slope")
    with tab3:
        note("Each cell gets one of four classes from its slope. Unknown cells stay blank and are never used for routing.")
        legend([(HAZARD_COLORS[i], HAZARD_LABELS[i]) for i in range(4)])
        fig = hazard_map(a)
        show_plot(fig)
        st.table(hazard_table(a["hazard"]))
        html_download(fig, "Download interactive map (HTML)", "gale_hazard.html", "dl_haz")


def page_3d():
    a = require_analysis(area_key)
    title("3D Terrain", "Drag to rotate, scroll to zoom, double-click to reset the view.")
    exaggeration = st.slider("Stretch heights by", 1.0, 20.0, 5.0, 0.5,
                             help="Mars relief is small next to its width, so heights are stretched to be visible.")
    fig = terrain_3d(a, exaggeration)
    show_plot(fig)
    html_download(fig, "Download interactive 3D view (HTML)", "gale_3d_terrain.html", "dl_3d")


def page_route():
    a = require_analysis(area_key)
    title("Route Planner", "Pick two points and the planner finds the cheapest path that avoids steep ground.")

    lat_lo, lat_hi = float(a["lats"].min()), float(a["lats"].max())
    lon_lo, lon_hi = float(a["lons"].min()), float(a["lons"].max())
    available = {name: v for name, v in PRESETS.items()
                 if lat_lo <= min(v[0], v[2]) and max(v[0], v[2]) <= lat_hi
                 and lon_lo <= min(v[1], v[3]) and max(v[1], v[3]) <= lon_hi}
    options = list(available) + ["Custom points"]
    preset = st.selectbox("Mission preset", options)
    default = available.get(preset) or (a["center_lat"] + 0.4, a["center_lon"] - 0.3,
                                        a["center_lat"] - 0.4, a["center_lon"] + 0.3)
    tag = f"{area_key}_{preset}"

    c1, c2, c3, c4 = st.columns(4)
    s_lat = c1.number_input("Start latitude", lat_lo, lat_hi, float(np.clip(default[0], lat_lo, lat_hi)),
                            0.01, format="%.3f", key=f"sla_{tag}")
    s_lon = c2.number_input("Start longitude", lon_lo, lon_hi, float(np.clip(default[1], lon_lo, lon_hi)),
                            0.01, format="%.3f", key=f"slo_{tag}")
    g_lat = c3.number_input("Destination latitude", lat_lo, lat_hi, float(np.clip(default[2], lat_lo, lat_hi)),
                            0.01, format="%.3f", key=f"gla_{tag}")
    g_lon = c4.number_input("Destination longitude", lon_lo, lon_hi, float(np.clip(default[3], lon_lo, lon_hi)),
                            0.01, format="%.3f", key=f"glo_{tag}")

    with st.expander("Advanced: how the planner weighs terrain"):
        e1, e2, e3, e4 = st.columns(4)
        cfg = dict(DEFAULT_CFG)
        cfg["slope_limit"] = e1.slider("Steepest slope allowed (°)", 10.0, 40.0, 30.0, 1.0)
        cfg["slope_weight"] = e2.slider("Slope penalty", 0.0, 10.0, 3.0, 0.5)
        cfg["elevation_weight"] = e3.slider("Height-change penalty", 0.0, 0.5, 0.15, 0.01)
        cfg["hazard_weight"] = e4.slider("Hazard penalty", 0.0, 3.0, 1.0, 0.25)
        st.caption("Higher penalties push the route away from steep or bumpy ground, even if it gets longer.")

    b1, b2 = st.columns(2)
    error = None
    if b1.button("🚀 Calculate route", type="primary"):
        error = compute_route(a, area_key, s_lat, s_lon, g_lat, g_lon, cfg)
    if b2.button("Quick demo"):
        demo = next(iter(available.values()), default)
        error = compute_route(a, area_key, *demo, dict(DEFAULT_CFG))
    if error:
        st.error(error)

    route = st.session_state.get("route")
    if not route:
        note("Choose a preset, then press Calculate route. Quick demo runs the first preset with default settings.")
        return

    main, base = route["main"], route["base"]
    show_plot(route_map(a, route))
    legend([("white", "Planned route"), ("yellow", "Straight line, ignores slope"),
            ("#2ECC71", "Start"), ("#00E5FF", "Destination")])

    kpi_row([
        ("Distance", f"{main['distance_km']:.2f} km", f"{len(main['coords'])} route points"),
        ("Average slope", f"{main['mean_slope']:.1f}°", f"steepest {main['max_slope']:.1f}°"),
        ("Climbing", f"{main['ascent']:.0f} m", f"descending {main['descent']:.0f} m"),
        ("Effort score", f"{main['cost'] / 1000:.1f}", "km-equivalent"),
    ])
    st.write("")
    note(f"A straight line would be {base['distance_km']:.2f} km with a steepest slope of {base['max_slope']:.1f}° "
         f"and would cross {base['blocked_cells']} cells that are too steep or unknown. "
         f"The planned route is {main['distance_km']:.2f} km and crosses none.")
    show_plot(profile_figure(main))

    d1, d2 = st.columns(2)
    d1.download_button("Download route CSV", route_csv(main), "mars_candidate_route.csv", "text/csv")
    d2.download_button("Download route GeoJSON", route_geojson(main), "mars_candidate_route.geojson",
                       "application/geo+json")
    st.caption("Preliminary result from 128 px/deg MOLA data. Not a certified safe rover route.")


def page_safety():
    title("Safety & EVA", "How long the walk takes and whether the route passes basic checks.")
    route = st.session_state.get("route")
    if not route:
        note("No route yet. Open Route Planner and press Quick demo first.")
        return

    main, cfg = route["main"], route["cfg"]
    c1, c2 = st.columns(2)
    speed = c1.slider("Walking speed on flat ground (km/h)", 1.0, 6.0, 3.0, 0.5)
    budget = c2.slider("EVA time budget (hours)", 1.0, 12.0, 6.0, 0.5)
    hours = eva_hours(main, speed)

    kpi_row([
        ("One-way time", f"{hours:.1f} h", f"{hours / budget * 100:.0f}% of budget"),
        ("Round trip", f"{2 * hours:.1f} h", "same route back"),
        ("Distance", f"{main['distance_km']:.1f} km", "one way"),
    ])
    st.write("")
    high = sum(1 for h in main["haz"] if h >= 2)
    checks = [
        ("Steepest slope is under the planner limit", main["max_slope"] < cfg["slope_limit"],
         f"{main['max_slope']:.1f}° against {cfg['slope_limit']:.0f}°"),
        ("No high-hazard cells on the route", high == 0, f"{high} of {len(main['haz'])} cells"),
        ("One-way trip fits the time budget", hours <= budget, f"{hours:.1f} h of {budget:.1f} h"),
        ("Round trip fits the time budget", 2 * hours <= budget, f"{2 * hours:.1f} h of {budget:.1f} h"),
        ("Total climb is under 500 m", main["ascent"] < 500, f"{main['ascent']:.0f} m"),
    ]
    passed = sum(ok for _, ok, _ in checks)
    st.markdown(f"### {'🟢 GO' if passed == len(checks) else '🔴 REVIEW'}: {passed} of {len(checks)} checks passed")
    for label, ok, detail in checks:
        st.markdown(f"{'🟢' if ok else '🔴'} **{label}**: {detail}")

    fig = go.Figure(go.Scatter(x=main["dist_km"], y=main["slope"], mode="lines",
                               line=dict(color="#F0A45D", width=3)))
    fig.add_hline(y=cfg["slope_limit"], line_dash="dash", line_color="#e74c3c")
    fig.update_layout(title="Slope along the route (dashed line is the limit)", xaxis_title="Distance (km)",
                      yaxis_title="Slope (°)", **PLOT_LAYOUT)
    show_plot(fig)
    note("Walking speed is multiplied by 1.0 on low hazard ground, 0.7 on moderate, 0.4 on high and 0.15 on very "
         "high. The 500 m climb check is also an assumption. None of these figures are mission validated.")


def page_gallery():
    title("Image Gallery", "Every image sits in an equal-size tile. Press Enlarge to open one full size.")
    tab1, tab2, tab3 = st.tabs(["Surface imagery", "Geology", "Thermal"])
    with tab1:
        gallery(IMAGERY_DIR, "imagery", compare=True)
    with tab2:
        gallery(GEOLOGY_DIR, "geology")
    with tab3:
        gallery(THERMAL_DIR, "thermal")


def page_data():
    title("Data & Method", "What data is used, how it is processed and where it falls short.")
    st.markdown("**Files found**")
    for label, path in [("MOLA elevation label", MOLA_XML), ("Surface imagery", IMAGERY_DIR),
                        ("Geology images", GEOLOGY_DIR), ("Thermal images", THERMAL_DIR)]:
        ok = path.exists()
        extra = f" ({len(get_images(path))} images)" if ok and path.is_dir() else ""
        st.markdown(f"{'🟢' if ok else '🔴'} **{label}**: `{path.relative_to(BASE_DIR)}`{extra}")

    a = None
    try:
        a = load_analysis(area_key)
    except Exception:
        pass
    if a is not None:
        st.write("")
        st.markdown("**Current study area**")
        valid = a["terrain"][np.isfinite(a["terrain"])]
        st.table([
            {"Item": "Grid size", "Value": f"{a['terrain'].shape[1]} x {a['terrain'].shape[0]} pixels"},
            {"Item": "Latitude range", "Value": f"{a['lats'].min():.2f}° to {a['lats'].max():.2f}°"},
            {"Item": "Longitude range", "Value": f"{a['lons'].min():.2f}°E to {a['lons'].max():.2f}°E"},
            {"Item": "Valid pixels", "Value": f"{valid.size:,} of {a['terrain'].size:,}"},
            {"Item": "Elevation min / mean / max", "Value": f"{valid.min():.0f} / {valid.mean():.0f} / {valid.max():.0f} m"},
            {"Item": "Pixel size", "Value": f"{a['dx']:.0f} m east-west, {a['dy']:.0f} m north-south"},
        ])

    meta = read_mola_metadata()
    if meta:
        with st.expander("MOLA label coordinates"):
            st.table(meta)

    st.markdown(
        """
**How it works.** Elevation comes from the NASA MOLA 128 pixels per degree tile `megt00n090hb`. Slope is
calculated from height differences between neighbouring pixels, using the true ground size of a pixel.
Slope is grouped into four hazard classes. The route planner searches an 8-neighbour grid with A*. Each
step costs its distance times factors for slope, height change and hazard, so a longer but gentler route
can win. Cells at or above the slope limit are blocked, unknown cells are blocked, and the route cannot cut
across diagonal corners. In the wide area the search samples every fourth pixel to stay fast.

**Limits.** One pixel covers about 460 m, so rocks, dunes and small scarps cannot be seen. Hazard here uses
slope only. Treat every result as a planning demonstration, not a certified safe route. Higher-resolution
terrain models such as HiRISE would be needed for real mission use.
""")


PAGE_FUNCTIONS = {
    "Start Here": page_start, "Terrain Maps": page_maps, "3D Terrain": page_3d, "Route Planner": page_route,
    "Safety & EVA": page_safety, "Image Gallery": page_gallery, "Data & Method": page_data,
}
PAGE_FUNCTIONS[page]()

st.markdown('<div class="footer">MARS EXPLORER | Preliminary terrain-based mission planning prototype</div>',
            unsafe_allow_html=True)