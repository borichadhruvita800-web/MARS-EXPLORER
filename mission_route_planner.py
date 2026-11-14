import numpy as np
import plotly.graph_objects as go
import streamlit as st
from pds4_tools import pds4_read
from scipy.ndimage import distance_transform_edt

from modules.hazard_analysis import classify_hazard, hazard_stats
from modules.route_planning import astar, path_metrics, straight_line
from modules.terrain_analysis import extract_region, pixel_size_m

MOLA_XML = "data/elevation/mola/megt00n090hb.xml"

st.set_page_config(page_title="Mars Mission Planner", page_icon="🚀", layout="wide")
st.title("🚀 MARS EXPLORER")
st.subheader("Interactive Gale Crater Mission Planner")
st.caption("MOLA elevation data | A* route planning | Terrain hazard analysis")


@st.cache_data(show_spinner="Loading MOLA terrain...")
def load_terrain():
    product = pds4_read(MOLA_XML, quiet=True)
    region = extract_region(np.asarray(product[0].data))
    terrain = region["terrain"]
    valid = np.isfinite(terrain)

    idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = terrain[tuple(idx)]

    dx, dy = pixel_size_m(float(np.mean(region["lats"])))
    dz_dy, dz_dx = np.gradient(filled, dy, dx)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    slope[~valid] = np.nan
    return region, terrain, slope, classify_hazard(slope), dx, dy


try:
    region, terrain, slope, hazard, dx, dy = load_terrain()
except Exception as error:
    st.error(f"Terrain loading failed: {error}")
    st.stop()

lats, lons = region["lats"], region["lons"]

# ---------------- sidebar ----------------
sb = st.sidebar
sb.header("Mission Configuration")
sb.subheader("Starting Location")
start_lat = sb.number_input("Start latitude", -17.3, -1.5, -3.5, 0.1)
start_lon = sb.number_input("Start longitude", 129.9, 145.7, 132.0, 0.1)
sb.subheader("Destination")
goal_lat = sb.number_input("Destination latitude", -17.3, -1.5, -8.0, 0.1)
goal_lon = sb.number_input("Destination longitude", 129.9, 145.7, 140.0, 0.1)
sb.divider()
sb.subheader("Route Optimization")
lam = sb.slider("Slope penalty (λ)", 0.0, 10.0, 3.0, 0.5)
mu = sb.slider("Hazard penalty (μ)", 0.0, 10.0, 2.0, 0.5)
max_slope = sb.slider("Maximum permitted slope (degrees)", 5.0, 45.0, 30.0, 1.0)
stride = sb.select_slider(
    "Planning resolution",
    options=[2, 4, 8],
    value=4,
    format_func=lambda s: f"{s} px step (~{s * dx / 1000:.1f} km)",
)

# ---------------- terrain overview ----------------
with st.expander("Terrain hazard summary", expanded=False):
    cols = st.columns(4)
    for col, (label, pct) in zip(cols, hazard_stats(hazard).items()):
        col.metric(label, f"{pct:.1f}%")


def to_pixel(lat, lon):
    return int(np.argmin(np.abs(lats - lat))), int(np.argmin(np.abs(lons - lon)))


def nearest_free(blocked, cell, radius=6):
    """Move a cell to the closest unblocked one so a start on a crater wall still works."""
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


# ---------------- route ----------------
if st.button("🚀 Calculate Candidate Route", type="primary"):
    # Plan on a coarser grid, then map the path back to full resolution
    sl, hz, tr = slope[::stride, ::stride], hazard[::stride, ::stride], terrain[::stride, ::stride]
    cdx, cdy = dx * stride, dy * stride
    blocked = ~np.isfinite(sl) | (sl >= max_slope)

    s_full, g_full = to_pixel(start_lat, start_lon), to_pixel(goal_lat, goal_lon)
    s_c = nearest_free(blocked, (s_full[0] // stride, s_full[1] // stride))
    g_c = nearest_free(blocked, (g_full[0] // stride, g_full[1] // stride))

    if s_c is None or g_c is None:
        st.error("Start or destination sits in blocked terrain. Pick another point or raise the slope limit.")
        st.stop()
    if s_c == g_c:
        st.error("Start and destination map to the same terrain cell.")
        st.stop()

    try:
        with st.spinner("Running A* route planner..."):
            path = astar(sl, hz, s_c, g_c, cdx, cdy, lam=lam, mu=mu, max_slope=max_slope)

        if path is None:
            st.error("No candidate route found. Try different coordinates or a higher slope limit.")
            st.stop()

        metrics = path_metrics(path, tr, sl, hz, cdx, cdy, lam=lam, mu=mu, max_slope=max_slope)
        baseline = straight_line(s_c, g_c)

        def geo(cells):
            return ([lats[min(r * stride, len(lats) - 1)] for r, _ in cells],
                    [lons[min(c * stride, len(lons) - 1)] for _, c in cells])

        route_lats, route_lons = geo(path)
        base_lats, base_lons = geo(baseline)
        blocked_on_baseline = sum(bool(blocked[p]) for p in baseline)

        # ---- map (row 0 is north, so no axis reversal needed) ----
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            x=lons[::4], y=lats[::4], z=slope[::4, ::4], colorscale="Turbo",
            zmin=0, zmax=40, colorbar=dict(title="Slope (°)"), connectgaps=False,
            name="Terrain slope"))
        fig.add_trace(go.Scatter(x=base_lons, y=base_lats, mode="lines",
                                 line=dict(color="yellow", width=2, dash="dash"),
                                 name="Straight-line baseline"))
        fig.add_trace(go.Scatter(x=route_lons, y=route_lats, mode="lines",
                                 line=dict(color="white", width=4), name="A* candidate route"))
        fig.add_trace(go.Scatter(
            x=[route_lons[0], route_lons[-1]], y=[route_lats[0], route_lats[-1]],
            mode="markers+text", marker=dict(size=13, color=["lime", "cyan"]),
            text=["START", "DESTINATION"], textposition="top center", name="Mission points"))
        fig.update_layout(title="Gale Crater | Candidate Mission Route", template="plotly_dark",
                          height=700, xaxis_title="Longitude (°E)", yaxis_title="Latitude (°)")
        st.plotly_chart(fig, use_container_width=True)

        # ---- metrics ----
        st.subheader("Mission Route Analytics")
        a, b, c = st.columns(3)
        a.metric("Route distance", f"{metrics['Distance (km)']:.2f} km")
        b.metric("Mean slope", f"{metrics['Mean slope (°)']:.2f}°")
        c.metric("Maximum slope", f"{metrics['Max slope (°)']:.2f}°")
        d, e, f = st.columns(3)
        d.metric("Total ascent", f"{metrics['Total ascent (m)']:.1f} m")
        e.metric("Traversal cost", f"{metrics['Traversal cost (km-equiv.)']:.2f}")
        f.metric("Route points", len(path))

        base_m = path_metrics(baseline, tr, sl, hz, cdx, cdy, lam=lam, mu=mu, max_slope=max_slope)
        st.caption(
            f"Straight line: {base_m['Distance (km)']:.1f} km, cost {base_m['Traversal cost (km-equiv.)']:.1f}, "
            f"{blocked_on_baseline} blocked cells crossed. "
            f"A*: {metrics['Distance (km)']:.1f} km, cost {metrics['Traversal cost (km-equiv.)']:.1f}."
        )

        # ---- elevation profile ----
        elev = [tr[p] for p in path]
        prof_d = [0.0]
        for p0, p1 in zip(path[:-1], path[1:]):
            prof_d.append(prof_d[-1] + np.hypot((p1[0] - p0[0]) * cdy, (p1[1] - p0[1]) * cdx) / 1000)
        prof = go.Figure(go.Scatter(x=prof_d, y=elev, mode="lines", fill="tozeroy",
                                    line=dict(color="#F08A4B", width=3), name="Elevation"))
        prof.update_layout(title="Route Elevation Profile", template="plotly_dark",
                           xaxis_title="Distance along route (km)", yaxis_title="Elevation (m)", height=400)
        st.plotly_chart(prof, use_container_width=True)

        # ---- export ----
        csv = "Latitude,Longitude,Elevation_m\n" + "\n".join(
            f"{la:.5f},{lo:.5f},{el:.1f}" for la, lo, el in zip(route_lats, route_lons, elev))
        st.download_button("Download candidate route CSV", data=csv,
                           file_name="mars_candidate_route.csv", mime="text/csv")

        st.success("Candidate route calculated.")
        st.info("The straight-line baseline is geometric only and may cross blocked terrain. "
                "The A* result is a preliminary candidate based on 128 px/deg MOLA data, "
                "not a certified safe route.")
    except Exception as error:
        st.error(f"Route calculation failed: {error}")

st.divider()
st.caption("MARS EXPLORER | Preliminary terrain-based mission planning prototype")