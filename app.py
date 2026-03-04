"""
Nordic Ski Analysis — Streamlit app.

Setup
─────
1. Create .streamlit/secrets.toml (see secrets.toml.example).
2. Run: streamlit run app.py
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import streamlit as st
from streamlit_folium import st_folium

from strava_auth import auth_url, exchange_code, ensure_valid_token, store_tokens, logout
from strava_fetch import list_activities, fetch_points, activity_label
from gpx_filter import GPXTrack, parse_gpx_bytes, count_trkpt
from plot_track import build_map, add_layer_control
from plot_stats import (
    _enrich, find_top_speeds, build_speed_chart, add_top_speed_markers,
    build_speed_histogram, build_terrain_breakdown,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Nordic Ski Analysis",
    page_icon="⛷",
    layout="wide",
)

# ── Secrets ───────────────────────────────────────────────────────────────────

CLIENT_ID     = st.secrets["strava"]["client_id"]
CLIENT_SECRET = st.secrets["strava"]["client_secret"]
REDIRECT_URI  = st.secrets["strava"]["redirect_uri"]

# ── Activity type groups & time period options ────────────────────────────────

_SPORT_GROUPS = {
    "🎿 Ski":   {"NordicSki", "BackcountrySki", "AlpineSki", "RollerSki"},
    "🏃 Run":   {"Run", "TrailRun", "VirtualRun"},
    "🚴 Bike":  {"Ride", "MountainBikeRide", "GravelRide", "VirtualRide",
                  "EBikeRide", "EMountainBikeRide"},
    "⛸️ Skate": {"InlineSkate"},
}

_TIME_OPTIONS = {
    "All time":     None,
    "This year":    lambda: datetime(datetime.now().year, 1, 1, tzinfo=timezone.utc),
    "Last 90 days": lambda: datetime.now(timezone.utc) - timedelta(days=90),
    "Last 30 days": lambda: datetime.now(timezone.utc) - timedelta(days=30),
    "Last 7 days":  lambda: datetime.now(timezone.utc) - timedelta(days=7),
}

# ── Cached analysis functions ─────────────────────────────────────────────────

@st.cache_data(show_spinner="Processing GPX…")
def _analyse_gpx(gpx_bytes: bytes, gpx_name: str,
                 sigma_gps: float, sigma_a_base: float, sigma_a_sensitivity: float,
                 segment_start_trim: int,
                 max_jump_factor: float,
                 pause_speed_ms: float, pause_min_sec: float) -> GPXTrack:
    points = parse_gpx_bytes(gpx_bytes)
    if not points:
        return None
    track = GPXTrack.from_points(
        points,
        sigma_gps=sigma_gps,
        sigma_a_base=sigma_a_base,
        sigma_a_sensitivity=sigma_a_sensitivity,
        segment_start_trim=segment_start_trim,
        max_jump_factor=max_jump_factor,
        pause_speed_ms=pause_speed_ms,
        pause_min_sec=pause_min_sec,
    )
    track.path = gpx_name
    return track


@st.cache_data(show_spinner="Loading activities…", ttl=300)
def _load_activities(token: str) -> list:
    return list_activities(token)


@st.cache_data(show_spinner="Fetching GPS data…", ttl=3600)
def _fetch_and_analyse(token: str, activity_id: int, _activity: dict,
                       sigma_gps: float, sigma_a_base: float, sigma_a_sensitivity: float,
                       segment_start_trim: int,
                       max_jump_factor: float,
                       pause_speed_ms: float, pause_min_sec: float) -> GPXTrack:
    points = fetch_points(token, _activity)
    if not points:
        return None
    return GPXTrack.from_points(
        points,
        sigma_gps=sigma_gps,
        sigma_a_base=sigma_a_base,
        sigma_a_sensitivity=sigma_a_sensitivity,
        segment_start_trim=segment_start_trim,
        max_jump_factor=max_jump_factor,
        pause_speed_ms=pause_speed_ms,
        pause_min_sec=pause_min_sec,
    )

# ── OAuth flow ────────────────────────────────────────────────────────────────

def handle_oauth() -> bool:
    """Handle the OAuth callback if ?code=... is in the URL.
    Returns True if the user is now authenticated with Strava."""
    params = st.query_params
    if "code" in params and "access_token" not in st.session_state:
        try:
            tokens = exchange_code(CLIENT_ID, CLIENT_SECRET, params["code"])
            store_tokens(tokens)
            st.query_params.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Strava authorisation failed: {e}")
            return False

    token = ensure_valid_token(CLIENT_ID, CLIENT_SECRET)
    return token is not None

# ── Mode detection ────────────────────────────────────────────────────────────

strava_ok = handle_oauth()
gpx_mode  = "gpx_bytes" in st.session_state

# ── Landing page (neither mode active) ───────────────────────────────────────

if not strava_ok and not gpx_mode:
    st.title("⛷ Nordic Ski Analysis")
    st.markdown("Connect your Strava account or upload a GPX file to get started.")
    st.markdown("")

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown("#### Connect with Strava")
        st.markdown("Pull activities directly from your Strava account.")
        st.link_button(
            "🔗 Connect with Strava",
            auth_url(CLIENT_ID, REDIRECT_URI),
            type="primary",
        )

    with col_r:
        st.markdown("#### Upload a GPX file")
        st.markdown("Analyse a local file from your device.")
        uploaded = st.file_uploader("GPX file", type=["gpx"], label_visibility="collapsed")
        if uploaded is not None:
            st.session_state.gpx_bytes = uploaded.read()
            st.session_state.gpx_name  = uploaded.name
            st.rerun()

    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    if strava_ok:
        athlete = st.session_state.get("athlete", {})
        st.markdown(f"### {athlete.get('firstname', '')} {athlete.get('lastname', '')}")
        if athlete.get("profile"):
            st.markdown(
                f'<img src="{athlete["profile"]}" width="64" style="border-radius:50%">',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(f"**File:** {st.session_state.gpx_name}")
        new_file = st.file_uploader("Upload a different file", type=["gpx"])
        if new_file is not None:
            st.session_state.gpx_bytes = new_file.read()
            st.session_state.gpx_name  = new_file.name
            st.rerun()

    st.divider()
    st.markdown("**Kalman smoother**")
    sigma_gps    = st.slider("GPS noise σ (m)", 1.0, 20.0, 10.0, 0.5)
    sigma_a_base = st.slider("Base acceleration σ (m/s²)", 0.1, 3.0, 0.8, 0.1)
    sigma_a_sens = st.slider("Terrain sensitivity", 0.0, 5.0, 1.2, 0.5)
    st.caption("Sensitivity scales how much steep gradients inflate the process noise "
               "(downhill: full boost; uphill: 40%).")

    st.divider()
    st.markdown("**Segment start**")
    start_trim = st.slider("Trim points at segment start", 0, 120, 60)

    st.divider()
    st.markdown("**Spike filter**")
    max_jump_factor = st.slider("Max jump factor", 1.5, 6.0, 3.0, 0.5)
    st.caption("A GPS step is flagged if its implied speed exceeds this multiple "
               "of the local Kalman speed. At 20 km/h the threshold is 3× = 60 km/h; "
               "at 60 km/h it is 3× = 180 km/h.")

    st.divider()
    st.markdown("**Pause detection**")
    pause_speed = st.slider("Pause threshold (km/h)", 0.5, 5.0, 2.9, 0.1)
    pause_min   = st.slider("Min pause duration (s)", 5, 60, 8)

    if strava_ok:
        st.divider()
        if st.button("Disconnect Strava"):
            logout()
            st.rerun()

# ── Load & analyse ────────────────────────────────────────────────────────────

st.title("⛷ Nordic Ski Analysis")

if gpx_mode:
    track = _analyse_gpx(
        st.session_state.gpx_bytes, st.session_state.gpx_name,
        sigma_gps, sigma_a_base, sigma_a_sens,
        segment_start_trim=start_trim,
        max_jump_factor=max_jump_factor,
        pause_speed_ms=pause_speed / 3.6,
        pause_min_sec=float(pause_min),
    )
else:
    token      = ensure_valid_token(CLIENT_ID, CLIENT_SECRET)
    activities = _load_activities(token)

    if not activities:
        st.info("No activities found on your Strava account.")
        st.stop()

    # ── Activity filters ──────────────────────────────────────────────────────
    col_type, col_time = st.columns([3, 2])
    with col_type:
        selected_groups = st.multiselect(
            "Activity type",
            list(_SPORT_GROUPS.keys()),
            default=list(_SPORT_GROUPS.keys()),
        )
    with col_time:
        time_label = st.selectbox("Time period", list(_TIME_OPTIONS.keys()))

    allowed_types: set = set()
    for g in selected_groups:
        allowed_types |= _SPORT_GROUPS[g]

    since_fn = _TIME_OPTIONS[time_label]
    since_dt = since_fn() if since_fn else None

    filtered = [
        a for a in activities
        if (a.get("sport_type") or a.get("type", "")) in allowed_types
        and (
            since_dt is None
            or datetime.strptime(a["start_date"], "%Y-%m-%dT%H:%M:%SZ")
               .replace(tzinfo=timezone.utc) >= since_dt
        )
    ]

    if not filtered:
        st.info("No activities match the current filters.")
        st.stop()

    options      = {activity_label(a): a for a in filtered}
    chosen_label = st.selectbox("Select activity", list(options.keys()))
    chosen       = options[chosen_label]

    track = _fetch_and_analyse(
        token, chosen["id"], chosen,
        sigma_gps, sigma_a_base, sigma_a_sens,
        segment_start_trim=start_trim,
        max_jump_factor=max_jump_factor,
        pause_speed_ms=pause_speed / 3.6,
        pause_min_sec=float(pause_min),
    )

if track is None or not track.segments:
    if gpx_mode:
        raw_pts = parse_gpx_bytes(st.session_state.gpx_bytes)
        if len(raw_pts) == 0:
            total_trkpt = count_trkpt(st.session_state.gpx_bytes)
            if total_trkpt == 0:
                st.error("No GPS track points found in this file. Make sure it is a GPX track file (not a waypoint or route file).")
            else:
                st.error(
                    f"Found {total_trkpt} GPS points but none have timestamps. "
                    "Speed analysis requires time data. Try exporting the activity "
                    "from your device or app with timestamps included."
                )
        else:
            st.warning(
                f"Parsed {len(raw_pts)} GPS points but all were removed during filtering. "
                f"Try reducing **Trim points at segment start** (currently {start_trim}) in the sidebar."
            )
    else:
        st.warning("This activity has no GPS data.")
    st.stop()

# ── Summary metrics ───────────────────────────────────────────────────────────

sm = track.summary
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Distance",         f"{sm['distance_km']:.1f} km")
col2.metric("Moving time",      f"{sm['duration_min']:.0f} min")
col3.metric("Elevation ↑",      f"{sm['ele_gain_m']:.0f} m")
col4.metric("Elevation ↓",      f"{sm['ele_loss_m']:.0f} m")
col5.metric("Segments",         sm['segments'])
col6.metric("Outliers removed", track.n_gated)

st.divider()

# ── Speed chart ───────────────────────────────────────────────────────────────

df  = _enrich(track.to_dataframe())
top = find_top_speeds(df)

fig = build_speed_chart(track, df, top)
st.plotly_chart(fig, use_container_width=True)

# ── Top 10 table ──────────────────────────────────────────────────────────────

with st.expander("Top 10 fastest locations", expanded=False):
    top_display = top[["rank", "speed_kmh", "time", "cum_dist_m", "ele"]].copy()
    top_display["speed_kmh"]   = top_display["speed_kmh"].round(1)
    top_display["cum_dist_m"]  = (top_display["cum_dist_m"] / 1000).round(2)
    top_display["ele"]         = top_display["ele"].round(0)
    top_display.columns        = ["Rank", "Speed (km/h)", "Time", "Distance (km)", "Elevation (m)"]
    st.dataframe(top_display, hide_index=True, use_container_width=True)

st.divider()

# ── Analysis charts ───────────────────────────────────────────────────────────

st.subheader("Analysis")
col_hist, col_terrain = st.columns(2)
with col_hist:
    st.plotly_chart(build_speed_histogram(df), use_container_width=True)
with col_terrain:
    st.plotly_chart(build_terrain_breakdown(df), use_container_width=True)

st.divider()

# ── Map ───────────────────────────────────────────────────────────────────────

st.subheader("Track map")
fmap = build_map(track)
add_top_speed_markers(fmap, top)
add_layer_control(fmap, collapsed=False)
st_folium(fmap, use_container_width=True, height=550, returned_objects=[])

# ── Segment stats table ───────────────────────────────────────────────────────

with st.expander("Segment statistics", expanded=False):
    sdf = track.stats_dataframe()
    sdf["segment"]      = sdf["segment"] + 1
    sdf["distance_km"]  = sdf["distance_km"].round(2)
    sdf["duration_min"] = sdf["duration_min"].round(1)
    sdf["avg_kmh"]      = sdf["avg_kmh"].round(1)
    sdf["max_kmh"]      = sdf["max_kmh"].round(1)
    sdf["ele_gain_m"]   = sdf["ele_gain_m"].round(0)
    sdf["ele_loss_m"]   = sdf["ele_loss_m"].round(0)
    sdf.columns = ["Segment", "Start", "End", "Points",
                   "Distance (km)", "Duration (min)", "Avg (km/h)", "Max (km/h)",
                   "Gain (m)", "Loss (m)"]
    st.dataframe(sdf, hide_index=True, use_container_width=True)
