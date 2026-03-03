# Nordic Ski GPX Analysis — Claude Code Context

## What this project is

A Python + Streamlit app that pulls nordic ski activities from Strava,
cleans the GPS data with a Kalman smoother, and produces an interactive
map and speed profile. The owner is a statistician, so explanations and
code should be at that level.

## How to run

```bash
streamlit run app.py          # local dev (localhost:8501)
python gpx_filter.py *.gpx    # CLI: process local GPX files
python plot_stats.py *.gpx    # CLI: generate map + speed chart HTML files
```

Strava credentials live in `.streamlit/secrets.toml` (gitignored).
Copy `.streamlit/secrets.toml.example` to get started.

## File map

| File | Purpose |
|---|---|
| `gpx_filter.py` | Core library — `Point` dataclass, Kalman smoother, `GPXTrack` class |
| `plot_track.py` | Builds folium map (speed-coloured track, segment badges) |
| `plot_stats.py` | Speed-vs-distance Plotly chart, top-10 speed finder |
| `strava_auth.py` | Strava OAuth 2.0 helpers (auth URL, token exchange, refresh) |
| `strava_fetch.py` | Strava API calls — list NordicSki activities, fetch GPS streams |
| `app.py` | Streamlit app — wires everything together |
| `kalman_smoother.tex` | LaTeX write-up of the Kalman + RTS smoother theory |
| `requirements.txt` | Python dependencies |

## Processing pipeline (GPXTrack)

1. **Parse** — GPX file via `parse_gpx()`, or Strava streams via `GPXTrack.from_points()`
2. **Segment** — split on time gaps > `gap_sec` (default 120 s)
3. **Trim** — drop first `segment_start_trim` points (default 60) from each segment;
   GPS fixes right after a break are unreliable while the receiver re-acquires satellites
4. **Kalman + RTS smooth** — forward Kalman filter with innovation gating, then
   Rauch-Tung-Striebel backward smoother; elevation is median-smoothed (window=6)
   before being stored; speed comes from the smoothed velocity state `(vx, vy)`,
   not from finite differences
5. **Remove pauses** — strip contiguous slow stretches (< `pause_speed_ms` m/s
   for at least `pause_min_sec` seconds) using a centred sliding-window speed estimate

## Key classes and functions

### `GPXTrack` (gpx_filter.py)
```python
track = GPXTrack("file.gpx")            # from file
track = GPXTrack.from_points(points)    # from Strava streams

track.segments          # List[List[Point]]
track.points            # flat List[Point]
track.stats             # List[dict] per segment
track.summary           # dict of totals
track.to_dataframe()    # pandas DataFrame: segment, lat, lon, ele, time, speed_ms, cum_dist_m
track.stats_dataframe() # pandas DataFrame: one row per segment
track.print_report()    # pretty-print to stdout
```

Parameters (all have sensible defaults):
- `gap_sec` — time gap (s) that starts a new segment (default 120)
- `segment_start_trim` — points to drop from segment start (default 60)
- `sigma_gps` — GPS measurement noise in metres (default 15); higher = smoother
- `sigma_a` — process noise / expected acceleration in m/s² (default 0.8);
  higher = filter follows GPS more closely; terrain-dependent
- `gate_alpha` — chi-squared tail probability for innovation gate, df=2 (default 0.01)
- `pause_speed_ms` — speed threshold for pause detection (default 0.8 m/s)
- `pause_min_sec` — minimum pause duration to strip (default 8 s)

### `Point` dataclass (gpx_filter.py)
```python
@dataclass
class Point:
    lat: float
    lon: float
    ele: Optional[float]    # median-smoothed after Kalman pass
    time: datetime
    speed_ms: Optional[float]  # from Kalman velocity state; None on raw points
```

### plot_track.py
```python
fmap = build_map(track)          # returns folium.Map — call before adding extra layers
add_layer_control(fmap)          # always add this last, after all FeatureGroups
save_map(fmap, path, open_browser=True)
plot(path)                       # convenience: build + save + open
```

### plot_stats.py
```python
df  = _enrich(track.to_dataframe())      # adds speed_kmh, makes cum_dist_m continuous
top = find_top_speeds(df)                # top 10, min 500 m separation
fig = build_speed_chart(track, df, top)  # plotly Figure
add_top_speed_markers(fmap, top)         # gold badges on folium map
```

### strava_auth.py
```python
url = auth_url(client_id, redirect_uri)     # send user here
tokens = exchange_code(client_id, secret, code)  # after redirect
store_tokens(tokens)                        # writes to st.session_state
token = ensure_valid_token(client_id, secret)    # refreshes if needed
logout()
```

### strava_fetch.py
```python
activities = list_activities(access_token)     # NordicSki only, paginated
points = fetch_points(access_token, activity)  # returns List[Point]
label  = activity_label(activity)              # human-readable string for selectbox
```

## Streamlit app (app.py)

OAuth flow: `handle_oauth()` checks `st.query_params` for `?code=...` on redirect,
exchanges it, stores tokens in `st.session_state`, clears params and reruns.

Sidebar sliders feed directly into `_fetch_and_analyse()` which is cached with
`@st.cache_data` — changing any slider re-runs the Kalman smoother automatically.

Map uses `st_folium(..., returned_objects=[])` to prevent map interactions
(pan/zoom) from triggering a Streamlit rerun.

## Next ideas (not yet built)
- Adaptive `sigma_a` driven by local elevation gradient (terrain-aware smoothing)
- Deploy to Streamlit Cloud
- Multi-activity comparison / season statistics
