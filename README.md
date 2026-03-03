# Nordic Ski GPS Analysis

Streamlit app that pulls nordic ski activities from Strava, cleans the GPS
track with a Kalman smoother, and produces an interactive map and speed profile.

## Features

- Strava OAuth integration — no manual GPX downloads
- Kalman + RTS smoother for GPS cleaning (see `kalman_smoother.tex` for theory)
- Speed profile chart (Plotly) with top-10 fastest locations
- Interactive map (Folium/OpenStreetMap) with speed-coloured track
- Sidebar sliders for all smoother parameters — results update live

## Setup

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# fill in your Strava client_id and client_secret
streamlit run app.py
```

### Strava API credentials

1. Create an app at [strava.com/settings/api](https://www.strava.com/settings/api)
2. Set **Authorization Callback Domain** to `localhost`
3. Copy **Client ID** and **Client Secret** into `.streamlit/secrets.toml`

## Local GPX files

The core library works independently of Strava:

```python
from gpx_filter import GPXTrack

track = GPXTrack("activity.gpx")
print(track.summary)
df = track.to_dataframe()   # pandas DataFrame
```

```bash
python plot_stats.py activity.gpx   # generates _map.html and _speed.html
```

## Smoother parameters

| Parameter | Default | Effect |
|---|---|---|
| `sigma_gps` | 15 m | GPS measurement noise — higher gives smoother track |
| `sigma_a` | 0.8 m/s² | Expected acceleration — higher follows GPS more closely |
| `gate_alpha` | 0.01 | Fraction of points rejected as outliers before smoothing |
| `segment_start_trim` | 60 pts | Points dropped from segment start (avoids satellite re-acquisition glitches) |
| `gap_sec` | 120 s | Time gap that splits the track into separate segments |

## Files

```
app.py              Streamlit app
gpx_filter.py       Core library (Point, GPXTrack, Kalman smoother)
plot_track.py       Folium map builder
plot_stats.py       Plotly speed chart and top-10 finder
strava_auth.py      OAuth 2.0 helpers
strava_fetch.py     Strava API client
kalman_smoother.tex LaTeX write-up of the statistical method
requirements.txt    Python dependencies
```
