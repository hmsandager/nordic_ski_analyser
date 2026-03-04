#!/usr/bin/env python3
"""
Speed analysis for nordic ski GPX tracks.

Generates (next to each .gpx file):
  <name>_speed.html  – interactive Plotly speed-vs-distance chart
  <name>_map.html    – track map updated with top-10 speed markers

The speed chart shows:
  - Raw instantaneous speed (faint)
  - Rolling-window smoothed speed (bold)
  - Top-10 fastest locations as gold vertical lines

Usage:
    python plot_stats.py [file.gpx ...]
    python plot_stats.py                   # all *.gpx in current dir
"""
from __future__ import annotations

import math
import os
import sys
import glob as _glob
import webbrowser
from typing import List

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from gpx_filter import GPXTrack
from plot_track import build_map, save_map, add_layer_control


# ── Parameters ────────────────────────────────────────────────────────────────

TOP_N      = 10    # number of fastest locations to find
MIN_SEP_M  = 500   # minimum distance between two top-speed markers (metres)


# ── Speed computation ─────────────────────────────────────────────────────────

def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add speed_kmh, gradient, and make cum_dist_m continuous across segments."""
    df = df.copy()
    df["speed_kmh"] = df["speed_ms"] * 3.6

    # cum_dist_m resets to 0 at the start of each segment; make it continuous
    offset = 0.0
    for seg_id in sorted(df["segment"].unique()):
        mask = df["segment"] == seg_id
        df.loc[mask, "cum_dist_m"] += offset
        offset = df.loc[mask, "cum_dist_m"].iloc[-1]

    # Terrain gradient (rise / run, dimensionless) per step, computed within
    # each segment so we never diff across a segment boundary.
    df["gradient"] = np.nan
    for seg_id in df["segment"].unique():
        mask  = df["segment"] == seg_id
        seg   = df[mask]
        if len(seg) < 2:
            continue
        lats    = seg["lat"].values
        lons    = seg["lon"].values
        eles    = seg["ele"].values
        lat_mid = np.radians((lats[:-1] + lats[1:]) / 2)
        dlat    = np.diff(lats) * 111_319.9
        dlon    = np.diff(lons) * 111_319.9 * np.cos(lat_mid)
        horiz   = np.sqrt(dlat ** 2 + dlon ** 2)
        d_ele   = np.diff(eles)
        with np.errstate(invalid="ignore", divide="ignore"):
            grad = np.where(horiz > 0.5, d_ele / horiz, np.nan)
        df.loc[mask, "gradient"] = np.concatenate([[np.nan], grad])

    return df


def find_top_speeds(df: pd.DataFrame, n: int = TOP_N, min_sep_m: float = MIN_SEP_M) -> pd.DataFrame:
    """
    Return a DataFrame of the n fastest locations (by Kalman speed) such
    that no two selected points are closer than min_sep_m metres.

    Uses a greedy approach: sort candidates by speed descending, accept each
    one only if it is far enough from all already-accepted points.

    The returned DataFrame has a 1-based 'rank' column (1 = fastest).
    """
    candidates = df.sort_values("speed_kmh", ascending=False)
    selected: List[pd.Series] = []

    for _, row in candidates.iterrows():
        if len(selected) >= n:
            break
        too_close = False
        for sel in selected:
            dx = (row["lon"] - sel["lon"]) * 111_319.9 * math.cos(math.radians(row["lat"]))
            dy = (row["lat"] - sel["lat"]) * 111_319.9
            if math.sqrt(dx * dx + dy * dy) < min_sep_m:
                too_close = True
                break
        if not too_close:
            selected.append(row)

    result = pd.DataFrame(selected).reset_index(drop=True)
    result.insert(0, "rank", range(1, len(result) + 1))
    return result


# ── Speed chart ───────────────────────────────────────────────────────────────

# One colour per segment
_SEG_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                 "#8c564b", "#e377c2", "#7f7f7f"]


def _hex_to_rgba(hex_colour: str, alpha: float) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def build_speed_chart(track: GPXTrack, df: pd.DataFrame, top: pd.DataFrame) -> go.Figure:
    """
    Build a Plotly Figure with two panels sharing the x-axis (distance):
      - Top (60%): speed (km/h) per segment, with gold top-10 markers
      - Bottom (40%): elevation (m) per segment
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.60, 0.40],
        vertical_spacing=0.06,
    )

    for seg_id in sorted(df["segment"].unique()):
        seg    = df[df["segment"] == seg_id]
        colour = _SEG_COLOURS[seg_id % len(_SEG_COLOURS)]
        label  = f"Segment {seg_id + 1}"

        # ── 95 % CI shaded band (row 1) ───────────────────────────────────────
        if "speed_std" in seg.columns:
            z = 1.96
            upper = (seg["speed_ms"] + z * seg["speed_std"]) * 3.6
            lower = ((seg["speed_ms"] - z * seg["speed_std"]).clip(lower=0)) * 3.6
            fill_col = _hex_to_rgba(colour, 0.15)
            x_km = seg["cum_dist_m"] / 1000
            # Upper boundary (invisible line — anchor for fill)
            fig.add_trace(go.Scatter(
                x=x_km, y=upper,
                mode="lines", line=dict(width=0),
                legendgroup=label, showlegend=False,
                hoverinfo="skip",
            ), row=1, col=1)
            # Lower boundary with fill back to upper
            fig.add_trace(go.Scatter(
                x=x_km, y=lower,
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor=fill_col,
                legendgroup=label, showlegend=False,
                hoverinfo="skip",
            ), row=1, col=1)

        # ── Speed trace (row 1) ────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=seg["cum_dist_m"] / 1000,
            y=seg["speed_kmh"],
            mode="lines",
            line=dict(color=colour, width=2.5),
            name=label,
            legendgroup=label,
            showlegend=True,
            customdata=seg[["time", "ele"]].values,
            hovertemplate=(
                "<b>%{x:.2f} km</b><br>"
                "Speed: %{y:.1f} km/h<br>"
                "Time: %{customdata[0]}<br>"
                "Ele: %{customdata[1]:.0f} m"
                f"<extra>{label}</extra>"
            ),
        ), row=1, col=1)

        # ── Elevation trace (row 2) ────────────────────────────────────────────
        ele_seg = seg.dropna(subset=["ele"])
        if not ele_seg.empty:
            fig.add_trace(go.Scatter(
                x=ele_seg["cum_dist_m"] / 1000,
                y=ele_seg["ele"],
                mode="lines",
                line=dict(color=colour, width=1.8),
                name=label,
                legendgroup=label,
                showlegend=False,
                hovertemplate=(
                    "<b>%{x:.2f} km</b><br>"
                    "Elevation: %{y:.0f} m"
                    f"<extra>{label}</extra>"
                ),
            ), row=2, col=1)


# Top-10 markers: gold vertical lines on speed panel only + annotations
    for _, row in top.iterrows():
        dist_km = row["cum_dist_m"] / 1000
        spd     = row["speed_kmh"]
        rank    = int(row["rank"])

        fig.add_vline(
            x=dist_km,
            line=dict(color="#e6b800", width=1.5, dash="dot"),
            row=1, col=1,
        )
        fig.add_annotation(
            x=dist_km,
            y=spd,
            xref="x",
            yref="y",
            text=f"#{rank}<br>{spd:.1f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#e6b800",
            arrowwidth=1.5,
            font=dict(size=10, color="#7a6000"),
            bgcolor="rgba(255,240,150,0.85)",
            bordercolor="#e6b800",
            borderwidth=1,
            borderpad=3,
        )

    fig.update_layout(
        title=dict(
            text=f"Speed & elevation — {os.path.basename(track.path)}",
            font=dict(size=16),
        ),
        yaxis=dict(title="Speed (km/h)", rangemode="tozero"),
        yaxis2=dict(title="Elevation (m)"),
        xaxis2=dict(title="Distance (km)"),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.12),
        margin=dict(t=70, b=80, l=60, r=30),
        template="plotly_white",
    )

    return fig


# ── Speed histogram ────────────────────────────────────────────────────────────

def build_speed_histogram(df: pd.DataFrame) -> go.Figure:
    """Histogram of time spent at each speed (0.5 km/h bins)."""
    speeds = df["speed_kmh"].dropna()
    median = float(speeds.median())

    bins    = np.arange(0, speeds.max() + 0.5, 0.5)
    counts, edges = np.histogram(speeds, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=centres,
        y=counts,
        width=0.48,
        marker=dict(
            color=centres,
            colorscale="RdYlGn",
            showscale=False,
        ),
        name="",
        showlegend=False,
        hovertemplate="%{x:.1f} km/h: %{y} pts<extra></extra>",
    ))

    fig.add_vline(
        x=median,
        line=dict(color="black", width=2, dash="dash"),
        annotation_text=f"Median {median:.1f} km/h",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Speed distribution",
        xaxis_title="Speed (km/h)",
        yaxis_title="Time (GPS points ≈ seconds)",
        bargap=0,
        margin=dict(t=70, b=50, l=60, r=30),
        template="plotly_white",
    )
    return fig


# ── Terrain breakdown ──────────────────────────────────────────────────────────

def build_terrain_breakdown(df: pd.DataFrame, thresh_pct: float = 2.0) -> go.Figure:
    """
    Stacked bar chart showing the fraction of time per segment spent
    climbing, on flat terrain, and descending.

    `thresh_pct` is the gradient threshold (%) that separates flat from
    climbing / descending.
    """
    thresh = thresh_pct / 100.0

    d = df.dropna(subset=["gradient"]).copy()
    d["terrain"] = "Flat"
    d.loc[d["gradient"] >  thresh, "terrain"] = "Climb"
    d.loc[d["gradient"] < -thresh, "terrain"] = "Descent"

    segments = sorted(d["segment"].unique())
    x_labels = [f"Segment {s + 1}" for s in segments]

    cats = [("Climb", "#e74c3c"), ("Flat", "#bdc3c7"), ("Descent", "#3498db")]

    fig = go.Figure()
    for cat_name, colour in cats:
        pcts = []
        for seg_id in segments:
            seg_d = d[d["segment"] == seg_id]
            total = len(seg_d)
            pcts.append((seg_d["terrain"] == cat_name).sum() / total * 100 if total else 0)

        fig.add_trace(go.Bar(
            x=x_labels,
            y=pcts,
            name=cat_name,
            marker_color=colour,
            hovertemplate="%{x}<br>" + cat_name + ": %{y:.1f}%<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        title=f"Terrain breakdown  (±{thresh_pct:.0f}% threshold)",
        xaxis_title="Segment",
        yaxis=dict(title="Time (%)", range=[0, 100]),
        legend=dict(orientation="h", y=1.10),
        margin=dict(t=70, b=50, l=60, r=30),
        template="plotly_white",
    )
    return fig


# ── Map markers ───────────────────────────────────────────────────────────────

def add_top_speed_markers(fmap: folium.Map, top: pd.DataFrame) -> None:
    """Add gold numbered badge markers for each top-speed location."""
    layer = folium.FeatureGroup(name="Top 10 speeds", show=True)

    for _, row in top.iterrows():
        rank = int(row["rank"])
        spd  = row["speed_kmh"]
        popup_html = (
            f"<b>#{rank} fastest</b><br>"
            f"Speed: {spd:.1f} km/h<br>"
            f"Time: {row['time']}<br>"
            f"Elevation: {row['ele']:.0f} m<br>"
            f"Distance: {row['cum_dist_m']/1000:.2f} km"
        )
        badge_html = (
            f'<div style="background:#e6b800;color:#333;font-weight:bold;'
            f"border:2px solid white;border-radius:50%;"
            f"width:26px;height:26px;line-height:26px;"
            f'text-align:center;font-size:12px;box-shadow:0 1px 5px rgba(0,0,0,.5)">'
            f"{rank}"
            f"</div>"
        )
        folium.Marker(
            location=(row["lat"], row["lon"]),
            popup=folium.Popup(popup_html, max_width=190),
            tooltip=f"#{rank}: {spd:.1f} km/h",
            icon=folium.DivIcon(html=badge_html, icon_size=(26, 26), icon_anchor=(13, 13)),
        ).add_to(layer)

    layer.add_to(fmap)


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_top_speeds(top: pd.DataFrame) -> None:
    """Print the top-N speed table to stdout."""
    print(f"\n  {'#':>2}  {'Speed':>9}  {'Time':>10}  {'Dist':>7}  {'Ele':>6}")
    print(f"  {'─'*46}")
    for _, row in top.iterrows():
        t = row["time"]
        time_str = t.strftime("%H:%M:%S") if hasattr(t, "strftime") else str(t)[:8]
        print(
            f"  {int(row['rank']):>2}  "
            f"{row['speed_kmh']:>7.1f} k/h  "
            f"{time_str:>10}  "
            f"{row['cum_dist_m']/1000:>6.2f}km  "
            f"{row['ele']:>5.0f}m"
        )
    print()


# ── Main entry point ──────────────────────────────────────────────────────────

def analyse(path: str, open_browser: bool = True) -> None:
    """Run full analysis for one GPX file: speed chart + annotated map."""
    print(f"\nAnalysing: {path}")
    track = GPXTrack(path)

    if not track.segments:
        print("  No segments — skipping.")
        return

    df  = _enrich(track.to_dataframe())
    top = find_top_speeds(df)

    print_top_speeds(top)

    # ── Speed chart ────────────────────────────────────────────────────────────
    fig       = build_speed_chart(track, df, top)
    speed_out = os.path.splitext(path)[0] + "_speed.html"
    fig.write_html(speed_out, include_plotlyjs="cdn")
    print(f"  Speed chart → {speed_out}")

    # ── Map with top-10 markers ────────────────────────────────────────────────
    fmap    = build_map(track)
    add_top_speed_markers(fmap, top)
    add_layer_control(fmap, collapsed=False)
    map_out = os.path.splitext(path)[0] + "_map.html"
    save_map(fmap, map_out, open_browser=False)

    if open_browser:
        webbrowser.open(f"file://{os.path.abspath(speed_out)}")
        webbrowser.open(f"file://{os.path.abspath(map_out)}")


if __name__ == "__main__":
    if sys.argv[1:]:
        paths = sys.argv[1:]
    else:
        name = input("Filename: ").strip()
        paths = [name] if name else []

    if not paths:
        print("No file specified.")
        sys.exit(1)

    for p in paths:
        analyse(p)
