#!/usr/bin/env python3
"""
Plot a filtered GPX track on an interactive map.

Creates an HTML file with:
  - Track colored by speed (blue = slow / uphill, red = fast / downhill)
  - Start (green) and end (red) markers with timestamps
  - A segment break marker (orange) if multiple segments exist
  - A speed-color legend in the corner
  - A summary popup on the map

Usage:
    python plot_track.py [file.gpx ...]          # saves file_map.html next to each gpx
    python plot_track.py                          # processes all *.gpx in current dir
"""
from __future__ import annotations

import colorsys
import math
import os
import sys
import glob as _glob
import webbrowser
from typing import List, Tuple

import folium

from gpx_filter import GPXTrack, Point, haversine


# ── Color helpers ─────────────────────────────────────────────────────────────

# Speed range used for the full color scale (km/h).
# Points faster than SPEED_MAX are clamped to the top colour.
SPEED_MIN_KMH = 2.0
SPEED_MAX_KMH = 35.0


def _speed_to_hex(speed_ms: float) -> str:
    """Map a speed in m/s to a hex colour string (blue → cyan → green → yellow → red)."""
    speed_kmh = speed_ms * 3.6
    t = (speed_kmh - SPEED_MIN_KMH) / (SPEED_MAX_KMH - SPEED_MIN_KMH)
    t = max(0.0, min(1.0, t))
    # Hue: 0.67 (blue) at t=0 → 0.0 (red) at t=1
    hue = 0.67 * (1.0 - t)
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


# ── Legend HTML ───────────────────────────────────────────────────────────────

def _legend_html(speed_min: float, speed_max: float) -> str:
    """Build an HTML speed-colour legend as a folium macro."""
    steps = 8
    swatches = ""
    for i in range(steps):
        t = i / (steps - 1)
        speed_ms = (speed_min + t * (speed_max - speed_min)) / 3.6
        colour = _speed_to_hex(speed_ms)
        label = f"{speed_min + t * (speed_max - speed_min):.0f}"
        swatches += (
            f'<div style="display:inline-block;text-align:center;margin:0 3px">'
            f'<div style="width:28px;height:14px;background:{colour};border-radius:2px"></div>'
            f'<div style="font-size:10px">{label}</div>'
            f"</div>"
        )

    return f"""
    <div style="
        position: fixed;
        bottom: 30px; left: 30px;
        z-index: 1000;
        background: rgba(255,255,255,0.88);
        padding: 8px 12px;
        border-radius: 6px;
        box-shadow: 0 1px 5px rgba(0,0,0,.4);
        font-family: sans-serif;
    ">
        <div style="font-weight:bold;margin-bottom:4px;font-size:12px">Speed (km/h)</div>
        {swatches}
    </div>
    """


# ── Plotting ──────────────────────────────────────────────────────────────────

def _centre(track: GPXTrack) -> Tuple[float, float]:
    """Return the geographic centre of the track."""
    pts = track.points
    return (
        sum(p.lat for p in pts) / len(pts),
        sum(p.lon for p in pts) / len(pts),
    )


def _draw_speed_line(
    fmap: folium.Map,
    seg: List[Point],
    seg_id: int,
) -> None:
    """Draw one segment as individual coloured polyline pairs (one per GPS step)."""
    feature_group = folium.FeatureGroup(name=f"Segment {seg_id + 1}", show=True)

    for i in range(len(seg) - 1):
        a, b = seg[i], seg[i + 1]
        dt = (b.time - a.time).total_seconds()
        speed_ms = haversine(a, b) / dt if dt > 0 else 0.0
        colour = _speed_to_hex(speed_ms)

        folium.PolyLine(
            locations=[(a.lat, a.lon), (b.lat, b.lon)],
            color=colour,
            weight=4,
            opacity=0.85,
        ).add_to(feature_group)

    feature_group.add_to(fmap)


def _endpoint_marker(fmap: folium.Map, pt: Point, label: str, colour: str, icon: str) -> None:
    folium.Marker(
        location=(pt.lat, pt.lon),
        popup=folium.Popup(
            f"<b>{label}</b><br>{pt.time.strftime('%H:%M:%S UTC')}",
            max_width=160,
        ),
        icon=folium.Icon(color=colour, icon=icon, prefix="fa"),
    ).add_to(fmap)


def _segment_start_marker(fmap: folium.Map, pt: Point, seg_id: int, stats: dict) -> None:
    """Numbered badge marker at the start of a segment with stats popup."""
    popup_html = (
        f"<b>Segment {seg_id}</b><br>"
        f"Start: {stats['start'].strftime('%H:%M:%S')}<br>"
        f"End: {stats['end'].strftime('%H:%M:%S')}<br>"
        f"Distance: {stats['distance_km']:.2f} km<br>"
        f"Duration: {stats['duration_min']:.0f} min<br>"
        f"Avg speed: {stats['avg_kmh']:.1f} km/h<br>"
        f"Elevation ↑{stats['ele_gain_m']:.0f} m  ↓{stats['ele_loss_m']:.0f} m"
    )
    badge_html = (
        f'<div style="'
        f"background:#ff7800;color:white;font-weight:bold;"
        f"border:2px solid white;border-radius:50%;"
        f"width:24px;height:24px;line-height:24px;"
        f'text-align:center;font-size:13px;box-shadow:0 1px 4px rgba(0,0,0,.5)">'
        f"{seg_id}"
        f"</div>"
    )
    folium.Marker(
        location=(pt.lat, pt.lon),
        popup=folium.Popup(popup_html, max_width=200),
        tooltip=f"Segment {seg_id}",
        icon=folium.DivIcon(html=badge_html, icon_size=(24, 24), icon_anchor=(12, 12)),
    ).add_to(fmap)


def add_layer_control(fmap: folium.Map, collapsed: bool = True) -> None:
    """Add a LayerControl to a map. Call this once, after all layers are added."""
    folium.LayerControl(collapsed=collapsed).add_to(fmap)


def build_map(track: GPXTrack) -> folium.Map:
    """
    Build and return a folium Map for a GPXTrack.

    Does not save or open anything — call this when you want to add extra
    layers (e.g. top-speed markers) before saving.
    """
    centre = _centre(track)
    fmap = folium.Map(location=centre, zoom_start=13, tiles="OpenStreetMap")

    for seg_id, seg in enumerate(track.segments):
        _draw_speed_line(fmap, seg, seg_id)

    _endpoint_marker(fmap, track.segments[0][0],   "Start", "green", "play")
    _endpoint_marker(fmap, track.segments[-1][-1], "End",   "red",   "stop")

    all_stats = track.stats
    for seg_id, (seg, seg_stats) in enumerate(zip(track.segments, all_stats)):
        _segment_start_marker(fmap, seg[0], seg_id + 1, seg_stats)

    sm = track.summary
    summary_html = (
        f"<b>{os.path.basename(track.path)}</b><br>"
        f"Distance: {sm['distance_km']:.2f} km<br>"
        f"Moving time: {sm['duration_min']:.0f} min<br>"
        f"Elevation ↑{sm['ele_gain_m']:.0f} m  ↓{sm['ele_loss_m']:.0f} m<br>"
        f"Segments: {sm['segments']}"
    )
    folium.Marker(
        location=centre,
        popup=folium.Popup(summary_html, max_width=200),
        icon=folium.DivIcon(html='<div style="font-size:0"></div>', icon_size=(0, 0)),
    ).add_to(fmap)

    fmap.get_root().html.add_child(
        folium.Element(_legend_html(SPEED_MIN_KMH, SPEED_MAX_KMH))
    )

    return fmap


def save_map(fmap: folium.Map, out_path: str, open_browser: bool = True) -> str:
    """Save a folium map to *out_path* and optionally open it in the browser."""
    fmap.save(out_path)
    print(f"  Saved → {out_path}")
    if open_browser:
        webbrowser.open(f"file://{os.path.abspath(out_path)}")
    return out_path


def plot(path: str, open_browser: bool = True) -> str:
    """
    Load, filter, and plot a GPX track.  Returns the path of the saved HTML file.

    For more control (e.g. adding extra markers before saving) use
    build_map() + save_map() directly.
    """
    track = GPXTrack(path)
    if not track.segments:
        print(f"  No segments found in {path}, skipping.")
        return ""
    fmap = build_map(track)
    add_layer_control(fmap)
    out_path = os.path.splitext(path)[0] + "_map.html"
    return save_map(fmap, out_path, open_browser)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    paths = sys.argv[1:] or _glob.glob("*.gpx")
    if not paths:
        print("Usage: python plot_track.py [file.gpx ...]")
        sys.exit(1)

    for p in paths:
        print(f"Processing: {p}")
        plot(p)
