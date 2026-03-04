#!/usr/bin/env python3
"""
GPX filter for nordic ski tracks.

Pipeline applied in order:
  1. Segment splitting – splits on time gaps > gap_sec (default 120 s)
  2. Kalman + RTS smoothing – per segment, rejects GPS spikes via innovation
     gating, then runs a forward Kalman filter followed by a backward RTS
     smoother so that future observations also pull past estimates into place
  3. Pause removal – removes stretches where local speed stays below threshold

Typical usage
─────────────
    from gpx_filter import GPXTrack

    track = GPXTrack("Skei_dag5.gpx")

    track.segments          # List[List[Point]]  – smoothed segments
    track.stats             # List[dict]         – per-segment stats
    track.summary           # dict               – totals across all segments
    track.to_dataframe()    # pandas DataFrame   – one row per point (requires pandas)
    track.stats_dataframe() # pandas DataFrame   – one row per segment
    track.print_report()    # pretty-print table to stdout

Kalman parameters
─────────────────
    sigma_gps  – GPS measurement noise (m).  Lower = trust GPS more.  Default 5.
    sigma_a_base        – Base process noise (m/s²).  Sets the floor for how
                          much acceleration is allowed.  Default 0.8.
    sigma_a_sensitivity – How much the terrain gradient boosts sigma_a.
                          0 = no adaptation; larger values make the smoother
                          follow the GPS more closely on steep gradients.
                          Default 2.0.
    gate_alpha – Tail probability for the chi-squared innovation gate (2 DOF).
                 Default 0.001, i.e. only measurements beyond the 99.9 % quantile
                 are rejected outright before smoothing.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Point:
    lat: float
    lon: float
    ele: Optional[float]
    time: datetime
    speed_ms: Optional[float] = None   # set by Kalman smoother; None on raw points


# ── Geometry ──────────────────────────────────────────────────────────────────

def haversine(a: Point, b: Point) -> float:
    """Straight-line distance in metres between two GPS points."""
    R = 6_371_000.0
    lat1, lon1 = math.radians(a.lat), math.radians(a.lon)
    lat2, lon2 = math.radians(b.lat), math.radians(b.lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


# ── GPX parsing ───────────────────────────────────────────────────────────────

_GPX_NS = "http://www.topografix.com/GPX/1/1"


def _local(tag: str) -> str:
    """Strip XML namespace prefix from a tag, e.g. '{http://...}trkpt' → 'trkpt'."""
    return tag.split("}")[-1] if "}" in tag else tag


def _parse_gpx_root(root) -> List[Point]:
    """Parse track-points from an ElementTree root element.

    Works with any GPX namespace (1.0, 1.1) or no namespace at all by
    matching on local tag names only.
    """
    points: List[Point] = []
    for trkpt in root.iter():
        if _local(trkpt.tag) != "trkpt":
            continue
        try:
            lat = float(trkpt.attrib["lat"])
            lon = float(trkpt.attrib["lon"])
        except (KeyError, ValueError):
            continue
        ele: Optional[float] = None
        time: Optional[datetime] = None
        for child in trkpt:
            name = _local(child.tag)
            if name == "ele" and child.text:
                try:
                    ele = float(child.text)
                except ValueError:
                    pass
            elif name == "time" and child.text:
                try:
                    time = datetime.fromisoformat(child.text.replace("Z", "+00:00"))
                except ValueError:
                    pass
        if time is not None:
            points.append(Point(lat=lat, lon=lon, ele=ele, time=time))
    return points


def parse_gpx(path: str) -> List[Point]:
    """Return all track-points from a GPX file path, in chronological order."""
    return _parse_gpx_root(ET.parse(path).getroot())


def parse_gpx_bytes(data: bytes) -> List[Point]:
    """Return all track-points from GPX file contents (bytes), in chronological order."""
    import io
    return _parse_gpx_root(ET.parse(io.BytesIO(data)).getroot())


def count_trkpt(data: bytes) -> int:
    """Return the total number of trkpt elements in GPX bytes, ignoring timestamps."""
    import io
    root = ET.parse(io.BytesIO(data)).getroot()
    return sum(1 for el in root.iter() if _local(el.tag) == "trkpt")


# ── Filter functions ──────────────────────────────────────────────────────────

def _adaptive_speed_gate(
    points: List[Point],
    max_jump_factor: float,
    window_sec: float = 5.0,
) -> List[Point]:
    """
    Remove GPS spikes using a speed-adaptive threshold based solely on
    previously accepted points.

    Local speed is the median of raw step speeds among the accepted points
    within the preceding ``window_sec`` seconds.  Because only already-
    accepted (non-spike) points contribute to the estimate, a cluster of
    outliers never inflates the threshold — the speed estimate stays
    grounded in true movement regardless of how many consecutive spikes
    occur.

    A point is rejected when its implied speed from the last accepted point
    exceeds ``max_jump_factor × local_speed_estimate``.

    Hard floor: 3 m/s (~11 km/h) so early-segment points with a still-
    forming speed history are not over-filtered.
    """
    ABS_FLOOR = 3.0   # m/s

    if len(points) < 2:
        return points

    accepted: List[Point] = [points[0]]

    for pt in points[1:]:
        # ── local speed from recently accepted raw points only ─────────────────
        t_pt = pt.time
        recent: List[Point] = []
        for p in reversed(accepted):
            if (t_pt - p.time).total_seconds() > window_sec:
                break
            recent.append(p)
        recent.reverse()

        if len(recent) >= 2:
            step_speeds = []
            for j in range(1, len(recent)):
                dt_j = (recent[j].time - recent[j - 1].time).total_seconds()
                if dt_j > 0:
                    step_speeds.append(haversine(recent[j - 1], recent[j]) / dt_j)
            if step_speeds:
                step_speeds.sort()
                local_speed = step_speeds[len(step_speeds) // 2]   # median
            else:
                local_speed = 0.0
        else:
            local_speed = 0.0

        threshold = max(max_jump_factor * local_speed, ABS_FLOOR)

        dt = (pt.time - accepted[-1].time).total_seconds()
        if dt <= 0:
            accepted.append(pt)
            continue
        if haversine(accepted[-1], pt) / dt <= threshold:
            accepted.append(pt)

    return accepted


def kalman_smooth(
    points: List[Point],
    sigma_gps: float = 10.0,
    sigma_a_base: float = 0.8,
    sigma_a_sensitivity: float = 1.2,
    gate_alpha: float = 0.01,
) -> Tuple[List[Point], int]:
    """
    Smooth a GPS segment using a Kalman filter + RTS backward smoother.

    State vector: [x, y, vx, vy] in local Cartesian metres (origin = first point).
    Motion model: constant velocity with adaptive random acceleration.
    Measurement:  [x, y] with isotropic noise sigma_gps metres.

    Adaptive process noise: sigma_a is computed per step from the terrain
    gradient (rise/run from smoothed elevation).  Downhill steps get the full
    boost (gradient ≤ 0 → weight 1.0); uphill steps get 40% of the boost
    (weight 0.4), reflecting that GPS error is more often a downhill spike.
    The boost saturates via tanh so the effect stays bounded:
        sigma_a[i] = base + sensitivity × weight × tanh(|slope| / 0.10)
    where 0.10 means a 10% grade produces ~76% of the maximum additional noise.

    Innovation gating: each raw GPS reading is tested against the predicted
    position before it is assimilated.  Readings with a Mahalanobis distance²
    exceeding the chi-squared quantile for `gate_alpha` (2 DOF) are skipped —
    the filter just coasts on its prediction for that step.  After the forward
    pass a backward RTS (Rauch-Tung-Striebel) smoother refines all estimates
    using future information, giving a globally smooth trajectory.

    Returns (smoothed_points, n_gated) where n_gated is the count of
    measurements rejected by the innovation gate.
    """
    import numpy as np

    # Chi-squared quantile for df=2 has closed form: -2 * ln(alpha)
    gate = -2.0 * math.log(gate_alpha)

    n = len(points)
    if n < 2:
        return points, 0

    # Local flat-earth Cartesian (accurate enough over segment lengths < ~50 km)
    lat0 = math.radians(points[0].lat)
    lon0 = points[0].lon
    lat0_deg = points[0].lat
    m_per_deg_lat = 111_319.9
    m_per_deg_lon = 111_319.9 * math.cos(lat0)

    def to_xy(pt: Point) -> "np.ndarray":
        return np.array([
            (pt.lon - lon0) * m_per_deg_lon,
            (pt.lat - lat0_deg) * m_per_deg_lat,
        ])

    measurements = np.array([to_xy(pt) for pt in points])  # (n, 2)

    # ── Elevation smoothing (median over 6 observations, centred) ──────────────
    # Raw GPS altitude is noisy (±5–10 m typical); smooth before storing.
    ele_raw = np.array([pt.ele if pt.ele is not None else np.nan for pt in points])
    ele_smooth = np.full(n, np.nan)
    half = 3
    for i in range(n):
        window = ele_raw[max(0, i - half): min(n, i + half)]
        valid  = window[~np.isnan(window)]
        if len(valid) > 0:
            ele_smooth[i] = float(np.median(valid))

    # ── Adaptive sigma_a from terrain gradient ─────────────────────────────────
    # slope = rise / run (dimensionless).  Computed from smoothed elevation and
    # raw Cartesian step distances.  Downhill gets full weight (GPS spikes tend
    # to appear as false downhill accelerations); uphill gets 40% weight.
    # A tanh cap keeps the effect bounded: 10% grade → tanh(1) ≈ 0.76 of max.
    G_SCALE = 0.10   # grade at which tanh argument = 1

    dm = np.diff(measurements, axis=0)                              # (n-1, 2)
    horiz = np.concatenate([[1.0], np.linalg.norm(dm, axis=1)])    # (n,)
    d_ele = np.zeros(n)
    d_ele[1:] = np.diff(ele_smooth)                                 # NaN propagates
    with np.errstate(invalid="ignore", divide="ignore"):
        raw_slope = d_ele / horiz
    raw_slope[0] = 0.0
    raw_slope[horiz < 0.5] = 0.0
    raw_slope[~np.isfinite(raw_slope)] = 0.0

    weight = np.where(raw_slope <= 0, 1.0, 0.4)    # downhill: full; uphill: 40%
    sigma_a_arr = (
        sigma_a_base
        + sigma_a_sensitivity * weight * np.tanh(np.abs(raw_slope) / G_SCALE)
    )

    H = np.array([[1., 0., 0., 0.],
                  [0., 1., 0., 0.]])
    R = sigma_gps ** 2 * np.eye(2)
    I4 = np.eye(4)

    # Storage for forward pass
    xf = np.zeros((n, 4))          # filtered state
    Pf = np.zeros((n, 4, 4))       # filtered covariance
    xp = np.zeros((n, 4))          # predicted state
    Pp = np.zeros((n, 4, 4))       # predicted covariance
    Fs = np.zeros((n, 4, 4))       # transition matrices (indexed by destination)
    n_gated = 0

    # Initialise: position from first point, velocity = 0
    xf[0] = [measurements[0, 0], measurements[0, 1], 0., 0.]
    Pf[0] = np.diag([sigma_gps ** 2, sigma_gps ** 2, 9., 9.])
    xp[0] = xf[0]
    Pp[0] = Pf[0]
    Fs[0] = I4

    # ── Forward Kalman filter ──────────────────────────────────────────────────
    for i in range(1, n):
        dt = max((points[i].time - points[i - 1].time).total_seconds(), 0.01)

        F = np.array([[1., 0., dt, 0.],
                      [0., 1., 0., dt],
                      [0., 0., 1., 0.],
                      [0., 0., 0., 1.]])
        Fs[i] = F

        # Random-acceleration process noise (adaptive per step)
        sa = float(sigma_a_arr[i])
        Q = sa ** 2 * np.array([
            [dt ** 4 / 4, 0.,          dt ** 3 / 2, 0.         ],
            [0.,          dt ** 4 / 4, 0.,          dt ** 3 / 2],
            [dt ** 3 / 2, 0.,          dt ** 2,     0.         ],
            [0.,          dt ** 3 / 2, 0.,          dt ** 2    ],
        ])

        # Predict
        xp[i] = F @ xf[i - 1]
        Pp[i] = F @ Pf[i - 1] @ F.T + Q

        # Innovation and gating
        innov = measurements[i] - H @ xp[i]
        S = H @ Pp[i] @ H.T + R
        S_inv = np.linalg.inv(S)
        maha2 = float(innov @ S_inv @ innov)

        if maha2 <= gate:
            K = Pp[i] @ H.T @ S_inv
            xf[i] = xp[i] + K @ innov
            Pf[i] = (I4 - K @ H) @ Pp[i]
        else:
            # Outlier: coast without updating
            xf[i] = xp[i]
            Pf[i] = Pp[i]
            n_gated += 1

    # ── RTS backward smoother ──────────────────────────────────────────────────
    xs = xf.copy()
    Ps = Pf.copy()

    for i in range(n - 2, -1, -1):
        G = Pf[i] @ Fs[i + 1].T @ np.linalg.inv(Pp[i + 1])
        xs[i] = xf[i] + G @ (xs[i + 1] - xp[i + 1])
        Ps[i] = Pf[i] + G @ (Ps[i + 1] - Pp[i + 1]) @ G.T

    # Convert smoothed Cartesian back to lat/lon.
    # Elevation uses the median-smoothed value; speed from the RTS velocity state.
    result: List[Point] = []
    for i, pt in enumerate(points):
        speed_ms = float(math.sqrt(xs[i, 2] ** 2 + xs[i, 3] ** 2))
        ele = None if np.isnan(ele_smooth[i]) else float(ele_smooth[i])
        result.append(Point(
            lat=lat0_deg + xs[i, 1] / m_per_deg_lat,
            lon=lon0    + xs[i, 0] / m_per_deg_lon,
            ele=ele,
            time=pt.time,
            speed_ms=speed_ms,
        ))

    return result, n_gated


def split_segments(
    points: List[Point],
    gap_sec: float = 120.0,
) -> List[List[Point]]:
    """
    Split a flat point list into segments wherever consecutive timestamps
    differ by more than gap_sec seconds (GPS switched off, long break, etc.).
    """
    if not points:
        return []

    segments: List[List[Point]] = []
    current: List[Point] = [points[0]]

    for pt in points[1:]:
        if (pt.time - current[-1].time).total_seconds() > gap_sec:
            segments.append(current)
            current = [pt]
        else:
            current.append(pt)

    segments.append(current)
    return segments


def _local_speed(points: List[Point], i: int, half_window: float) -> float:
    """Speed (m/s) at index i estimated over a centred ±half_window-second window."""
    n = len(points)
    t0 = points[i].time

    lo = i
    while lo > 0 and (t0 - points[lo - 1].time).total_seconds() <= half_window:
        lo -= 1

    hi = i
    while hi < n - 1 and (points[hi + 1].time - t0).total_seconds() <= half_window:
        hi += 1

    if hi == lo:
        return 0.0

    total_dist = sum(haversine(points[j], points[j + 1]) for j in range(lo, hi))
    total_time = (points[hi].time - points[lo].time).total_seconds()
    return total_dist / total_time if total_time > 0 else 0.0


def remove_pauses(
    points: List[Point],
    speed_ms: float = 0.8,
    window_sec: float = 6.0,
    min_pause_sec: float = 8.0,
) -> Tuple[List[Point], int]:
    """
    Remove contiguous runs of slow/stationary points.

    A point is "paused" when its centred-window local speed < speed_ms.
    A run is only stripped when it lasts at least min_pause_sec seconds —
    momentary slowdowns (turns, hill crests) are kept.

    Returns (filtered_points, n_removed).
    """
    if len(points) < 3:
        return points, 0

    half = window_sec / 2.0
    slow = [_local_speed(points, i, half) < speed_ms for i in range(len(points))]

    result: List[Point] = []
    n_removed = 0
    i = 0

    while i < len(points):
        if slow[i]:
            j = i + 1
            while j < len(points) and slow[j]:
                j += 1
            duration = (points[j - 1].time - points[i].time).total_seconds()
            if duration >= min_pause_sec:
                n_removed += j - i
            else:
                result.extend(points[i:j])
            i = j
        else:
            result.append(points[i])
            i += 1

    return result, n_removed


# ── Per-segment statistics ────────────────────────────────────────────────────

def _segment_stats(seg: List[Point], segment_id: int = 0) -> Dict:
    """Compute stats dict for a single segment."""
    if len(seg) < 2:
        return {}

    dists = [haversine(seg[i], seg[i + 1]) for i in range(len(seg) - 1)]
    times = [(seg[i + 1].time - seg[i].time).total_seconds() for i in range(len(seg) - 1)]

    total_dist = sum(dists)
    duration = (seg[-1].time - seg[0].time).total_seconds()
    avg_speed = total_dist / duration if duration > 0 else 0.0

    speeds_ms = [d / t for d, t in zip(dists, times) if t > 0]
    max_speed = max(speeds_ms) if speeds_ms else 0.0

    ele_vals = [p.ele for p in seg if p.ele is not None]
    gain = loss = 0.0
    if len(ele_vals) >= 2:
        for k in range(len(ele_vals) - 1):
            diff = ele_vals[k + 1] - ele_vals[k]
            if diff > 0:
                gain += diff
            else:
                loss += abs(diff)

    return {
        "segment":      segment_id,
        "start":        seg[0].time,
        "end":          seg[-1].time,
        "points":       len(seg),
        "distance_km":  total_dist / 1000,
        "duration_min": duration / 60,
        "avg_kmh":      avg_speed * 3.6,
        "max_kmh":      max_speed * 3.6,
        "ele_gain_m":   gain,
        "ele_loss_m":   loss,
    }


# ── Main class ────────────────────────────────────────────────────────────────

class GPXTrack:
    """
    Load, smooth, and analyse a GPX file.

    Pipeline
    --------
    1. Split on time gaps > gap_sec  (default 120 s)
    2. Kalman + RTS smooth each segment  (rejects hard GPS spikes via
       innovation gating, then globally smooths with the RTS backward pass)
    3. Remove pauses  (contiguous slow stretches > pause_min_sec)

    Parameters
    ----------
    path : str
        Path to the .gpx file.
    gap_sec : float
        Time gap (s) that starts a new segment.  Default 120.
    segment_start_trim : int
        Number of points to drop from the start of each segment.  GPS fixes
        immediately after a break are often wildly inaccurate while the
        receiver re-acquires satellites.  Default 60.
    sigma_gps : float
        GPS measurement noise (m).  Default 15.
    sigma_a_base : float
        Base process noise (m/s²) — floor for acceleration allowed.  Default 0.8.
    sigma_a_sensitivity : float
        How strongly terrain gradient inflates sigma_a.  0 = flat/constant.
        Default 2.0.
    gate_alpha : float
        Tail probability for the chi-squared innovation gate (2 DOF).  Default 0.001 (≈ 99.9 %).
    pause_speed_ms : float
        Speed (m/s) below which a point is considered paused.  Default 0.8.
    pause_window_sec : float
        Full width of centred window for local speed estimation.  Default 6.
    pause_min_sec : float
        Minimum pause duration (s) before a slow run is stripped.  Default 8.
    """

    def __init__(
        self,
        path: str,
        gap_sec: float = 120.0,
        segment_start_trim: int = 60,
        sigma_gps: float = 10.0,
        sigma_a_base: float = 0.8,
        sigma_a_sensitivity: float = 1.2,
        gate_alpha: float = 0.01,
        max_jump_factor: Optional[float] = 3.0,
        pause_speed_ms: float = 0.8,
        pause_window_sec: float = 6.0,
        pause_min_sec: float = 8.0,
    ) -> None:
        self.path = path
        self._params = dict(
            gap_sec=gap_sec,
            segment_start_trim=segment_start_trim,
            sigma_gps=sigma_gps,
            sigma_a_base=sigma_a_base,
            sigma_a_sensitivity=sigma_a_sensitivity,
            gate_alpha=gate_alpha,
            max_jump_factor=max_jump_factor,
            pause_speed_ms=pause_speed_ms,
            pause_window_sec=pause_window_sec,
            pause_min_sec=pause_min_sec,
        )

        self.raw: List[Point] = parse_gpx(path)
        self._run_pipeline()

    @classmethod
    def from_points(
        cls,
        points: List[Point],
        gap_sec: float = 120.0,
        segment_start_trim: int = 60,
        sigma_gps: float = 10.0,
        sigma_a_base: float = 0.8,
        sigma_a_sensitivity: float = 1.2,
        gate_alpha: float = 0.01,
        max_jump_factor: Optional[float] = 3.0,
        pause_speed_ms: float = 0.8,
        pause_window_sec: float = 6.0,
        pause_min_sec: float = 8.0,
    ) -> "GPXTrack":
        """Create a GPXTrack directly from a list of Points (no GPX file needed)."""
        obj = object.__new__(cls)
        obj.path = "<stream>"
        obj._params = dict(
            gap_sec=gap_sec,
            segment_start_trim=segment_start_trim,
            sigma_gps=sigma_gps,
            sigma_a_base=sigma_a_base,
            sigma_a_sensitivity=sigma_a_sensitivity,
            gate_alpha=gate_alpha,
            max_jump_factor=max_jump_factor,
            pause_speed_ms=pause_speed_ms,
            pause_window_sec=pause_window_sec,
            pause_min_sec=pause_min_sec,
        )
        obj.raw = points
        obj._run_pipeline()
        return obj

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def _run_pipeline(self) -> None:
        p = self._params

        raw_segments = split_segments(self.raw, p["gap_sec"])

        self._segments: List[List[Point]] = []
        self._n_gated = 0
        self._n_pause_pts = 0

        trim = p["segment_start_trim"]
        kalman_kwargs = dict(
            sigma_gps=p["sigma_gps"],
            sigma_a_base=p["sigma_a_base"],
            sigma_a_sensitivity=p["sigma_a_sensitivity"],
            gate_alpha=p["gate_alpha"],
        )
        for seg in raw_segments:
            seg = seg[trim:]   # drop noisy GPS fixes at segment start
            if len(seg) < 2:
                continue

            if p["max_jump_factor"] is not None:
                n_before = len(seg)
                seg = _adaptive_speed_gate(seg, max_jump_factor=p["max_jump_factor"])
                self._n_gated += n_before - len(seg)
                if len(seg) < 2:
                    continue

            smoothed, n_gated = kalman_smooth(seg, **kalman_kwargs)
            self._n_gated += n_gated

            cleaned, n_pause = remove_pauses(
                smoothed,
                speed_ms=p["pause_speed_ms"],
                window_sec=p["pause_window_sec"],
                min_pause_sec=p["pause_min_sec"],
            )
            self._n_pause_pts += n_pause
            if len(cleaned) >= 2:
                self._segments.append(cleaned)

    # ── Core data ─────────────────────────────────────────────────────────────

    @property
    def n_gated(self) -> int:
        """Total GPS points rejected by the Kalman innovation gate across all segments."""
        return self._n_gated

    @property
    def segments(self) -> List[List[Point]]:
        """Filtered track segments — each is a list of Points."""
        return self._segments

    @property
    def points(self) -> List[Point]:
        """All filtered points as a single flat list."""
        return [pt for seg in self._segments for pt in seg]

    # ── Statistics ────────────────────────────────────────────────────────────

    @property
    def stats(self) -> List[Dict]:
        """
        Per-segment statistics as a list of dicts.

        Keys: segment, start, end, points, distance_km, duration_min,
              avg_kmh, max_kmh, ele_gain_m, ele_loss_m
        """
        return [_segment_stats(seg, i) for i, seg in enumerate(self._segments)]

    @property
    def summary(self) -> Dict:
        """Totals across all segments."""
        s = self.stats
        if not s:
            return {}
        return {
            "segments":     len(s),
            "points":       sum(x["points"] for x in s),
            "distance_km":  sum(x["distance_km"] for x in s),
            "duration_min": sum(x["duration_min"] for x in s),
            "ele_gain_m":   sum(x["ele_gain_m"] for x in s),
            "ele_loss_m":   sum(x["ele_loss_m"] for x in s),
            "start":        s[0]["start"],
            "end":          s[-1]["end"],
        }

    # ── DataFrame helpers ─────────────────────────────────────────────────────

    def to_dataframe(self):
        """
        Return a pandas DataFrame with one row per filtered point.

        Columns: segment, lat, lon, ele, time, speed_ms, cum_dist_m
        Requires pandas to be installed.
        """
        import pandas as pd  # type: ignore

        rows = []
        for seg_id, seg in enumerate(self._segments):
            cum_dist = 0.0
            for i, pt in enumerate(seg):
                d = haversine(pt, seg[i + 1]) if i < len(seg) - 1 else 0.0
                if pt.speed_ms is not None:
                    # Prefer the Kalman-smoothed velocity state
                    speed = pt.speed_ms
                else:
                    # Fallback: finite difference from positions
                    t = (seg[i + 1].time - pt.time).total_seconds() if i < len(seg) - 1 else 0.0
                    speed = d / t if t > 0 else 0.0
                rows.append({
                    "segment":    seg_id,
                    "lat":        pt.lat,
                    "lon":        pt.lon,
                    "ele":        pt.ele,
                    "time":       pt.time,
                    "speed_ms":   speed,
                    "cum_dist_m": cum_dist,
                })
                cum_dist += d

        return pd.DataFrame(rows)

    def stats_dataframe(self):
        """
        Return a pandas DataFrame with one row per segment.
        Requires pandas to be installed.
        """
        import pandas as pd  # type: ignore

        return pd.DataFrame(self.stats)

    # ── Reporting ─────────────────────────────────────────────────────────────

    def print_report(self) -> None:
        """Print a formatted stats table plus a pipeline summary."""
        p = self._params
        print(f"\nFile  : {self.path}")
        print(f"Raw   : {len(self.raw)} pts  →  Kalman gated: {self._n_gated}"
              f"  →  pause pts removed: {self._n_pause_pts}")
        print(f"Result: {len(self._segments)} segment(s), "
              f"{sum(len(s) for s in self._segments)} pts")

        sep = "─" * 70
        print(f"\n{sep}")
        print(f"{'#':>2}  {'Start':>19}  {'End':>19}  {'km':>6}  "
              f"{'min':>5}  {'avg':>6}  {'max':>6}  {'↑m':>5}  {'↓m':>5}")
        print(sep)

        tot_km = tot_min = tot_gain = tot_loss = 0.0
        for s in self.stats:
            print(
                f"{s['segment']:>2}  {str(s['start']):>19}  {str(s['end']):>19}"
                f"  {s['distance_km']:>6.2f}  {s['duration_min']:>5.1f}"
                f"  {s['avg_kmh']:>5.1f}k  {s['max_kmh']:>5.1f}k"
                f"  {s['ele_gain_m']:>5.0f}  {s['ele_loss_m']:>5.0f}"
            )
            tot_km   += s["distance_km"]
            tot_min  += s["duration_min"]
            tot_gain += s["ele_gain_m"]
            tot_loss += s["ele_loss_m"]

        print(sep)
        print(f"{'':>2}  {'TOTAL':>19}  {'':>19}"
              f"  {tot_km:>6.2f}  {tot_min:>5.1f}"
              f"  {'':>6}  {'':>6}"
              f"  {tot_gain:>5.0f}  {tot_loss:>5.0f}")
        print(sep)

    def __repr__(self) -> str:
        sm = self.summary
        return (
            f"GPXTrack({self.path!r}, "
            f"segments={sm.get('segments', 0)}, "
            f"distance_km={sm.get('distance_km', 0):.2f}, "
            f"duration_min={sm.get('duration_min', 0):.1f})"
        )


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import glob as _glob

    paths = sys.argv[1:] or _glob.glob("*.gpx")
    if not paths:
        print("Usage: python gpx_filter.py [file.gpx ...]")
        sys.exit(1)

    for p in paths:
        GPXTrack(p).print_report()
