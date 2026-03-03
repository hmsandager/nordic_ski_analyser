"""
Strava API data fetching.

Key functions
─────────────
    list_activities(token)              → list of activity dicts (NordicSki only)
    fetch_points(token, activity)       → List[Point] ready for GPXTrack.from_points()
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

import requests

from gpx_filter import Point

_API = "https://www.strava.com/api/v3"


def _headers(access_token: str) -> dict:
    return {"Authorization": f"Bearer {access_token}"}


def list_activities(access_token: str, max_pages: int = 10) -> List[dict]:
    """
    Return all NordicSki activities for the authenticated athlete, newest first.
    Paginates automatically up to max_pages * 50 activities.
    """
    activities = []
    for page in range(1, max_pages + 1):
        resp = requests.get(
            f"{_API}/athlete/activities",
            headers=_headers(access_token),
            params={"per_page": 50, "page": page},
            timeout=10,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        for act in batch:
            # Strava uses both 'type' (legacy) and 'sport_type' (current)
            if act.get("sport_type") == "NordicSki" or act.get("type") == "NordicSki":
                activities.append(act)
        if len(batch) < 50:
            break

    return activities


def fetch_points(access_token: str, activity: dict) -> List[Point]:
    """
    Fetch latlng / time / altitude streams for one activity and return
    a list of Points suitable for GPXTrack.from_points().

    The Strava 'time' stream gives elapsed seconds from activity start.
    We reconstruct absolute UTC timestamps from start_date.
    """
    activity_id = activity["id"]
    start_time  = datetime.strptime(
        activity["start_date"], "%Y-%m-%dT%H:%M:%SZ"
    ).replace(tzinfo=timezone.utc)

    resp = requests.get(
        f"{_API}/activities/{activity_id}/streams",
        headers=_headers(access_token),
        params={"keys": "latlng,time,altitude", "key_by_type": "true"},
        timeout=15,
    )
    resp.raise_for_status()
    streams = resp.json()

    latlng    = streams.get("latlng",    {}).get("data", [])
    times     = streams.get("time",      {}).get("data", [])
    altitudes = streams.get("altitude",  {}).get("data", [])

    points: List[Point] = []
    for i, (ll, t) in enumerate(zip(latlng, times)):
        ele = float(altitudes[i]) if i < len(altitudes) else None
        points.append(Point(
            lat=float(ll[0]),
            lon=float(ll[1]),
            ele=ele,
            time=start_time + timedelta(seconds=int(t)),
        ))

    return points


def activity_label(act: dict) -> str:
    """Human-readable label for a selectbox."""
    date     = act["start_date_local"][:10]
    name     = act["name"]
    dist_km  = act["distance"] / 1000
    gain     = act.get("total_elevation_gain", 0)
    duration = act["moving_time"] // 60
    return f"{date}  ·  {name}  ·  {dist_km:.1f} km  ·  {duration} min  ·  ↑{gain:.0f} m"
