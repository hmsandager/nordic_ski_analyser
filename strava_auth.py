"""
Strava OAuth 2.0 helpers for Streamlit.

Flow
────
1. Direct the user to auth_url() — opens Strava's consent page.
2. Strava redirects back to your app with ?code=XXX in the URL.
3. Call exchange_code() to swap the code for access + refresh tokens.
4. On every page load, call ensure_valid_token() — it silently refreshes
   the access token when it is within 60 s of expiry.

Tokens are stored in st.session_state:
    access_token   str
    refresh_token  str
    expires_at     int   (Unix timestamp)
    athlete        dict  (basic Strava athlete info)
"""
from __future__ import annotations

import time
from typing import Optional
from urllib.parse import urlencode

import requests
import streamlit as st

_AUTH_URL  = "https://www.strava.com/oauth/authorize"
_TOKEN_URL = "https://www.strava.com/oauth/token"
_SCOPE     = "read,activity:read_all"


def auth_url(client_id: str, redirect_uri: str) -> str:
    """Return the Strava OAuth consent-page URL."""
    params = {
        "client_id":       client_id,
        "redirect_uri":    redirect_uri,
        "response_type":   "code",
        "scope":           _SCOPE,
        "approval_prompt": "auto",
    }
    return f"{_AUTH_URL}?{urlencode(params)}"


def exchange_code(client_id: str, client_secret: str, code: str) -> dict:
    """Exchange a one-time auth code for access + refresh tokens."""
    resp = requests.post(_TOKEN_URL, data={
        "client_id":     client_id,
        "client_secret": client_secret,
        "code":          code,
        "grant_type":    "authorization_code",
    }, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _refresh(client_id: str, client_secret: str) -> None:
    """Silently refresh the access token using the stored refresh token."""
    resp = requests.post(_TOKEN_URL, data={
        "client_id":     client_id,
        "client_secret": client_secret,
        "refresh_token": st.session_state.refresh_token,
        "grant_type":    "refresh_token",
    }, timeout=10)
    resp.raise_for_status()
    tokens = resp.json()
    st.session_state.access_token  = tokens["access_token"]
    st.session_state.refresh_token = tokens["refresh_token"]
    st.session_state.expires_at    = tokens["expires_at"]


def ensure_valid_token(client_id: str, client_secret: str) -> Optional[str]:
    """
    Return a valid access token, refreshing silently if necessary.
    Returns None if the user has not authenticated yet.
    """
    if "access_token" not in st.session_state:
        return None
    if time.time() >= st.session_state.expires_at - 60:
        _refresh(client_id, client_secret)
    return st.session_state.access_token


def store_tokens(token_response: dict) -> None:
    """Write a token-exchange or refresh response into session state."""
    st.session_state.access_token  = token_response["access_token"]
    st.session_state.refresh_token = token_response["refresh_token"]
    st.session_state.expires_at    = token_response["expires_at"]
    st.session_state.athlete       = token_response.get("athlete", {})


def logout() -> None:
    for key in ("access_token", "refresh_token", "expires_at", "athlete"):
        st.session_state.pop(key, None)
