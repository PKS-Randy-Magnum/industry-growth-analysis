"""Grouped sector / subsector selection for the Streamlit sidebar."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analysis.industry_filters import load_config, sector_subsector_split, trust_funds_line_ids
from src.dashboard.data import load_profile_data


def _meta_panel(profile: str) -> pd.DataFrame:
    raw = load_profile_data(profile)
    bea_q = raw.get("bea_quarterly", pd.DataFrame())
    if bea_q.empty or "line_id" not in bea_q.columns:
        return pd.DataFrame()
    cols = [c for c in ["line_id", "industry_name", "indent_level", "is_private"] if c in bea_q.columns]
    return bea_q[cols].drop_duplicates("line_id")


def sector_subsector_options(profile: str) -> tuple[list[str], list[str]]:
    cfg = load_config()
    sector_names = list(cfg["sector_plot"])
    meta = _meta_panel(profile)
    if meta.empty:
        return sector_names, []

    _, sub_df = sector_subsector_split(meta, cfg)
    subsectors = sorted(sub_df["industry_name"].dropna().unique().tolist())

    if profile == "full":
        tf_ids = set(trust_funds_line_ids(cfg))
        tf_rows = meta[meta["line_id"].isin(tf_ids)]
        for name in tf_rows["industry_name"].dropna().unique():
            if name not in subsectors and name not in sector_names:
                subsectors.append(name)
        subsectors = sorted(subsectors)

    return sector_names, subsectors


def render_sector_picker(profile: str, *, default_sectors: list[str] | None = None) -> list[str]:
    sector_opts, subsector_opts = sector_subsector_options(profile)
    default_sectors = default_sectors or list(sector_opts)

    st.subheader("Sectors")
    c1, c2 = st.columns(2)
    sector_state = f"sector_pick_v2_{profile}"
    sub_state = f"subsector_pick_{profile}"

    if sector_state not in st.session_state:
        st.session_state[sector_state] = [s for s in default_sectors if s in sector_opts]
    if sub_state not in st.session_state:
        st.session_state[sub_state] = []

    with c1:
        if st.button("All sectors", key="all_sectors", use_container_width=True):
            st.session_state[sector_state] = list(sector_opts)
            st.rerun()
    with c2:
        if st.button("Clear sectors", key="clear_sectors", use_container_width=True):
            st.session_state[sector_state] = []
            st.rerun()

    picked_sectors = st.multiselect(
        "Top-level sectors",
        sector_opts,
        key=sector_state,
        label_visibility="collapsed",
    )

    st.subheader("Subsectors")
    c3, c4 = st.columns(2)
    with c3:
        if st.button("All subsectors", key="all_subsectors", use_container_width=True):
            st.session_state[sub_state] = list(subsector_opts)
            st.rerun()
    with c4:
        if st.button("Clear subsectors", key="clear_subsectors", use_container_width=True):
            st.session_state[sub_state] = []
            st.rerun()

    picked_subsectors = st.multiselect(
        "Industry subsectors",
        subsector_opts,
        key=sub_state,
        label_visibility="collapsed",
    )

    return list(dict.fromkeys(picked_sectors + picked_subsectors))
