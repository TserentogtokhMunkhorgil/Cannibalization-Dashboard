"""Interactive dashboard for saved regional CU cannibalization results."""



from __future__ import annotations



import math
import numpy as np
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import t as student_t



from cannibalization_simulator import (

    DISTANCE_BANDS,

    CapacityResult,

    CoefficientEstimate,

    ScenarioResult,

    maximum_openings,

    optimize_openings,

    predict_apsd,

    regional_sales_estimates,

    simulate_mixed_scenario,

    validate_sales_inputs,

)





BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "outputs" / "cannibalization_regional"

INCOME_FILE = BASE_DIR / "store_master_data.xlsx"

DISTANCE_FILE = BASE_DIR / "store_to_store_distance.xlsx"

REGIONS = ("City Center", "City Outskirts", "Far / Outer Areas", "Rural")

DISTANCE_LABELS = {

    "0_100": "Very close (0-100 m)",

    "100_250": "Close (100-250 m)",

    "250_500": "Medium (250-500 m)",

    "500_1000": "Farther nearby (500-1,000 m)",

}

EVENT_ORDER = [

    "−12 to −7",

    "−6 to −3",

    "−2",

    "0",

    "+1 to +3",

    "+4 to +6",

    "+7 to +12",

    "+13 to +18",

    "+19 to +24",

    "+25 months onward",

]





@st.cache_data(show_spinner=False)

def read_csv(name: str) -> pd.DataFrame:

    path = OUTPUT_DIR / name

    if not path.is_file():

        return pd.DataFrame()

    return pd.read_csv(path)





@st.cache_data(show_spinner=False)

def read_panel() -> pd.DataFrame:

    path = OUTPUT_DIR / "cu_regional_cannibalization_panel.parquet"

    if not path.is_file():

        return pd.DataFrame()

    frame = pd.read_parquet(path)

    if "SalesMonth" in frame:

        frame["SalesMonth"] = pd.to_datetime(frame["SalesMonth"], errors="coerce")

    return frame





@st.cache_data(show_spinner=False)

def read_pair_data() -> pd.DataFrame:

    if not DISTANCE_FILE.is_file():

        return pd.DataFrame()

    pairs = pd.read_excel(DISTANCE_FILE, sheet_name="Sheet1")

    required = {

        "existing_store_id",

        "new_store_id",

        "distance_m",

        "new_opening_date",

    }

    if not required.issubset(pairs.columns):

        return pd.DataFrame()

    pairs["new_opening_date"] = pd.to_datetime(

        pairs["new_opening_date"], errors="coerce"

    )

    pairs["distance_m"] = pd.to_numeric(pairs["distance_m"], errors="coerce")
    return pairs


@st.cache_data(show_spinner=False)
def read_store_master() -> pd.DataFrame:
    if not INCOME_FILE.is_file():
        return pd.DataFrame()
    master = pd.read_excel(INCOME_FILE, sheet_name="Sheet1")
    columns = [
        column
        for column in (
            "store_id",
            "store_name",
            "ost_dt",
            "cls_dt",
            "Store_Status",
        )
        if column in master.columns
    ]
    return master[columns].drop_duplicates().copy()


def normalize_store_id(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def _parse_dashboard_dates(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    return parsed.mask(parsed.dt.year.ge(9000))


def _month_index(values: pd.Series) -> pd.Series:
    return values.dt.year * 12 + values.dt.month


def _clustered_mean_interval(
    values: pd.Series, clusters: pd.Series
) -> tuple[float, float, float]:
    valid = values.notna() & clusters.notna()
    sample = pd.DataFrame(
        {"value": values.loc[valid].astype(float), "cluster": clusters.loc[valid]}
    )
    count = len(sample)
    cluster_count = sample["cluster"].nunique()
    mean = float(sample["value"].mean()) if count else float("nan")
    if count < 2 or cluster_count < 2:
        return mean, float("nan"), float("nan")

    centered = sample["value"] - mean
    cluster_scores = centered.groupby(sample["cluster"], sort=False).sum()
    variance = (
        cluster_count
        / (cluster_count - 1)
        * count
        / (count - 1)
        * float((cluster_scores**2).sum())
        / count**2
    )
    standard_error = math.sqrt(max(variance, 0.0))
    critical_value = float(student_t.ppf(0.975, cluster_count - 1))
    return mean, mean - critical_value * standard_error, mean + critical_value * standard_error


@st.cache_data(show_spinner="Preparing pair-level opening and event-month data...")
def prepare_dashboard_openings(
    panel: pd.DataFrame,
    pair_source: pd.DataFrame,
    master_source: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[str]]:
    required_pairs = {
        "existing_store_id",
        "new_store_id",
        "distance_m",
        "new_opening_date",
        "distance_band",
    }
    missing = required_pairs.difference(pair_source.columns)
    if missing:
        raise ValueError(f"Distance workbook is missing columns: {sorted(missing)}")
    required_panel = {
        "existing_store_id",
        "SalesMonth",
        "APSD_income",
        "APSD_customers",
        "ost_dt",
        "analysis_region",
    }
    missing = required_panel.difference(panel.columns)
    if missing:
        raise ValueError(f"Saved analysis panel is missing columns: {sorted(missing)}")

    source = pair_source.copy()
    source["_existing_key"] = normalize_store_id(source["existing_store_id"])
    source["_new_key"] = normalize_store_id(source["new_store_id"])
    source["new_opening_date"] = _parse_dashboard_dates(
        source["new_opening_date"]
    )
    source["_opening_key"] = (
        source["_new_key"]
        + "::"
        + source["new_opening_date"].dt.strftime("%Y-%m-%d").fillna("Unknown")
    )
    pair_key = ["_existing_key", "_opening_key"]
    duplicate_pair_rows = int(source.duplicated(pair_key, keep=False).sum())
    if duplicate_pair_rows:
        raise ValueError(
            f"Distance workbook has {duplicate_pair_rows} duplicate existing-store/"
            "new-store-opening rows (store IDs plus opening date); resolve duplicates "
            "before calculating opening impacts."
        )
    if "existing_opening_date" in source:
        source["existing_opening_date"] = _parse_dashboard_dates(
            source["existing_opening_date"]
        )
    else:
        source["existing_opening_date"] = pd.NaT
    source["distance_m"] = pd.to_numeric(source["distance_m"], errors="coerce")
    source["distance_band"] = source["distance_band"].astype("string").str.strip()
    source["distance_band"] = source["distance_band"].mask(
        source["distance_band"].eq(""), pd.NA
    )

    panel_metadata = panel.copy()
    panel_metadata["_existing_key"] = normalize_store_id(
        panel_metadata["existing_store_id"]
    )
    panel_metadata["ost_dt"] = _parse_dashboard_dates(panel_metadata["ost_dt"])
    metadata_columns = [
        "_existing_key",
        "ost_dt",
        "analysis_region",
    ]
    metadata = (
        panel_metadata.sort_values("SalesMonth")
        .drop_duplicates("_existing_key")[metadata_columns]
        .rename(
            columns={
                "ost_dt": "master_existing_opening_date",
                "analysis_region": "existing_store_region",
            }
        )
    )
    if not master_source.empty and {"store_id", "store_name"}.issubset(
        master_source.columns
    ):
        names = master_source[["store_id", "store_name"]].copy()
        names["_existing_key"] = normalize_store_id(names["store_id"])
        names["store_name"] = names["store_name"].astype("string").str.strip()
        names = names.dropna(subset=["store_name"]).drop_duplicates(
            "_existing_key"
        )
        metadata = metadata.merge(
            names[["_existing_key", "store_name"]],
            on="_existing_key",
            how="left",
            validate="one_to_one",
        )
    source = source.merge(
        metadata,
        on="_existing_key",
        how="left",
        validate="many_to_one",
    )
    source["existing_opening_date_used"] = source[
        "master_existing_opening_date"
    ].fillna(source["existing_opening_date"])
    source["new_store_opening_month"] = (
        source["new_opening_date"].dt.to_period("M").dt.to_timestamp()
    )
    source["opening_month_index"] = _month_index(
        source["new_store_opening_month"]
    )
    source["existing_opening_month"] = (
        source["existing_opening_date_used"].dt.to_period("M").dt.to_timestamp()
    )
    date_mismatch = (
        source["master_existing_opening_date"].notna()
        & source["existing_opening_date"].notna()
        & (
            source["master_existing_opening_date"].dt.normalize()
            != source["existing_opening_date"].dt.normalize()
        )
    )
    has_dates = (
        source["new_opening_date"].notna()
        & source["existing_opening_date_used"].notna()
    )
    preexisting = (
        has_dates
        & (source["existing_opening_date_used"] < source["new_opening_date"])
    )
    valid_distance = source["distance_m"].notna() & source["distance_m"].ge(0)
    source["distance_band"] = source["distance_band"].fillna("Unclassified")
    eligible = source.loc[has_dates & preexisting & valid_distance].copy()
    eligible["event_id"] = (
        eligible["_existing_key"].astype(str)
        + "::"
        + eligible["_opening_key"].astype(str)
    )
    eligible["new_store_name"] = (
        eligible.get("new_store_name", pd.Series(index=eligible.index, dtype="string"))
        .astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    )
    eligible["event_order"] = (
        eligible.sort_values(
            ["_existing_key", "new_opening_date", "_new_key"],
            kind="stable",
        )
        .groupby("_existing_key", sort=False)
        .cumcount()
        .add(1)
        .reindex(eligible.index)
        .astype("int64")
    )
    eligible["exposure_order"] = eligible["event_order"].map(
        lambda order: (
            "First observed opening event"
            if order == 1
            else "Second observed opening event"
            if order == 2
            else "Third or later observed opening event"
        )
    )
    eligible["previous_nearby_openings_in_pair_history"] = (
        eligible["event_order"] - 1
    )
    eligible["region"] = eligible["existing_store_region"].fillna("Unmapped")

    band_minimums = (
        source.loc[source["distance_m"].notna()]
        .groupby("distance_band", dropna=False)["distance_m"]
        .min()
        .sort_values(kind="stable")
    )
    band_order = band_minimums.index.astype(str).tolist()
    if "Unclassified" in source["distance_band"].astype(str).unique():
        band_order.append("Unclassified")

    month_data = panel.copy()
    month_data["_existing_key"] = normalize_store_id(
        month_data["existing_store_id"]
    )
    month_data["SalesMonth"] = _parse_dashboard_dates(month_data["SalesMonth"])
    month_data["SalesMonth"] = (
        month_data["SalesMonth"].dt.to_period("M").dt.to_timestamp()
    )
    if month_data.duplicated(["_existing_key", "SalesMonth"]).any():
        raise ValueError(
            "Saved analysis panel has duplicate existing-store/month keys; "
            "opening-event statistics are not calculated."
        )
    month_data["_calendar_month_index"] = _month_index(month_data["SalesMonth"])
    keep = [
        "_existing_key",
        "SalesMonth",
        "_calendar_month_index",
        "APSD_income",
        "APSD_customers",
    ]
    for optional in ("Active_Days", "operating_month_known"):
        if optional in month_data:
            keep.append(optional)
    month_data = month_data[keep]
    for outcome_column in ("APSD_income", "APSD_customers"):
        month_data[outcome_column] = pd.to_numeric(
            month_data[outcome_column], errors="coerce"
        )
        if "operating_month_known" in month_data:
            month_data.loc[
                ~month_data["operating_month_known"].fillna(False),
                outcome_column,
            ] = pd.NA
    event_columns = [
        "event_id",
        "_existing_key",
        "_new_key",
        "_opening_key",
        "existing_store_id",
        "new_store_id",
        "new_store_name",
        "new_opening_date",
        "new_store_opening_month",
        "opening_month_index",
        "distance_m",
        "distance_band",
        "event_order",
        "exposure_order",
        "previous_nearby_openings_in_pair_history",
        "region",
        "existing_opening_date_used",
    ]
    events = eligible[event_columns].copy()
    observations = month_data.merge(
        events.drop(columns="existing_store_id"),
        on="_existing_key",
        how="inner",
        validate="many_to_many",
    )
    observations["event_time_month"] = (
        observations["_calendar_month_index"] - observations["opening_month_index"]
    )
    diagnostics = {
        "source_pair_rows": int(len(source)),
        "duplicate_pair_rows": duplicate_pair_rows,
        "missing_new_opening_date": int(source["new_opening_date"].isna().sum()),
        "missing_existing_opening_date_after_fallback": int(
            source["existing_opening_date_used"].isna().sum()
        ),
        "invalid_distance_rows": int((~valid_distance).sum()),
        "missing_distance_band_rows": int(
            source["distance_band"].eq("Unclassified").sum()
        ),
        "opening_date_mismatches": int(date_mismatch.sum()),
        "pairs_not_preexisting": int((has_dates & ~preexisting).sum()),
        "eligible_opening_events": int(len(events)),
        "unique_new_store_openings": int(events["_opening_key"].nunique()),
        "distinct_existing_stores_exposed": int(events["_existing_key"].nunique()),
        "panel_start_month": str(month_data["SalesMonth"].min().date()),
        "panel_end_month": str(month_data["SalesMonth"].max().date()),
        "events_outside_observed_panel_dates": int(
            (~events["new_store_opening_month"].between(
                month_data["SalesMonth"].min(), month_data["SalesMonth"].max()
            )).sum()
        ),
    }
    return events, observations, diagnostics, band_order


def _filter_events_by_dates(
    events: pd.DataFrame, widget_key: str
) -> tuple[pd.DataFrame, tuple[Any, Any]]:
    if events.empty:
        return events.copy(), (None, None)
    start = events["new_opening_date"].min().date()
    end = events["new_opening_date"].max().date()
    selected = st.date_input(
        "New-store opening date range",
        value=(start, end),
        min_value=start,
        max_value=end,
        key=widget_key,
    )
    if not isinstance(selected, (tuple, list)) or len(selected) != 2:
        return events.iloc[0:0].copy(), (None, None)
    lower, upper = selected
    if lower is None or upper is None:
        return events.iloc[0:0].copy(), (lower, upper)
    mask = events["new_opening_date"].dt.date.between(lower, upper)
    return events.loc[mask].copy(), (lower, upper)


def build_opening_event_comparisons(
    events: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    before_months: int,
    after_months: int,
    include_event_month: bool,
) -> pd.DataFrame:
    """Create paired opening-event means; event month 0 is separate by default."""
    if events.empty:
        return pd.DataFrame()
    end_post = after_months
    start_post = 0 if include_event_month else 1
    keep_ids = set(events["event_id"])
    relevant = observations.loc[
        observations["event_id"].isin(keep_ids)
        & observations["event_time_month"].between(-before_months, end_post)
    ]
    results: list[pd.DataFrame] = []
    for outcome in ("APSD_income", "APSD_customers"):
        pre = (
            relevant.loc[relevant["event_time_month"].between(-before_months, -1)]
            .groupby("event_id", as_index=False)
            .agg(
                pre_mean=(outcome, "mean"),
                pre_observed_months=(outcome, "count"),
            )
        )
        post = (
            relevant.loc[
                relevant["event_time_month"].between(start_post, end_post)
            ]
            .groupby("event_id", as_index=False)
            .agg(
                post_mean=(outcome, "mean"),
                post_observed_months=(outcome, "count"),
            )
        )
        paired = (
            events.merge(pre, on="event_id", how="inner", validate="one_to_one")
            .merge(post, on="event_id", how="inner", validate="one_to_one")
        )
        paired = paired.loc[
            paired["pre_mean"].notna() & paired["post_mean"].notna()
        ].copy()
        paired["outcome"] = outcome
        paired["absolute_change"] = paired["post_mean"] - paired["pre_mean"]
        paired["percentage_change"] = pd.NA
        valid_baseline = paired["pre_mean"] > 0
        paired.loc[valid_baseline, "percentage_change"] = (
            paired.loc[valid_baseline, "post_mean"]
            / paired.loc[valid_baseline, "pre_mean"]
            - 1
        ) * 100
        results.append(paired)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def summarize_opening_changes(
    changes: pd.DataFrame,
    outcome: str,
    *,
    minimum_events: int,
    minimum_stores: int,
) -> pd.DataFrame:
    """Summarize event-weighted paired changes with existing-store clustered CIs."""
    if changes.empty:
        return pd.DataFrame()
    selected = changes.loc[changes["outcome"] == outcome].copy()
    groups: list[tuple[str, str | None, str | None, str | None, pd.DataFrame]] = [
        ("Overall", None, None, None, selected)
    ]
    for region, rows in selected.groupby("region", dropna=False):
        groups.append(("Region", str(region), None, None, rows))
    for band, rows in selected.groupby("distance_band", dropna=False):
        groups.append(("Distance band", None, str(band), None, rows))
    for (region, band), rows in selected.groupby(
        ["region", "distance_band"], dropna=False
    ):
        groups.append(("Region × distance band", str(region), str(band), None, rows))
    for order, rows in selected.groupby("exposure_order", dropna=False):
        groups.append(("Exposure order", None, None, str(order), rows))
    for (order, region), rows in selected.groupby(
        ["exposure_order", "region"], dropna=False
    ):
        groups.append(
            ("Exposure order × region", str(region), None, str(order), rows)
        )
    for (order, band), rows in selected.groupby(
        ["exposure_order", "distance_band"], dropna=False
    ):
        groups.append(
            ("Exposure order × distance band", None, str(band), str(order), rows)
        )
    for (order, region, band), rows in selected.groupby(
        ["exposure_order", "region", "distance_band"], dropna=False
    ):
        groups.append(
            (
                "Exposure order × region × distance band",
                str(region),
                str(band),
                str(order),
                rows,
            )
        )

    summary_rows: list[dict[str, Any]] = []
    for group_type, region, band, exposure_order, rows in groups:
        deltas = pd.to_numeric(rows["absolute_change"], errors="coerce")
        valid_delta = deltas.notna()
        rows = rows.loc[valid_delta].copy()
        deltas = deltas.loc[valid_delta]
        event_count = int(rows["event_id"].nunique())
        store_count = int(rows["_existing_key"].nunique())
        mean_delta, ci_lower, ci_upper = _clustered_mean_interval(
            deltas.reset_index(drop=True),
            rows["_existing_key"].reset_index(drop=True),
        )
        pct_values = pd.to_numeric(rows["percentage_change"], errors="coerce")
        pct_mean, pct_lower, pct_upper = _clustered_mean_interval(
            pct_values.reset_index(drop=True),
            rows["_existing_key"].reset_index(drop=True),
        )
        decline_count = int(deltas.lt(0).sum())
        increase_count = int(deltas.gt(0).sum())
        summary_rows.append(
            {
                "group_type": group_type,
                "region": region if region is not None else "All Regions",
                "distance_band": band if band is not None else "All Bands",
                "exposure_order": exposure_order or "All Orders",
                "outcome": outcome,
                "existing_stores": store_count,
                "opening_events": event_count,
                "unique_new_store_openings": int(rows["_opening_key"].nunique()),
                "mean_before_per_opening_event": float(rows["pre_mean"].mean())
                if event_count else float("nan"),
                "mean_after_per_opening_event": float(rows["post_mean"].mean())
                if event_count else float("nan"),
                "mean_absolute_change_per_opening_event": mean_delta,
                "mean_percentage_change_per_valid_baseline_event": pct_mean,
                "median_absolute_change_per_opening_event": float(deltas.median())
                if event_count else float("nan"),
                "standard_deviation_event_change": float(deltas.std(ddof=1))
                if event_count > 1 else float("nan"),
                "clustered_standard_error_event_change": (
                    (ci_upper - ci_lower)
                    / (2 * float(student_t.ppf(0.975, store_count - 1)))
                    if store_count > 1 and pd.notna(ci_lower) and pd.notna(ci_upper)
                    else float("nan")
                ),
                "clustered_ci_95_lower_change": ci_lower,
                "clustered_ci_95_upper_change": ci_upper,
                "mean_percentage_ci_95_lower": pct_lower,
                "mean_percentage_ci_95_upper": pct_upper,
                "decline_events": decline_count,
                "decline_event_percentage": (
                    decline_count / event_count * 100 if event_count else float("nan")
                ),
                "increase_events": increase_count,
                "increase_event_percentage": (
                    increase_count / event_count * 100 if event_count else float("nan")
                ),
                "unchanged_events": int(deltas.eq(0).sum()),
                "mean_observed_pre_months_per_event": float(
                    rows["pre_observed_months"].mean()
                ) if event_count else float("nan"),
                "mean_observed_post_months_per_event": float(
                    rows["post_observed_months"].mean()
                ) if event_count else float("nan"),
                "support_status": (
                    "Supported"
                    if event_count >= minimum_events and store_count >= minimum_stores
                    else "Insufficient support"
                ),
            }
        )
    return pd.DataFrame(summary_rows)


def summarize_event_time(
    observations: pd.DataFrame,
    events: pd.DataFrame,
    outcome: str,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    selected = observations.loc[
        observations["event_id"].isin(set(events["event_id"]))
        & observations["event_time_month"].between(-12, 12)
    ]
    selected = selected.loc[selected[outcome].notna()]
    rows: list[dict[str, Any]] = []
    for event_time, sample in selected.groupby("event_time_month", sort=True):
        mean, lower, upper = _clustered_mean_interval(
            sample[outcome].reset_index(drop=True),
            sample["_existing_key"].reset_index(drop=True),
        )
        rows.append(
            {
                "event_time_month": int(str(event_time)),
                "mean_outcome": mean,
                "ci_95_lower": lower,
                "ci_95_upper": upper,
                "pair_month_observations": int(len(sample)),
                "opening_events": int(sample["event_id"].nunique()),
                "existing_stores": int(sample["_existing_key"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _event_order_label(order: object) -> str:
    if order == 1:
        return "First observed opening event"
    if order == 2:
        return "Second observed opening event"
    return "Third or later observed opening event"


def _metric_label(outcome: str) -> str:
    return (
        "APSD sales (MNT per active sales day)"
        if outcome == "APSD_income"
        else "APSD customers per active sales day"
    )


def regional_distance_statistics_page(
    panel: pd.DataFrame,
    pair_source: pd.DataFrame,
    master: pd.DataFrame,
    outcome: str,
    outcome_label: str,
) -> None:
    st.header("Regional & Distance Statistics")
    st.caption(
        "Descriptive paired changes around nearby new-store opening events. "
        "These before-and-after comparisons are not causal estimates; other "
        "openings, seasonality, promotions, and local demand changes may also affect outcomes."
    )
    if pair_source.empty:
        st.error("The pairwise distance workbook is unavailable or has no usable rows.")
        return
    try:
        all_events, observations, diagnostics, band_order = prepare_dashboard_openings(
            panel, pair_source, master
        )
    except ValueError as error:
        st.error(f"Cannot calculate opening statistics: {error}")
        return
    if all_events.empty:
        st.warning("No qualifying existing-store/new-store opening pairs were found.")
        return

    with st.expander("Data checks and exclusions"):
        st.write(
            "Pair events are unique existing-store/new-store pairs. Existing-store "
            "opening dates from the saved panel take precedence; the pair workbook "
            "date is only a fallback. Missing or invalid dates, unknown existing "
            "opening dates, distances, and events before the existing store opened "
            "are not used."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {"Check": label, "Count / value": value}
                    for label, value in (
                        ("Pair rows in source workbook", diagnostics["source_pair_rows"]),
                        ("Eligible pair opening events", diagnostics["eligible_opening_events"]),
                        ("Distinct new-store openings", diagnostics["unique_new_store_openings"]),
                        ("Existing stores exposed", diagnostics["distinct_existing_stores_exposed"]),
                        ("Missing new-opening dates", diagnostics["missing_new_opening_date"]),
                        ("Missing existing-opening dates after fallback", diagnostics["missing_existing_opening_date_after_fallback"]),
                        ("Invalid distances", diagnostics["invalid_distance_rows"]),
                        ("Unclassified distance bands", diagnostics["missing_distance_band_rows"]),
                        ("Existing-opening date discrepancies", diagnostics["opening_date_mismatches"]),
                        ("Pairs where existing store was not yet open", diagnostics["pairs_not_preexisting"]),
                        ("Opening events outside outcome-panel dates", diagnostics["events_outside_observed_panel_dates"]),
                    )
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            f"Observed outcomes cover {diagnostics['panel_start_month']} through "
            f"{diagnostics['panel_end_month']}. Event orders are based on all "
            "qualifying opening pairs available in the pair workbook, not only "
            "events within the selected date range. Therefore, labels say "
            "'first observed' rather than 'first ever'."
        )

    region_values = sorted(
        all_events["region"].dropna().astype(str).unique().tolist()
    )
    region_options = ["All Regions", *region_values]
    distance_options = ["All Bands", *band_order]
    order_options = [
        "First observed opening event",
        "Second observed opening event",
        "Third or later observed opening event",
    ]
    filter_cols = st.columns(3)
    selected_regions = filter_cols[0].multiselect(
        "Existing-store region",
        options=region_options,
        default=["All Regions"],
        key="regional_stats_regions",
    )
    selected_bands = filter_cols[1].multiselect(
        "Distance band (source workbook)",
        options=distance_options,
        default=["All Bands"],
        key="regional_stats_distance_bands",
        help="Uses the workbook's labels; bands are not re-binned.",
    )
    selected_orders = filter_cols[2].multiselect(
        "Opening order",
        options=order_options,
        default=order_options,
        key="regional_stats_opening_order",
    )
    selected_events, _ = _filter_events_by_dates(
        all_events, "regional_stats_opening_date_range"
    )
    window_cols = st.columns(4)
    before_months = window_cols[0].slider(
        "Pre-opening months",
        min_value=1,
        max_value=12,
        value=6,
        key="regional_stats_pre_window",
    )
    after_months = window_cols[1].slider(
        "Post-opening months",
        min_value=1,
        max_value=12,
        value=6,
        key="regional_stats_post_window",
    )
    include_event_month = window_cols[2].checkbox(
        "Include opening month (0) in post period",
        value=False,
        key="regional_stats_include_month_zero",
        help="Month 0 may be only partly exposed; it is separate by default.",
    )
    minimum_events = window_cols[3].number_input(
        "Minimum opening events per group",
        min_value=1,
        max_value=500,
        value=5,
        step=1,
        key="regional_stats_min_events",
    )
    minimum_stores = st.number_input(
        "Minimum distinct existing stores per group",
        min_value=1,
        max_value=250,
        value=3,
        step=1,
        key="regional_stats_min_stores",
    )

    if selected_regions and "All Regions" not in selected_regions:
        selected_events = selected_events.loc[
            selected_events["region"].isin(selected_regions)
        ]
    if selected_bands and "All Bands" not in selected_bands:
        selected_events = selected_events.loc[
            selected_events["distance_band"].isin(selected_bands)
        ]
    selected_events = selected_events.loc[
        selected_events["exposure_order"].isin(selected_orders)
    ]
    event_ids = set(selected_events["event_id"])
    filtered_observations = observations.loc[
        observations["event_id"].isin(event_ids)
    ]
    changes = build_opening_event_comparisons(
        selected_events,
        filtered_observations,
        before_months=before_months,
        after_months=after_months,
        include_event_month=include_event_month,
    )
    summary = summarize_opening_changes(
        changes,
        outcome,
        minimum_events=minimum_events,
        minimum_stores=minimum_stores,
    )

    st.subheader(f"Opening impact summary — {outcome_label}")
    all_event_count = int(selected_events["_opening_key"].nunique())
    exposed_store_count = int(selected_events["_existing_key"].nunique())
    overall = summary.loc[summary["group_type"] == "Overall"] if not summary.empty else pd.DataFrame()
    paired = overall.iloc[0] if not overall.empty else pd.Series(dtype=object)
    kpis = st.columns(4)
    kpis[0].metric("Unique new-store openings", f"{all_event_count:,}")
    kpis[1].metric("Existing stores exposed", f"{exposed_store_count:,}")
    kpis[2].metric(
        "Opening events with paired valid outcomes",
        f"{int(paired.get('opening_events', 0)):,}",
    )
    kpis[3].metric(
        "Average APSD before opening",
        f"{paired.get('mean_before_per_opening_event', float('nan')):,.2f}",
    )
    second_kpis = st.columns(4)
    second_kpis[0].metric(
        "Average APSD after opening",
        f"{paired.get('mean_after_per_opening_event', float('nan')):,.2f}",
    )
    second_kpis[1].metric(
        "Mean absolute change",
        f"{paired.get('mean_absolute_change_per_opening_event', float('nan')):+,.2f}",
    )
    second_kpis[2].metric(
        "Mean percentage change",
        f"{paired.get('mean_percentage_change_per_valid_baseline_event', float('nan')):+.2f}%",
    )
    second_kpis[3].metric(
        "Median absolute change",
        f"{paired.get('median_absolute_change_per_opening_event', float('nan')):+,.2f}",
    )
    third_kpis = st.columns(4)
    third_kpis[0].metric(
        "Opening events with a decline",
        f"{int(paired.get('decline_events', 0)):,} "
        f"({paired.get('decline_event_percentage', float('nan')):.1f}%)",
    )
    third_kpis[1].metric(
        "Opening events with an increase",
        f"{int(paired.get('increase_events', 0)):,} "
        f"({paired.get('increase_event_percentage', float('nan')):.1f}%)",
    )
    third_kpis[2].metric(
        "Mean observed pre-period months/event",
        f"{paired.get('mean_observed_pre_months_per_event', float('nan')):.1f}",
    )
    third_kpis[3].metric(
        "Mean observed post-period months/event",
        f"{paired.get('mean_observed_post_months_per_event', float('nan')):.1f}",
    )
    post_start = 0 if include_event_month else 1
    st.caption(
        f"Each paired observation is one existing-store/new-store opening event. "
        f"Pre period: months −{before_months} to −1; post period: +{post_start} "
        f"to +{after_months}. Means are averages of each event's observed monthly "
        "APSD means, not averages over all raw store-month rows. Event month 0 is "
        "excluded unless selected. Percent changes are calculated only for events "
        "with a positive pre-opening APSD baseline."
    )
    st.caption(
        "Confidence intervals in the group table use a store-clustered paired-mean "
        "standard error, allowing repeated opening events for the same existing "
        "store to be dependent. Groups below the selected support thresholds are "
        "flagged; no p-values or causal claims are reported."
    )

    st.subheader("Paired comparison by region and distance band")
    if summary.empty:
        st.info("No opening events have observed APSD values in both selected periods.")
    else:
        group_options = [
            "Overall",
            "Region",
            "Distance band",
            "Region × distance band",
            "Exposure order",
            "Exposure order × region",
            "Exposure order × distance band",
            "Exposure order × region × distance band",
        ]
        chosen_groups = st.multiselect(
            "Summary breakdowns",
            options=group_options,
            default=["Region", "Distance band", "Region × distance band"],
            key="regional_stats_summary_breakdowns",
        )
        table = summary.loc[summary["group_type"].isin(chosen_groups)].copy()
        table = table.sort_values(
            ["group_type", "region", "distance_band", "exposure_order"],
            kind="stable",
        )
        st.dataframe(table, width="stretch", hide_index=True)
        download_table(
            table,
            "Download paired summary CSV",
            f"opening_summary_{outcome}.csv",
        )
        download_table(
            changes.loc[changes["outcome"] == outcome],
            "Download event-level paired changes CSV",
            f"opening_event_changes_{outcome}.csv",
        )

    st.subheader("Distance gradient")
    distance_summary = summary.loc[
        (summary["group_type"] == "Distance band")
        & (summary["support_status"] == "Supported")
    ].copy() if not summary.empty else pd.DataFrame()
    if distance_summary.empty:
        st.info("No distance band meets the selected event and store support thresholds.")
    else:
        distance_summary["distance_band"] = pd.Categorical(
            distance_summary["distance_band"],
            categories=band_order,
            ordered=True,
        )
        gradient_cols = st.columns(2)
        for column, value, low, high, title, y_label in (
            (
                gradient_cols[0],
                "mean_absolute_change_per_opening_event",
                "clustered_ci_95_lower_change",
                "clustered_ci_95_upper_change",
                "Mean APSD change by distance",
                "Mean APSD change",
            ),
            (
                gradient_cols[1],
                "mean_percentage_change_per_valid_baseline_event",
                "mean_percentage_ci_95_lower",
                "mean_percentage_ci_95_upper",
                "Mean percentage APSD change by distance",
                "Mean percentage change (%)",
            ),
        ):
            fig = go.Figure(
                go.Bar(
                    x=distance_summary["distance_band"].astype(str),
                    y=distance_summary[value],
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": distance_summary[high] - distance_summary[value],
                        "arrayminus": distance_summary[value] - distance_summary[low],
                    },
                    customdata=distance_summary[
                        ["existing_stores", "opening_events"]
                    ].to_numpy(),
                    hovertemplate=(
                        "%{x}<br>Mean: %{y:.2f}"
                        "<br>Existing stores: %{customdata[0]}"
                        "<br>Opening events: %{customdata[1]}<extra></extra>"
                    ),
                )
            )
            fig.update_layout(
                title=title,
                xaxis_title="Source distance band",
                yaxis_title=y_label,
                xaxis={"categoryorder": "array", "categoryarray": band_order},
            )
            fig.add_hline(y=0, line_color="black", line_width=1)
            column.plotly_chart(fig, width="stretch")

    st.subheader("Region × distance heatmaps")
    heatmap_outcome = st.selectbox(
        "Heatmap outcome",
        ["Both outcomes", "APSD sales", "APSD customers"],
        index=0,
        key="regional_stats_heatmap_outcome",
    )
    map_outcomes = (
        [("APSD sales", "APSD_income"), ("APSD customers", "APSD_customers")]
        if heatmap_outcome == "Both outcomes"
        else [
            (
                heatmap_outcome,
                "APSD_income" if heatmap_outcome == "APSD sales" else "APSD_customers",
            )
        ]
    )
    heatmap_regions = (
        region_values
        if not selected_regions or "All Regions" in selected_regions
        else [region for region in selected_regions if region in region_values]
    )
    heatmap_bands = (
        band_order
        if not selected_bands or "All Bands" in selected_bands
        else [band for band in band_order if band in selected_bands]
    )
    heatmap_changes = changes
    for heatmap_label, heatmap_outcome_column in map_outcomes:
        outcome_changes = heatmap_changes.loc[
            heatmap_changes["outcome"] == heatmap_outcome_column
        ]
        if outcome_changes.empty:
            outcome_summary = pd.DataFrame()
        elif heatmap_outcome_column == outcome:
            outcome_summary = summary
        else:
            outcome_summary = summarize_opening_changes(
                heatmap_changes,
                heatmap_outcome_column,
                minimum_events=minimum_events,
                minimum_stores=minimum_stores,
            )
        cells = outcome_summary.loc[
            outcome_summary["group_type"] == "Region × distance band"
        ].copy() if not outcome_summary.empty else pd.DataFrame()
        cell_grid = pd.MultiIndex.from_product(
            [heatmap_regions, heatmap_bands], names=["region", "distance_band"]
        ).to_frame(index=False)
        if not cells.empty:
            cell_grid = cell_grid.merge(
                cells,
                on=["region", "distance_band"],
                how="left",
                validate="one_to_one",
            )
        for metric_column, title, percent in (
            (
                "mean_absolute_change_per_opening_event",
                f"{heatmap_label}: mean absolute APSD change",
                False,
            ),
            (
                "mean_percentage_change_per_valid_baseline_event",
                f"{heatmap_label}: mean percentage APSD change",
                True,
            ),
        ):
            if cell_grid.empty:
                st.info(f"No region-by-distance groups are available for {heatmap_label}.")
                continue
            supported = (
                cell_grid["support_status"].eq("Supported")
                if "support_status" in cell_grid
                else pd.Series(False, index=cell_grid.index)
            )
            z_values = cell_grid[metric_column].where(supported)
            pivot = pd.DataFrame(
                {
                    "region": cell_grid["region"],
                    "distance_band": cell_grid["distance_band"],
                    "value": z_values,
                    "opening_events": cell_grid.get(
                        "opening_events", pd.Series(0, index=cell_grid.index)
                    ).fillna(0),
                    "existing_stores": cell_grid.get(
                        "existing_stores", pd.Series(0, index=cell_grid.index)
                    ).fillna(0),
                    "support_status": cell_grid.get(
                        "support_status", pd.Series("Insufficient support", index=cell_grid.index)
                    ).fillna("Insufficient support"),
                }
            )
            z_matrix = pivot.pivot(
                index="region", columns="distance_band", values="value"
            ).reindex(index=heatmap_regions, columns=heatmap_bands)
            custom_matrix = []
            text_matrix = []
            for region_value in heatmap_regions:
                custom_row = []
                text_row = []
                for band_value in heatmap_bands:
                    cell = pivot.loc[
                        (pivot["region"] == region_value)
                        & (pivot["distance_band"] == band_value)
                    ]
                    if cell.empty:
                        event_n, store_n, status = 0, 0, "Insufficient support"
                    else:
                        item = cell.iloc[0]
                        event_n = int(item["opening_events"])
                        store_n = int(item["existing_stores"])
                        status = str(item["support_status"])
                    custom_row.append([event_n, store_n, status])
                    cell_value = z_matrix.at[region_value, band_value]
                    if pd.isna(cell_value):
                        text_row.append(f"Low support<br>n={event_n} events")
                    else:
                        text_row.append(
                            f"{cell_value:+.1f}{'%' if percent else ''}"
                            f"<br>n={event_n}"
                        )
                custom_matrix.append(custom_row)
                text_matrix.append(text_row)
            fig = go.Figure(
                go.Heatmap(
                    z=z_matrix.to_numpy(dtype=float),
                    x=heatmap_bands,
                    y=heatmap_regions,
                    text=text_matrix,
                    texttemplate="%{text}",
                    customdata=custom_matrix,
                    zmid=0,
                    colorscale="RdYlGn",
                    colorbar={"title": "Change (%)" if percent else "APSD change"},
                    hovertemplate=(
                        "Region: %{y}<br>Distance band: %{x}"
                        "<br>Change: %{z:.2f}"
                        "<br>Opening events: %{customdata[0]}"
                        "<br>Existing stores: %{customdata[1]}"
                        "<br>Status: %{customdata[2]}<extra></extra>"
                    ),
                )
            )
            fig.update_layout(
                title=title,
                xaxis_title="Source distance band (categorical)",
                yaxis_title="Existing-store region",
            )
            st.plotly_chart(fig, width="stretch")
    st.caption(
        "Heatmap cells below the selected minimum event/store support are visibly "
        "flagged rather than colored as stable estimates. Counts in each cell are "
        "unique paired opening events and existing stores."
    )

    st.subheader("Event-time trajectory")
    compare_by = st.selectbox(
        "Compare event trajectories by",
        ["Exposure order", "Region", "Distance band", "No split"],
        key="regional_stats_trajectory_split",
    )
    if compare_by == "Exposure order":
        compare_column = "exposure_order"
    elif compare_by == "Region":
        compare_column = "region"
    elif compare_by == "Distance band":
        compare_column = "distance_band"
    else:
        compare_column = ""
    trajectory_groups = (
        [(str(value), group) for value, group in selected_events.groupby(compare_column)]
        if compare_column
        else [("Selected events", selected_events)]
    )
    trajectory_rows: list[pd.DataFrame] = []
    for group_label, group_events in trajectory_groups:
        stats = summarize_event_time(filtered_observations, group_events, outcome)
        if stats.empty:
            continue
        stats = stats.set_index("event_time_month").reindex(range(-12, 13))
        stats.index.name = "event_time_month"
        stats = stats.reset_index()
        for count_column in (
            "pair_month_observations",
            "opening_events",
            "existing_stores",
        ):
            stats[count_column] = stats[count_column].fillna(0).astype(int)
        supported = (
            stats["opening_events"].ge(minimum_events)
            & stats["existing_stores"].ge(minimum_stores)
        )
        stats.loc[~supported, ["mean_outcome", "ci_95_lower", "ci_95_upper"]] = np.nan
        stats["support_status"] = np.where(
            supported, "Supported", "Insufficient support"
        )
        stats["series"] = group_label
        trajectory_rows.append(stats)
    trajectories = (
        pd.concat(trajectory_rows, ignore_index=True)
        if trajectory_rows
        else pd.DataFrame()
    )
    if trajectories.empty:
        st.info("No event-time outcome observations are available for this selection.")
    else:
        fig = go.Figure()
        for group_label, group in trajectories.groupby("series", sort=False):
            custom = group[
                ["pair_month_observations", "opening_events", "existing_stores"]
            ].to_numpy()
            fig.add_trace(
                go.Scatter(
                    x=group["event_time_month"],
                    y=group["mean_outcome"],
                    mode="lines+markers",
                    name=group_label,
                    connectgaps=False,
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": group["ci_95_upper"] - group["mean_outcome"],
                        "arrayminus": group["mean_outcome"] - group["ci_95_lower"],
                        "visible": True,
                    },
                    customdata=custom,
                    hovertemplate=(
                        "Event month %{x}<br>Mean APSD: %{y:.2f}"
                        "<br>Pair-event observations: %{customdata[0]}"
                        "<br>Unique opening events: %{customdata[1]}"
                        "<br>Existing stores: %{customdata[2]}<extra></extra>"
                    ),
                )
            )
        fig.add_vrect(
            x0=-12.5, x1=-0.5, fillcolor="#dbeafe", opacity=0.25, line_width=0
        )
        fig.add_vrect(
            x0=0.5, x1=12.5, fillcolor="#dcfce7", opacity=0.18, line_width=0
        )
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="#b91c1c",
            annotation_text="Opening month (0; may be partial)",
        )
        fig.update_layout(
            title=f"Observed mean {_metric_label(outcome)} around opening events",
            xaxis_title="Calendar months from new-store opening month",
            yaxis_title=_metric_label(outcome),
            hovermode="x unified",
            legend_title=compare_by,
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Each point averages available event-month observations; the number of "
            "pair-months, unique openings, and existing stores appears on hover. "
            "The sample composition can change across event months. Missing months "
            "remain missing, not zero. Shaded intervals separate pre-opening from "
            "post-opening months; clustered 95% intervals account for repeated "
            "observations from the same existing store."
        )
        st.dataframe(
            trajectories,
            width="stretch",
            hide_index=True,
        )

    st.subheader("Do stores with previous nearby openings respond differently?")
    order_summary = summary.loc[
        summary["group_type"] == "Exposure order"
    ].copy() if not summary.empty else pd.DataFrame()
    repeated = changes.loc[
        (changes["outcome"] == outcome) & changes["event_order"].ge(2)
    ].copy() if not changes.empty else pd.DataFrame()
    if order_summary.empty:
        st.info("No supported paired events by opening order are available.")
    else:
        st.dataframe(order_summary, width="stretch", hide_index=True)
        st.caption(
            "Opening order reflects prior qualifying nearby pair events in the "
            "available distance workbook, including events before the outcome panel "
            "starts. `previous_nearby_openings_in_pair_history` counts earlier "
            "opening pairs for that existing store; it is not the number of new "
            "stores opened in the selected before/after window."
        )
    if not repeated.empty:
        st.write(
            f"Repeated-exposure sample: {repeated['_existing_key'].nunique():,} "
            f"existing stores and {repeated['event_id'].nunique():,} paired opening "
            "events with valid before/after outcomes."
        )
        repeat_event_summary = summarize_opening_changes(
            repeated,
            outcome,
            minimum_events=minimum_events,
            minimum_stores=minimum_stores,
        )
        repeat_table = repeat_event_summary.loc[
            repeat_event_summary["group_type"].isin(
                [
                    "Exposure order × region",
                    "Exposure order × distance band",
                    "Exposure order × region × distance band",
                ]
            )
        ]
        st.dataframe(repeat_table, width="stretch", hide_index=True)
    else:
        st.info("No repeated-opening events with paired valid outcomes in this selection.")


def download_table(data: pd.DataFrame, label: str, file_name: str) -> None:

    st.download_button(

        label=label,

        data=data.to_csv(index=False).encode("utf-8-sig"),

        file_name=file_name,

        mime="text/csv",

    )





def effect_chart(data: pd.DataFrame, x: str, color: str | None = None) -> None:

    if data.empty:

        st.info("No estimates are available for this selection.")

        return

    fig = go.Figure()

    data = data.copy()

    for column in ("observations", "stores", "exposed_stores"):

        if column not in data:

            data[column] = pd.NA

    groups = [(None, data)] if color is None else list(data.groupby(color, dropna=False))

    for group_name, group in groups:

        group = group.sort_values(x)

        fig.add_trace(

            go.Scatter(

                x=group[x],

                y=group["percent_effect"],

                error_y={

                    "type": "data",

                    "symmetric": False,

                    "array": group["percent_ci_95_upper"] - group["percent_effect"],

                    "arrayminus": group["percent_effect"] - group["percent_ci_95_lower"],

                },

                mode="lines+markers",

                name=str(group_name) if group_name is not None else "Estimate",

                customdata=group[

                    ["observations", "stores", "exposed_stores"]

                ].to_numpy(),

                hovertemplate=(

                    "%{x}<br>Association: %{y:.1f}%"

                    "<br>95% CI upper/lower shown"

                    "<br>Store-month observations: %{customdata[0]}"

                    "<br>Stores: %{customdata[1]}"

                    "<br>Exposed stores: %{customdata[2]}<extra></extra>"

                ),

            )

        )

    fig.add_hline(y=0, line_color="black", line_width=1)

    fig.update_layout(

        yaxis_title="Estimated associated change (%)",

        xaxis_title=x.replace("_", " ").title(),

        hovermode="x unified",

        legend_title_text=color.replace("_", " ").title() if color else "",

    )

    st.plotly_chart(fig, width="stretch")





def add_common_warning(pretrends: pd.DataFrame, region: str, outcome: str) -> None:

    selected = pretrends.loc[

        (pretrends["region"] == region)

        & (pretrends["outcome"] == outcome)

        & (pretrends["model_type"] == "binned_event_study")

    ] if not pretrends.empty else pd.DataFrame()

    if selected.empty:

        st.warning(

            "Pre-trend diagnostic is unavailable. Interpret these estimates as "

            "associations, not established causal effects."

        )

        return

    p_value = pd.to_numeric(

        pd.Series([selected.iloc[0].get("joint_p_value")]), errors="coerce"

    ).iloc[0]

    if pd.notna(p_value) and p_value < 0.05:

        st.error(

            f"Pre-opening leads are jointly significant (p={p_value:.3g}). "

            "double check needed."

        )

    elif pd.notna(p_value):

        st.info(

            f"No joint pre-trend rejection at 5% (p={p_value:.3g}); this does not "

            "by itself establish causal identification."

        )

    else:

        st.warning(

            "The joint pre-trend test could not be estimated. Treat results as "

            "associations."

        )





def _format_mnt(value: float) -> str:

    return f"MNT {value:,.0f}"





def _format_beta_percent(beta: float) -> str:

    try:

        return f"{100 * math.expm1(beta):.1f}%"

    except OverflowError:

        return "Outside numeric range"





def _capacity_cell(result: CapacityResult) -> str:

    if result.status == "Finite capacity":

        return f"{int(result.maximum_openings or 0):,}"

    return result.status





def _show_capacity_diagnostics(

    estimates: dict[str, CoefficientEstimate],

    region: str,

    pretrends: pd.DataFrame,

) -> None:

    diagnostic = pretrends.loc[

        (pretrends["region"] == region)

        & (pretrends["outcome"] == "sales")

        & (pretrends["model_type"] == "binned_event_study")

    ] if not pretrends.empty else pd.DataFrame()

    if diagnostic.empty:

        st.warning(

            "The regional sales pre-trend diagnostic is unavailable; interpret "

            "the estimates as associations."

        )

    else:

        p_value = pd.to_numeric(

            pd.Series([diagnostic.iloc[0].get("joint_p_value")]), errors="coerce"

        ).iloc[0]

        if pd.notna(p_value) and p_value < 0.05:

            st.warning(

                f"The regional binned event-study pre-trend test rejects at 5% "

                f"(p={p_value:.3g}); causal interpretation is not supported."

            )

        elif pd.notna(p_value):

            st.info(

                f"No regional pre-trend rejection at 5% (p={p_value:.3g}); "

                "this does not by itself establish causal identification."

            )

        else:

            st.warning("The regional pre-trend test could not be estimated.")



    limited_support = [

        f"{dict(DISTANCE_BANDS)[band]} ({estimate.exposed_stores} exposed stores)"

        for band, estimate in estimates.items()

        if estimate.beta is not None

        and estimate.exposed_stores is not None

        and estimate.exposed_stores < 10

    ]

    if limited_support:

        st.warning(

            "Very few stores contribute exposure support for: "

            + "; ".join(limited_support)

            + ". Treat these estimates as especially uncertain."

        )

    missing_estimates = [

        dict(DISTANCE_BANDS)[band]

        for band, estimate in estimates.items()

        if estimate.beta is None

    ]

    if missing_estimates:

        st.warning(

            "No valid cumulative sales coefficient is available for: "

            + ", ".join(missing_estimates)

            + ". Those bands are not treated as zero effect."

        )

    uncertain_intervals = [

        dict(DISTANCE_BANDS)[band]

        for band, estimate in estimates.items()

        if estimate.confidence_lower is not None

        and estimate.confidence_upper is not None

        and (

            estimate.confidence_lower <= 0 <= estimate.confidence_upper

            or estimate.confidence_upper - estimate.confidence_lower

            > 2 * abs(estimate.beta or 0)

        )

    ]

    if uncertain_intervals:

        st.warning(

            "The 95% coefficient intervals are imprecise (include zero or are "

            "wider than twice the absolute central estimate) for: "

            + ", ".join(uncertain_intervals)

            + "."

        )



    rows = []

    for band, label in DISTANCE_BANDS:

        estimate = estimates[band]

        rows.append(

            {

                "Distance band": label,

                "Log coefficient": (

                    f"{estimate.beta:.4f}"

                    if estimate.beta is not None

                    else "Not estimable"

                ),

                "Percent effect per opening": (

                    _format_beta_percent(estimate.beta)

                    if estimate.beta is not None

                    else "Not estimable"

                ),

                "95% CI (percent)": (

                    f"{100 * math.expm1(estimate.confidence_lower):.1f}% to "

                    f"{100 * math.expm1(estimate.confidence_upper):.1f}%"

                    if estimate.confidence_lower is not None

                    and estimate.confidence_upper is not None

                    else "Unavailable"

                ),

                "Exposed stores": (

                    f"{estimate.exposed_stores:,}"

                    if estimate.exposed_stores is not None

                    else "Unavailable"

                ),

                "Observations": (

                    f"{estimate.observations:,}"

                    if estimate.observations is not None

                    else "Unavailable"

                ),

                "Status": estimate.issue or "Estimated",

            }

        )

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)





def store_opening_capacity_page(

    sales_coefficients: pd.DataFrame,

    pretrends: pd.DataFrame,

) -> None:

    st.header("Store Opening Capacity Simulator")

    st.caption(

        "The simulator estimates how many additional nearby CU openings are "

        "compatible with a user-defined minimum APSD sales threshold, using "

        "historical regional cannibalization regression estimates. Results are "

        "conditional scenarios, not guaranteed forecasts or definitive "

        "site-opening recommendations."

    )

    st.markdown(

        "The estimates describe historical associations and may not identify "

        "purely causal effects. Future openings may differ from historical ones; "

        "the calculations assume estimated effects apply to the proposed "

        "combination of openings. New stores may affect one another. Actual "

        "capacity also depends on demand, market size, operating costs, new-store "

        "sales, and geographic availability. This is not a count of stores CU can "

        "physically or commercially open."

    )



    with st.form("store_opening_capacity_form"):

        baseline = st.number_input(

            "Current APSD Sales (MNT per active sales day)",

            min_value=0.01,

            value=4_000_000.0,

            step=100_000.0,

            format="%.0f",

        )

        minimum = st.number_input(

            "Minimum Required APSD Sales (MNT per active sales day)",

            min_value=0.01,

            value=3_000_000.0,

            step=100_000.0,

            format="%.0f",

        )

        region = st.selectbox("Existing-store region", list(REGIONS))

        mode = st.radio(

            "Simulation mode",

            [

                "Single-distance-band simulation",

                "Mixed-distance simulation",

                "Maximum Combined Openings",

            ],

        )

        selected_band = st.selectbox(

            "Distance band for single-band simulation and sensitivity chart",

            list(DISTANCE_LABELS),

            format_func=lambda band: DISTANCE_LABELS[band],

        )

        proposed: dict[str, int] = {}

        upper_bounds: dict[str, int] = {}

        if mode == "Mixed-distance simulation":

            st.markdown("**Proposed additional openings by distance band**")

            cols = st.columns(4)

            for col, (band, label) in zip(cols, DISTANCE_BANDS):

                proposed[band] = col.number_input(

                    f"{label}",

                    min_value=0,

                    value=0,

                    step=1,

                    key=f"mixed_openings_{band}",

                )

        elif mode == "Maximum Combined Openings":

            st.markdown(

                "**Maximum candidate openings by distance band** "

                "(each bound is capped at 25 to keep exhaustive search bounded)"

            )

            cols = st.columns(4)

            for col, (band, label) in zip(cols, DISTANCE_BANDS):

                upper_bounds[band] = col.number_input(

                    f"{label} upper bound",

                    min_value=0,

                    max_value=25,

                    value=10,

                    step=1,

                    key=f"opening_bound_{band}",

                )

        submitted = st.form_submit_button("Run Simulation")



    estimates = regional_sales_estimates(sales_coefficients, region)

    with st.expander(f"Selected region estimates and diagnostics — {region}", expanded=True):

        _show_capacity_diagnostics(estimates, region, pretrends)



    if submitted:

        st.session_state.pop("store_opening_capacity_result", None)

        try:

            validate_sales_inputs(float(baseline), float(minimum))

            betas = {band: estimate.beta for band, estimate in estimates.items()}

            if mode == "Single-distance-band simulation":

                capacity = maximum_openings(

                    float(baseline), float(minimum), estimates[selected_band].beta

                )

                count = (

                    capacity.maximum_openings

                    if capacity.status == "Finite capacity"

                    else (1 if estimates[selected_band].beta is not None else 0)

                )

                counts = {selected_band: int(count or 0)}

                scenario = (

                    simulate_mixed_scenario(

                        float(baseline), float(minimum), counts, betas

                    )

                    if estimates[selected_band].beta is not None

                    else None

                )

                allocation = None

            elif mode == "Mixed-distance simulation":

                capacity = None

                counts = proposed.copy()

                scenario = simulate_mixed_scenario(

                    float(baseline), float(minimum), counts, betas

                )

                allocation = None

            else:

                capacity = None

                allocation, scenario = optimize_openings(

                    float(baseline), float(minimum), upper_bounds, betas

                )

                counts = allocation.copy()

            st.session_state["store_opening_capacity_result"] = {

                "baseline": float(baseline),

                "minimum": float(minimum),

                "region": region,

                "mode": mode,

                "selected_band": selected_band,

                "counts": counts,

                "capacity": capacity,

                "scenario": scenario,

                "allocation": allocation,

            }

        except (ValueError, ArithmeticError) as error:

            st.error(str(error))



    result = st.session_state.get("store_opening_capacity_result")

    if result is None:

        return



    baseline = result["baseline"]

    minimum = result["minimum"]

    result_region = result["region"]

    result_estimates = regional_sales_estimates(sales_coefficients, result_region)

    betas = {band: estimate.beta for band, estimate in result_estimates.items()}

    scenario: ScenarioResult | None = result["scenario"]

    counts: dict[str, int] = result["counts"]

    capacity: CapacityResult | None = result["capacity"]



    st.subheader("Simulation results")

    if result["mode"] == "Single-distance-band simulation":

        label = DISTANCE_LABELS[result["selected_band"]]

        st.markdown(f"**Single-band scenario — all openings within {label}.**")

        if capacity is not None:

            if capacity.status == "Finite capacity":

                st.success(

                    f"Estimated maximum: **{capacity.maximum_openings:,}** "

                    f"additional opening(s) in this single band."

                )

                st.caption(

                    f"Boundary check: APSD at the maximum is "

                    f"{_format_mnt(capacity.predicted_at_maximum or 0)}; at one "

                    f"additional opening it is "

                    f"{_format_mnt(capacity.predicted_at_next or 0)}."

                )

            elif capacity.status == "No finite maximum under this model":

                st.info(capacity.issue)

            else:

                st.warning(f"{capacity.status}: {capacity.issue}")

    elif result["mode"] == "Maximum Combined Openings":

        st.markdown(

            "**Model- and bound-constrained allocation.** This allocation is "

            "optimal only for these coefficient estimates and user-defined bounds; "

            "it is not an optimal real-world site plan."

        )

        st.dataframe(

            pd.DataFrame(

                [

                    {

                        "Distance band": DISTANCE_LABELS[band],

                        "Selected openings": result["allocation"][band],

                    }

                    for band, _ in DISTANCE_BANDS

                ]

            ),

            width="stretch",

            hide_index=True,

        )



    if scenario is not None:

        metric_cols = st.columns(4)

        metric_cols[0].metric("Total additional openings", f"{scenario.total_openings:,}")

        metric_cols[1].metric("Predicted APSD sales", _format_mnt(scenario.predicted_apsd))

        metric_cols[2].metric(

            "Predicted APSD change",

            _format_mnt(scenario.absolute_change),

            f"{scenario.percentage_change:+.2f}%",

        )

        metric_cols[3].metric(

            "Headroom above minimum",

            _format_mnt(scenario.headroom),

            "Constraint satisfied" if scenario.constraint_satisfied else "Below minimum",

        )

        if scenario.constraint_satisfied:

            st.success("The predicted APSD meets the specified minimum.")

        else:

            st.error("The predicted APSD is below the specified minimum.")

    else:

        st.warning(

            "A predicted APSD scenario is unavailable because the selected regional "

            "coefficient is not estimable."

        )



    show_conservative = st.checkbox(

        "Show conservative 95% CI sensitivity",

        value=False,

        help=(

            "Uses the lower endpoint of each available pointwise 95% coefficient "

            "interval. This is a sensitivity case, not a guaranteed worst case."

        ),

        key="show_capacity_conservative",

    )

    if show_conservative and scenario is not None:

        active_bands = [band for band, count in counts.items() if count > 0]

        missing_intervals = [

            band

            for band in active_bands

            if result_estimates[band].confidence_lower is None

        ]

        if missing_intervals:

            st.warning(

                "Conservative sensitivity is unavailable because valid 95% "

                "confidence intervals are missing for: "

                + ", ".join(DISTANCE_LABELS[band] for band in missing_intervals)

                + "."

            )

        else:

            conservative_betas = {

                band: (

                    result_estimates[band].confidence_lower

                    if result_estimates[band].confidence_lower is not None

                    else estimate.beta

                )

                for band, estimate in result_estimates.items()

            }

            conservative = simulate_mixed_scenario(

                baseline, minimum, counts, conservative_betas

            )

            st.info(

                f"Conservative sensitivity APSD: "

                f"{_format_mnt(conservative.predicted_apsd)} "

                f"({conservative.percentage_change:+.2f}% vs baseline); "

                f"minimum met: {'Yes' if conservative.constraint_satisfied else 'No'}. "

                "This uses pointwise lower confidence bounds and is not a guaranteed "

                "worst case or a joint 95% interval."

            )



    comparison_rows: list[dict[str, object]] = []

    comparison_plot_rows: list[dict[str, object]] = []

    for compare_region in REGIONS:

        compare_estimates = regional_sales_estimates(sales_coefficients, compare_region)

        row: dict[str, object] = {

            "Region": compare_region,

            "Current APSD sales": _format_mnt(baseline),

            "Minimum required APSD sales": _format_mnt(minimum),

        }

        for band, label in DISTANCE_BANDS:

            estimate = compare_estimates[band]

            result_for_band = maximum_openings(baseline, minimum, estimate.beta)

            row[f"Maximum openings, {label}"] = _capacity_cell(result_for_band)

            if result_for_band.status == "Finite capacity":

                comparison_plot_rows.append(

                    {

                        "Region": compare_region,

                        "Distance band": label,

                        "Maximum additional openings": result_for_band.maximum_openings,

                    }

                )

        comparison_rows.append(row)



    st.subheader("Regional comparison — single-band capacities")

    st.caption(

        "Each cell is a separate single-distance-band scenario. The maxima for "

        "different bands must not be added or assumed achievable simultaneously."

    )

    st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)

    if comparison_plot_rows:

        capacity_chart = px.bar(

            pd.DataFrame(comparison_plot_rows),

            x="Region",

            y="Maximum additional openings",

            color="Distance band",

            barmode="group",

            title="Maximum additional openings by region and distance band",

            category_orders={"Region": list(REGIONS)},

        )

        capacity_chart.update_layout(yaxis_title="Additional openings (count)")

        st.plotly_chart(capacity_chart, width="stretch")

    else:

        st.info("No finite regional capacities are available to plot.")



    st.subheader("APSD sales scenario")

    predicted_value = scenario.predicted_apsd if scenario is not None else None

    apsd_chart = go.Figure()

    apsd_chart.add_trace(

        go.Bar(

            x=["Current APSD", "Minimum required APSD", "Predicted APSD"],

            y=[baseline, minimum, predicted_value],

            marker_color=["#4C78A8", "#F58518", "#54A24B"],

            hovertemplate="%{x}<br>MNT %{y:,.0f}<extra></extra>",

        )

    )

    apsd_chart.update_layout(

        yaxis_title="APSD sales (MNT per active sales day)",

        showlegend=False,

    )

    st.plotly_chart(apsd_chart, width="stretch")



    st.subheader("Proposed opening composition")

    composition = pd.DataFrame(

        [

            {

                "Distance band": DISTANCE_LABELS[band],

                "Proposed openings": counts.get(band, 0),

            }

            for band, _ in DISTANCE_BANDS

        ]

    )

    composition_chart = px.bar(

        composition,

        x="Distance band",

        y="Proposed openings",

        title="Scenario composition by distance band",

    )

    composition_chart.update_layout(yaxis_title="Proposed additional openings (count)")

    st.plotly_chart(composition_chart, width="stretch")



    st.subheader("APSD sales sensitivity — selected distance band")

    sensitivity_beta = result_estimates[result["selected_band"]].beta

    if sensitivity_beta is None:

        st.warning(

            "Sensitivity is unavailable because the selected distance-band "

            "coefficient is not estimable for this region."

        )

    else:

        selected_capacity = maximum_openings(

            baseline, minimum, sensitivity_beta

        )

        end_count = (

            min(max(selected_capacity.maximum_openings or 0, 10) + 1, 50)

            if selected_capacity.status == "Finite capacity"

            else 10

        )

        try:

            sensitivity = pd.DataFrame(

                {

                    "Additional openings": list(range(end_count + 1)),

                    "Predicted APSD sales (MNT)": [

                        predict_apsd(

                            baseline,

                            {result["selected_band"]: count},

                            {result["selected_band"]: sensitivity_beta},

                        )

                        for count in range(end_count + 1)

                    ],

                }

            )

        except ValueError as error:

            st.warning(f"APSD sensitivity cannot be plotted: {error}")

        else:

            sensitivity_chart = px.line(

                sensitivity,

                x="Additional openings",

                y="Predicted APSD sales (MNT)",

                markers=True,

                title=f"Single-band sensitivity — {DISTANCE_LABELS[result['selected_band']]}",

            )

            sensitivity_chart.add_hline(

                y=minimum,

                line_dash="dash",

                line_color="#E45756",

                annotation_text="Minimum required APSD",

            )

            sensitivity_chart.update_yaxes(tickprefix="MNT ", tickformat=",.0f")

            st.plotly_chart(sensitivity_chart, width="stretch")

            if selected_capacity.status == "Finite capacity" and selected_capacity.maximum_openings is not None and selected_capacity.maximum_openings >= 50:

                st.caption("The sensitivity chart is capped at 50 openings for readability.")



    st.markdown(

        "**Interpretation limitations:** estimates reflect historical associations "

        "and can be affected by local demand trends. The simulated combination "

        "assumes additive log-scale coefficients, while actual stores may interact. "

        "Pointwise confidence bounds cannot be combined into a joint interval "

        "because coefficient covariance is unavailable."

    )


def store_level_exploration_page(
    panel: pd.DataFrame,
    pair_source: pd.DataFrame,
    master: pd.DataFrame,
    outcome: str,
) -> None:
    st.header("Store-level exploration")
    panel_source = panel.copy()
    panel_source["_existing_key"] = normalize_store_id(
        panel_source["existing_store_id"]
    )
    name_lookup = pd.DataFrame(columns=["_existing_key", "store_name"])
    if "store_name" in panel_source:
        name_lookup = panel_source[
            ["_existing_key", "store_name"]
        ].copy()
    if not master.empty and {"store_id", "store_name"}.issubset(master.columns):
        master_names = master[["store_id", "store_name"]].copy()
        master_names["_existing_key"] = normalize_store_id(
            master_names["store_id"]
        )
        master_names = master_names[["_existing_key", "store_name"]]
        name_lookup = (
            pd.concat([name_lookup, master_names], ignore_index=True)
            if not name_lookup.empty
            else master_names.copy()
        )
    name_lookup["store_name"] = name_lookup["store_name"].astype("string").str.strip()
    name_lookup["store_name"] = name_lookup["store_name"].mask(
        name_lookup["store_name"].isin(["", "nan", "None", "<NA>"]), pd.NA
    )
    name_lookup = (
        name_lookup.dropna(subset=["_existing_key", "store_name"])
        .drop_duplicates("_existing_key")
    )
    store_options = (
        panel_source[["_existing_key", "existing_store_id"]]
        .dropna(subset=["_existing_key"])
        .drop_duplicates("_existing_key")
        .merge(name_lookup, on="_existing_key", how="left", validate="one_to_one")
    )
    store_options["display_name"] = store_options["store_name"].fillna("Unnamed store")
    duplicate_names = store_options["display_name"].duplicated(keep=False)
    store_options.loc[duplicate_names, "display_name"] = (
        store_options.loc[duplicate_names, "display_name"]
        + " (ID: "
        + store_options.loc[duplicate_names, "_existing_key"]
        + ")"
    )
    store_options = store_options.sort_values(
        "display_name", key=lambda values: values.str.lower()
    )
    if store_options.empty:
        st.warning("No existing stores are available in the analysis panel.")
        return
    store_lookup = dict(
        zip(store_options["display_name"], store_options["_existing_key"])
    )
    selected_name = st.selectbox(
        "Select store",
        store_options["display_name"].tolist(),
        help="The selected store name is mapped to its internal store ID.",
        key="store_exploration_name",
    )
    selected_key = store_lookup[selected_name]
    store_panel = panel_source.loc[
        panel_source["_existing_key"] == selected_key
    ].sort_values("SalesMonth").copy()
    if store_panel.empty:
        st.warning("No panel observations were found for this store.")
        return
    selected_id = store_panel["existing_store_id"].iloc[0]
    store_name = selected_name
    region = (
        str(store_panel["analysis_region"].dropna().iloc[0])
        if store_panel["analysis_region"].notna().any()
        else "Unmapped"
    )
    opening_date = (
        _parse_dashboard_dates(store_panel["ost_dt"]).dropna().iloc[0]
        if store_panel["ost_dt"].notna().any()
        else pd.NaT
    )
    closing_date = (
        _parse_dashboard_dates(store_panel["cls_dt"]).dropna().iloc[-1]
        if store_panel["cls_dt"].notna().any()
        else pd.NaT
    )
    if pair_source.empty:
        store_events = pd.DataFrame()
        all_observations = pd.DataFrame()
        band_order: list[str] = []
    else:
        try:
            all_events, all_observations, _, band_order = prepare_dashboard_openings(
                panel, pair_source, master
            )
        except ValueError as error:
            st.error(f"Cannot build store opening events: {error}")
            return
        store_events = all_events.loc[
            all_events["_existing_key"] == selected_key
        ].copy()
        all_observations = all_observations.loc[
            all_observations["_existing_key"] == selected_key
        ].copy()

    store_event_history = store_events.copy()
    if "new_store_name" in store_events:
        store_events["new_store_name"] = (
            store_events["new_store_name"]
            .astype("string")
            .str.strip()
            .mask(lambda names: names.eq(""), pd.NA)
            .fillna("Unnamed new store")
        )
        store_event_history["new_store_name"] = store_events["new_store_name"]

    st.write(f"**Store:** {store_name}")
    st.write(f"**Store ID:** {selected_id}")
    st.write(f"**Region:** {region}")
    st.write(
        f"**Store opening date:** "
        f"{opening_date:%Y-%m-%d}" if pd.notna(opening_date) else
        "**Store opening date:** unavailable"
    )
    st.write(
        f"**Store closing date:** "
        f"{closing_date:%Y-%m-%d}" if pd.notna(closing_date) else
        "**Store closing date:** open or unavailable"
    )
    outcome_column = "APSD_income" if outcome == "sales" else "APSD_customers"
    outcome_title = _metric_label(outcome_column)
    latest_sales = (
        store_panel.dropna(subset=["APSD_income"])
        .sort_values("SalesMonth")
        .tail(1)
    )
    latest_customers = (
        store_panel.dropna(subset=["APSD_customers"])
        .sort_values("SalesMonth")
        .tail(1)
    )

    if not store_events.empty:
        store_events, _ = _filter_events_by_dates(
            store_events, f"store_opening_dates_{selected_key}"
        )
        selected_store_bands = st.multiselect(
            "Nearby opening distance bands",
            options=band_order,
            default=band_order,
            key=f"store_distance_bands_{selected_key}",
        )
        selected_store_events = store_events.loc[
            store_events["distance_band"].isin(selected_store_bands)
        ].copy()
    else:
        selected_store_events = pd.DataFrame()
        st.info("No qualifying nearby-store opening pair is recorded for this store.")

    metrics = st.columns(4)
    metrics[0].metric(
        "Latest APSD sales",
        f"{latest_sales['APSD_income'].iloc[0]:,.2f}"
        if not latest_sales.empty else "Unavailable",
        latest_sales["SalesMonth"].iloc[0].strftime("%Y-%m")
        if not latest_sales.empty else "No valid outcome",
    )
    metrics[1].metric(
        "Latest APSD customers",
        f"{latest_customers['APSD_customers'].iloc[0]:,.2f}"
        if not latest_customers.empty else "Unavailable",
        latest_customers["SalesMonth"].iloc[0].strftime("%Y-%m")
        if not latest_customers.empty else "No valid outcome",
    )
    metrics[2].metric(
        "Nearby stores opened in selected date/band filters",
        f"{selected_store_events['_opening_key'].nunique():,}"
        if not selected_store_events.empty else "0",
    )
    metrics[3].metric(
        "Qualifying opening-pair events",
        f"{len(selected_store_events):,}",
    )

    store_events_for_plot = selected_store_events.sort_values("new_opening_date")
    timeline = store_panel[
        ["SalesMonth", outcome_column]
        + (["Active_Days"] if "Active_Days" in store_panel else [])
    ].copy()
    timeline["SalesMonth"] = pd.to_datetime(timeline["SalesMonth"], errors="coerce")
    timeline = timeline.dropna(subset=["SalesMonth"]).set_index("SalesMonth")
    if not timeline.empty:
        full_months = pd.date_range(
            timeline.index.min(), timeline.index.max(), freq="MS"
        )
        timeline = timeline.reindex(full_months)
        timeline.index.name = "SalesMonth"
        timeline = timeline.reset_index()
    else:
        timeline = pd.DataFrame(columns=["SalesMonth", outcome_column])
    fig = go.Figure()
    if not timeline.empty:
        fig.add_trace(
            go.Scatter(
                x=timeline["SalesMonth"],
                y=timeline[outcome_column],
                mode="lines+markers",
                connectgaps=False,
                name=outcome_title,
                customdata=(
                    timeline[["Active_Days"]].to_numpy()
                    if "Active_Days" in timeline
                    else None
                ),
                hovertemplate=(
                    "%{x|%Y-%m}<br>Observed APSD: %{y:.2f}"
                    "<br>Active days: %{customdata[0]}<extra></extra>"
                    if "Active_Days" in timeline
                    else "%{x|%Y-%m}<br>Observed APSD: %{y:.2f}<extra></extra>"
                ),
            )
        )
    marker_count = 0
    for event in store_events_for_plot.itertuples(index=False):
        marker_count += 1
        fig.add_shape(
            type="line",
            x0=event.new_opening_date,
            x1=event.new_opening_date,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line={"dash": "dot", "color": "#D62728", "width": 1},
            opacity=0.55,
        )
        fig.add_annotation(
            x=event.new_opening_date,
            y=1,
            xref="x",
            yref="paper",
            text=f"{event.new_store_name} ({event.distance_band})",
            textangle=-90,
            yanchor="top",
            showarrow=False,
        )
    if marker_count:
        st.caption(
            f"{marker_count:,} nearby opening-date marker(s) are overlaid. "
            "Use the distance/date filters to reduce overlap."
        )
    fig.update_layout(
        title=f"Observed {outcome_title} — {store_name}",
        xaxis_title="Observed calendar month",
        yaxis_title=outcome_title,
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")

    if not selected_store_events.empty:
        event_table = selected_store_events.copy()
        event_table["opening_event_order"] = event_table["event_order"].map(
            _event_order_label
        )
        event_table["event_month_relative_to_opening"] = 0
        event_table["new_store_name"] = event_table["new_store_name"].fillna(
            "Unnamed new store"
        )
        table_columns = [
            column
            for column in (
                "new_store_name",
                "new_store_id",
                "new_opening_date",
                "distance_m",
                "distance_band",
                "opening_event_order",
                "previous_nearby_openings_in_pair_history",
                "event_month_relative_to_opening",
                "event_id",
            )
            if column in event_table
        ]
        st.subheader("Nearby new-store opening events")
        st.dataframe(
            event_table[table_columns].sort_values("new_opening_date"),
            width="stretch",
            hide_index=True,
        )
        download_table(
            event_table[table_columns],
            "Download nearby opening events",
            f"store_{selected_key}_nearby_openings.csv",
        )

        event_options: dict[str, str] = {}
        for event in selected_store_events.itertuples(index=False):
            event_options[
                (
                    f"{event.new_store_name or 'Unnamed new store'} "
                    f"(ID {event.new_store_id}) — {event.new_opening_date:%Y-%m-%d} — "
                    f"{event.distance_m:,.0f} m, {event.distance_band} — "
                    f"{_event_order_label(event.event_order)}"
                )
            ] = str(event.event_id)
        event_labels = list(event_options)
        selected_event_label = st.selectbox(
            "Opening event for the store-specific event study",
            event_labels,
            key=f"store_selected_event_{selected_key}",
        )
        selected_event_id = event_options[selected_event_label]
        selected_event = selected_store_events.loc[
            selected_store_events["event_id"] == selected_event_id
        ].iloc[0]
        event_window = st.slider(
            "Months shown on each side of this opening",
            min_value=1,
            max_value=12,
            value=12,
            key=f"store_event_window_{selected_key}",
        )
        event_month_index = int(selected_event["opening_month_index"])
        month_index = _month_index(
            pd.to_datetime(store_panel["SalesMonth"], errors="coerce")
        )
        store_observations = store_panel.copy()
        store_observations["_calendar_month_index"] = month_index
        event_months = pd.DataFrame(
            {
                "event_time_month": range(-event_window, event_window + 1),
            }
        )
        event_months["SalesMonth"] = pd.to_datetime(
            (
                event_month_index + event_months["event_time_month"]
            ).map(lambda value: f"{(value - 1) // 12:04d}-{(value - 1) % 12 + 1:02d}-01"),
            errors="coerce",
        )
        event_plot = event_months.merge(
            store_observations[["SalesMonth", outcome_column] + (
                ["Active_Days"] if "Active_Days" in store_observations else []
            )],
            on="SalesMonth",
            how="left",
            validate="one_to_one",
        )
        overlapping = store_event_history.loc[
            store_event_history["event_id"].ne(selected_event_id)
            & (
                (
                    store_event_history["opening_month_index"] - event_month_index
                ).between(-event_window, event_window)
            )
        ] if not store_event_history.empty else pd.DataFrame()
        st.subheader("Selected-opening event study")
        st.write(
            f"**New store:** {selected_event['new_store_name'] or 'Unnamed new store'} "
            f"(ID {selected_event['new_store_id']}); "
            f"**opened:** {selected_event['new_opening_date']:%Y-%m-%d}; "
            f"**distance:** {selected_event['distance_m']:,.0f} m "
            f"({selected_event['distance_band']}); "
            f"**order:** {_event_order_label(selected_event['event_order'])}."
        )
        if not overlapping.empty:
            st.warning(
                f"{len(overlapping)} other qualifying nearby opening(s) fall within "
                "this event-study window. The selected-store chart shows observed "
                "outcomes around the selected date, not an isolated effect of that one opening."
            )
            st.dataframe(
                overlapping[
                    [
                        "new_store_name",
                        "new_store_id",
                        "new_opening_date",
                        "distance_m",
                        "distance_band",
                        "event_order",
                    ]
                ].sort_values("new_opening_date"),
                width="stretch",
                hide_index=True,
            )
        event_fig = go.Figure(
            go.Scatter(
                x=event_plot["event_time_month"],
                y=event_plot[outcome_column],
                mode="lines+markers",
                connectgaps=False,
                name=outcome_title,
                customdata=(
                    event_plot[["Active_Days"]].to_numpy()
                    if "Active_Days" in event_plot
                    else None
                ),
                hovertemplate=(
                    "Event month %{x}<br>Observed APSD: %{y:.2f}"
                    "<br>Active days: %{customdata[0]}<extra></extra>"
                    if "Active_Days" in event_plot
                    else "Event month %{x}<br>Observed APSD: %{y:.2f}<extra></extra>"
                ),
            )
        )
        event_fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="#B91C1C",
            annotation_text="New-store opening month (0)",
        )
        event_fig.add_hline(y=0, line_color="#777777", line_width=1)
        event_fig.update_layout(
            title=f"{store_name}: observed {outcome_title} around selected opening",
            xaxis_title="Months relative to new-store opening month",
            yaxis_title=outcome_title,
        )
        st.plotly_chart(event_fig, width="stretch")
        st.caption(
            "Months absent from the saved panel are shown as gaps, never filled with "
            "zero. Month 0 is the calendar month containing the opening date and may "
            "be only partly exposed. Active days are shown where available, but daily "
            "records are not available to establish complete-month exposure."
        )
    else:
        st.info("No opening events match the selected distance and date filters.")

    st.subheader("Cumulative nearby-opening exposure")
    exposure_bands = st.multiselect(
        "Distance bands for cumulative opening count",
        options=band_order,
        default=band_order,
        key=f"store_exposure_bands_{selected_key}",
    )
    show_outcome_overlay = st.checkbox(
        f"Overlay {outcome_title} on secondary axis",
        value=True,
        key=f"store_exposure_overlay_{selected_key}",
    )
    if not timeline.empty:
        exposure_timeline = timeline[["SalesMonth", outcome_column]].copy()
        panel_month_index = _month_index(
            pd.to_datetime(exposure_timeline["SalesMonth"], errors="coerce")
        )
        exposure_timeline["_month_index"] = panel_month_index
        opening_counts = (
            store_events.loc[store_events["distance_band"].isin(exposure_bands)]
            .groupby(["distance_band", "opening_month_index"])
            .size()
            .rename("openings")
            .reset_index()
            if not store_events.empty
            else pd.DataFrame(columns=["distance_band", "opening_month_index", "openings"])
        )
        exposure_fig = go.Figure()
        for band in exposure_bands:
            counts = (
                opening_counts.loc[opening_counts["distance_band"] == band]
                .groupby("opening_month_index")["openings"]
                .sum()
                .sort_index()
                .cumsum()
                if not opening_counts.empty
                else pd.Series(dtype=float)
            )
            cumulative_counts = counts.reindex(
                counts.index.union(
                    pd.Index(exposure_timeline["_month_index"])
                )
            ).sort_index().ffill().fillna(0)
            exposure_timeline[f"cumulative_{band}"] = exposure_timeline[
                "_month_index"
            ].map(cumulative_counts)
            exposure_fig.add_trace(
                go.Scatter(
                    x=exposure_timeline["SalesMonth"],
                    y=exposure_timeline[f"cumulative_{band}"],
                    mode="lines",
                    name=f"Cumulative openings — {band}",
                    yaxis="y",
                )
            )
        if show_outcome_overlay:
            exposure_fig.add_trace(
                go.Scatter(
                    x=exposure_timeline["SalesMonth"],
                    y=exposure_timeline[outcome_column],
                    mode="lines+markers",
                    connectgaps=False,
                    name=outcome_title,
                    yaxis="y2",
                )
            )
        exposure_fig.update_layout(
            title=f"Cumulative new-store openings — {store_name}",
            xaxis_title="Calendar month",
            yaxis={"title": "Cumulative pair opening events"},
            yaxis2={
                "title": outcome_title,
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
            },
            hovermode="x unified",
        )
        st.plotly_chart(exposure_fig, width="stretch")
        contributing_events = store_events.loc[
            store_events["distance_band"].isin(exposure_bands)
            & store_events["new_store_opening_month"].le(timeline["SalesMonth"].max())
        ].copy() if not store_events.empty else pd.DataFrame()
        if not contributing_events.empty:
            st.caption(
                "Exposure is counted from unique existing/new-store pair events in "
                "each source distance band. It is not a count of store-month rows."
            )
            st.dataframe(
                contributing_events[
                    [
                        "new_store_name",
                        "new_store_id",
                        "new_opening_date",
                        "distance_m",
                        "distance_band",
                        "event_order",
                        "previous_nearby_openings_in_pair_history",
                    ]
                ].sort_values("new_opening_date"),
                width="stretch",
                hide_index=True,
            )
    st.info(
        "All store-level charts show observed outcomes and nearby opening chronology. "
        "They do not estimate a store-specific causal effect."
    )
    download_table(
        store_panel.drop(columns="_existing_key"),
        "Download this store's panel",
        f"store_{selected_key}_panel.csv",
    )


def main() -> None:

    st.set_page_config(

        page_title="CU Store Cannibalization",

        page_icon="🏪",

        layout="wide",

    )

    st.title("CU store cannibalization")

    st.caption(

        "Regression-based associations between nearby CU openings and existing "

        "stores' APSD sales and customer-equivalents. Estimates are not guaranteed "

        "forecasts or automatically causal effects."

    )



    sales_coefs = read_csv("regional_sales_coefficients.csv")

    customer_coefs = read_csv("regional_customer_coefficients.csv")

    pretrends = read_csv("pretrend_tests.csv")

    sample_summary = read_csv("regional_sample_summary.csv")

    event_support = read_csv("long_term_event_support.csv")

    seasonal_effects = read_csv("seasonal_effects.csv")

    seasonal_means = read_csv("seasonal_observed_means.csv")

    established = read_csv("established_store_effects.csv")

    followup = read_csv("established_followup_support.csv")

    distance_summary = read_csv("distance_exposure_summary.csv")

    model_status = read_csv("model_status.csv")

    openings_by_month = read_csv("new_store_openings_by_month.csv")

    panel = read_panel()

    pairs = read_pair_data()



    required_files = [

        "regional_model_comparison.csv",

        "regional_sales_coefficients.csv",

        "regional_customer_coefficients.csv",

        "cu_regional_cannibalization_panel.parquet",

    ]

    missing = [name for name in required_files if not (OUTPUT_DIR / name).is_file()]

    if missing:

        st.error(

            "Regional analysis outputs are incomplete. Run "

            "`04 Regional Cannibalization.py` first. Missing: "

            + ", ".join(missing)

        )

        st.stop()

    if panel.empty:

        st.error("The saved analysis panel could not be loaded.")

        st.stop()



    st.sidebar.header("Dashboard")

    page = st.sidebar.selectbox(

        "View",

        [

            "Executive Overview",

            "Regional Comparison",

            "Distance Analysis",

            "Long-Term Effects",

            "Seasonality",
            "Established Stores",
            "Regional & Distance Statistics",
            "Store-Level Exploration",

            "Store Opening Capacity Simulator",

        ],

    )

    outcome_label = st.sidebar.selectbox(

        "Outcome",

        ["APSD sales", "APSD customers"],

        help="APSD customers are transaction-derived customer-equivalents per active day, not unique people.",

    )

    outcome = "sales" if outcome_label == "APSD sales" else "customers"

    coefficients = sales_coefs if outcome == "sales" else customer_coefs



    if page == "Regional & Distance Statistics":
        regional_distance_statistics_page(
            panel,
            pairs,
            read_store_master(),
            "APSD_income" if outcome == "sales" else "APSD_customers",
            outcome_label,
        )
    elif page == "Store-Level Exploration":
        store_level_exploration_page(
            panel,
            pairs,
            read_store_master(),
            outcome,
        )
    elif page == "Store Opening Capacity Simulator":
        store_opening_capacity_page(sales_coefs, pretrends)

    elif page == "Executive Overview":

        st.header("Executive overview")

        region_row = (

            sample_summary.select_dtypes(include="number").sum()

            if not sample_summary.empty

            else pd.Series(dtype=object)

        )

        sales_effect = sales_coefs.loc[

            (sales_coefs["region"] == "All Regions")

            & (sales_coefs["model_type"] == "cumulative")

            & (sales_coefs["distance_band"] == "0_100")

        ]

        customer_effect = customer_coefs.loc[

            (customer_coefs["region"] == "All Regions")

            & (customer_coefs["model_type"] == "cumulative")

            & (customer_coefs["distance_band"] == "0_100")

        ]

        if region_row.empty:

            st.warning("Regional sample summary is unavailable.")

        else:

            new_opening_count = (

                int(openings_by_month["distinct_new_store_openings"].sum())

                if "distinct_new_store_openings" in openings_by_month

                else 0

            )

            cols = st.columns(4)

            cols[0].metric(

                "Existing stores in panel",

                f"{panel['existing_store_id'].nunique():,}",

            )

            cols[1].metric("Distinct new-store openings", f"{new_opening_count:,}")

            exposed_stores = panel.loc[

                panel["cum_openings_within_1km"] > 0,

                "existing_store_id",

            ].nunique()

            cols[2].metric("Stores exposed within 1 km", f"{exposed_stores:,}")

            cols[3].metric(

                "Mapped existing/new-store pair events",

                f"{int(region_row.get('new_store_pair_events', 0)):,}",

            )

            st.caption(

                f"Analysis panel: {len(panel):,} store-month observations; regional "

                "regressions use stores with a validated regional classification."

            )

        cols = st.columns(2)

        for col, data, title in (

            (cols[0], sales_effect, "APSD sales, one additional opening within 100 m"),

            (cols[1], customer_effect, "APSD customers, one additional opening within 100 m"),

        ):

            if not data.empty:

                row = data.iloc[0]

                col.metric(

                    title,

                    f"{row['percent_effect']:.1f}%",

                    help=(

                        f"95% CI: {row['percent_ci_95_lower']:.1f}% to "

                        f"{row['percent_ci_95_upper']:.1f}%; "

                        f"{int(row['observations']):,} observations, "

                        f"{int(row['exposed_stores']):,} exposed stores."

                    ),

                )

            else:

                col.metric(title, "Not estimable")

        long_term_items = []

        for outcome_name, outcome_data in (

            ("APSD sales", sales_coefs),

            ("APSD customers", customer_coefs),

        ):

            estimate = outcome_data.loc[

                (outcome_data["region"] == "All Regions")

                & (outcome_data["model_type"] == "binned_event_study")

                & (outcome_data["distance_band"] == "0_100")

                & (outcome_data["event_period_label"] == "+19 to +24")

            ]

            support = event_support.loc[

                (event_support["region"] == "All Regions")

                & (event_support["outcome"] == ("sales" if outcome_name == "APSD sales" else "customers"))

                & (event_support["time_kind"] == "binned")

                & (event_support["event_period"] == "+19 to +24")

                & (event_support["distance_band"] == "0_100")

                & (event_support["sample"] == "all qualifying openings")

            ] if not event_support.empty else pd.DataFrame()

            long_term_items.append((outcome_name, estimate, support))

        st.subheader("Long-term association, 19–24 months after opening")

        long_term_cols = st.columns(2)

        for col, (title, estimate, support) in zip(long_term_cols, long_term_items):

            if estimate.empty or support.empty or int(support.iloc[0]["distinct_pair_events"]) == 0:

                col.metric(title, "Not estimable")

                col.caption("There is no qualifying follow-up support for this estimate.")

            else:

                row = estimate.iloc[0]

                support_row = support.iloc[0]

                col.metric(

                    title,

                    f"{row['percent_effect']:.1f}%",

                    help=(

                        f"95% CI: {row['percent_ci_95_lower']:.1f}% to "

                        f"{row['percent_ci_95_upper']:.1f}%."

                    ),

                )

                col.caption(

                    f"{int(support_row['distinct_pair_events']):,} pair events; "

                    f"{int(support_row['exposed_existing_stores']):,} stores; "

                    f"{int(support_row['store_month_observations']):,} store-months."

                )

        st.markdown(

            "**What this measures:** the estimated change in the existing store's "

            "average daily sales or transaction-derived customer-equivalents after "

            "new stores open nearby, conditional on store and calendar-month effects. "

            "The main cumulative coefficient is per additional qualifying opening."

        )

        add_common_warning(pretrends, "All Regions", outcome)

        if not model_status.empty:

            st.subheader("Model availability")

            statuses = model_status.groupby(

                ["model_type", "status"], dropna=False

            ).size().rename("models").reset_index()

            st.dataframe(statuses, width="stretch", hide_index=True)



    elif page == "Regional Comparison":

        st.header("Regional comparison")

        region = st.selectbox("Existing-store region", list(REGIONS))

        add_common_warning(pretrends, region, outcome)

        primary = coefficients.loc[

            (coefficients["model_type"] == "cumulative")

            & (coefficients["distance_band"].notna())

            & (coefficients["region"] == region)

        ].copy()

        st.subheader(f"Distance effects in {region}")

        effect_chart(primary, "distance_band_label")

        regional_compare = coefficients.loc[

            (coefficients["model_type"] == "cumulative")

            & (coefficients["distance_band"] == "0_100")

            & (coefficients["region"].isin(REGIONS))

        ].copy()

        st.subheader("Very-close-opening comparison across regions")

        effect_chart(regional_compare, "region")

        if not sample_summary.empty:

            st.subheader("Regional sample and event counts")

            st.dataframe(sample_summary, width="stretch", hide_index=True)

        download_table(primary, "Download regional estimates", "regional_estimates.csv")



    elif page == "Distance Analysis":

        st.header("Distance analysis")

        region = st.selectbox("Existing-store region", ["All Regions", *REGIONS])

        distance_choice = st.selectbox(

            "Distance band",

            ["All distance bands", *DISTANCE_LABELS.values()],

        )

        primary = coefficients.loc[

            (coefficients["model_type"] == "cumulative")

            & (coefficients["region"] == region)

        ].copy()

        primary["distance_band_label"] = primary["distance_band"].map(

            DISTANCE_LABELS

        )

        if distance_choice != "All distance bands":

            primary = primary.loc[

                primary["distance_band_label"] == distance_choice

            ]

        primary["distance_band_label"] = pd.Categorical(

            primary["distance_band_label"],

            categories=list(DISTANCE_LABELS.values()),

            ordered=True,

        )

        effect_chart(primary, "distance_band_label")

        if not distance_summary.empty:

            st.subheader("Pair-event support")

            st.dataframe(

                distance_summary.loc[distance_summary["region"] == region],

                width="stretch",

                hide_index=True,

            )

        download_table(primary, "Download distance estimates", "distance_estimates.csv")



    elif page == "Long-Term Effects":

        st.header("Long-term event-study estimates")

        region = st.selectbox("Existing-store region", ["All Regions", *REGIONS])

        distance_choice = st.selectbox(

            "Distance band", list(DISTANCE_LABELS.values())

        )

        horizon_options = {

            "All observed periods": EVENT_ORDER,

            "Pre-opening": ["−12 to −7", "−6 to −3", "−2"],

            "Opening through 3 months": ["0", "+1 to +3"],

            "4–6 months": ["+4 to +6"],

            "7–12 months": ["+7 to +12"],

            "13–18 months": ["+13 to +18"],

            "19–24 months": ["+19 to +24"],

            "25+ months": ["+25 months onward"],

        }

        horizon_choice = st.selectbox("Event-time horizon", list(horizon_options))

        selected = coefficients.loc[

            (coefficients["region"] == region)

            & (coefficients["model_type"] == "binned_event_study")

            & (coefficients["distance_band"] == {

                label: band for band, label in DISTANCE_LABELS.items()

            }[distance_choice])

            & coefficients["event_period_label"].notna()

            & (coefficients["event_period_label"] != "−1 (reference)")

        ].copy()

        selected = selected.loc[

            selected["event_period_label"].isin(horizon_options[horizon_choice])

        ]

        selected["event_period_label"] = pd.Categorical(

            selected["event_period_label"], categories=EVENT_ORDER, ordered=True

        )

        effect_chart(selected, "event_period_label", "distance_band_label")

        add_common_warning(pretrends, region, outcome)

        support = event_support.loc[

            (event_support["region"] == region)

            & (event_support["outcome"] == outcome)

            & (event_support["time_kind"] == "binned")

            & (

                event_support["distance_band"]

                == {

                    label: band for band, label in DISTANCE_LABELS.items()

                }[distance_choice]

            )

            & (event_support["sample"] == "all qualifying openings")

        ].copy() if not event_support.empty else pd.DataFrame()

        if not support.empty:

            support = support.loc[

                support["event_period"].isin(horizon_options[horizon_choice])

            ]

            support = support.sort_values(

                "event_period",

                key=lambda column: column.map(

                    {label: i for i, label in enumerate(EVENT_ORDER)}

                ),

            )

            st.subheader("Events and observations contributing to each period")

            st.dataframe(support, width="stretch", hide_index=True)

            if not support.loc[

                support["event_period"].isin(["+19 to +24", "+25 months onward"]),

                "distinct_pair_events",

            ].gt(0).any():

                st.warning(

                    "There are no qualifying 19-24 month observations; a two-year "

                    "effect is not supported for this selection."

                )

        download_table(selected, "Download event-study estimates", "event_study_estimates.csv")



    elif page == "Seasonality":

        st.header("Seasonal heterogeneity")

        st.caption(

            "The chart separates descriptive observed seasonal patterns from "

            "regression-based exposure interactions. No unvalidated winter, summer, "

            "or holiday calendar is assumed."

        )

        if not seasonal_means.empty:

            region = st.selectbox("Observed pattern region", ["All Regions", *REGIONS])

            means = seasonal_means.loc[seasonal_means["region"] == region]

            col1, col2 = st.columns(2)

            for col, metric, title in (

                (col1, "mean_APSD_income", "Observed mean APSD sales"),

                (col2, "mean_APSD_customers", "Observed mean APSD customers"),

            ):

                if metric in means:

                    fig = px.line(

                        means,

                        x="month_of_year",

                        y=metric,

                        markers=True,

                        title=f"{title} (descriptive)",

                    )

                    fig.update_xaxes(dtick=1, title="Month of year")

                    col.plotly_chart(fig, width="stretch")

        if not seasonal_effects.empty:

            st.subheader("Exposure association by month of year")

            effects = seasonal_effects.loc[

                (seasonal_effects["outcome"] == outcome)

                & (seasonal_effects["effect_scope"]

                   == "One additional new-store opening within 1 km")

            ]

            effect_chart(effects, "month_label", "region")

            interaction_effects = seasonal_effects.loc[

                (seasonal_effects["outcome"] == outcome)

                & (seasonal_effects["effect_scope"]

                   == "Exposure interaction difference vs December")

                & (seasonal_effects["region"] != "All Regions")

            ]

            st.subheader("Seasonal difference relative to December")

            st.caption(

                "These interaction estimates show how the exposure association "

                "differs from December; they are not standalone monthly effects."

            )

            effect_chart(interaction_effects, "month_label", "region")

            download_table(

                seasonal_effects,

                "Download seasonal effects",

                "seasonal_effects.csv",

            )



    elif page == "Established Stores":

        st.header("Established stores (24+ months before opening)")

        st.caption(

            "Eligible stores had at least 24 calendar months of operating tenure "

            "before a nearby new-store opening. The repeated-opening exposure model "

            "retains overlapping qualifying openings."

        )

        region = st.selectbox("Existing-store region", ["All Regions", *REGIONS])

        distance_choice = st.selectbox(

            "Distance band", ["All distance bands", *DISTANCE_LABELS.values()]

        )

        horizon_choice = st.selectbox(

            "Event-time horizon",

            ["All observed periods", *EVENT_ORDER],

        )

        selected = established.loc[

            (established["region"] == region)

            & (established["outcome"] == outcome)

            & (established["model_type"] == "established_binned_event_study")

        ].copy() if not established.empty else pd.DataFrame()

        if distance_choice != "All distance bands":

            selected = selected.loc[

                selected["distance_band_label"] == distance_choice

            ]

        if horizon_choice != "All observed periods":

            selected = selected.loc[

                selected["event_period_label"] == horizon_choice

            ]

        effect_chart(selected, "event_period_label", "distance_band_label")

        if not followup.empty:

            st.subheader("Observed operating and outcome follow-up")

            support = followup.loc[

                (followup["region"] == region)

                & (followup["outcome"] == outcome)

            ]

            if distance_choice != "All distance bands":

                support = support.loc[

                    support["distance_band"] == distance_choice

                ]

            st.dataframe(support, width="stretch", hide_index=True)

        download_table(selected, "Download established-store effects", "established_effects.csv")



if __name__ == "__main__":

    main()
