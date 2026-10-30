#!/usr/bin/env python3
"""Daily pipeline: Metabase saved question -> pandas aggregation -> small summary tabs in Google Sheets.

Raw rows are never written to Google Sheets (cell limit). Only aggregated tabs + Refresh_Log are written.
Configuration comes from environment variables only.
"""
import json
import logging
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import gspread
import numpy as np
import pandas as pd
import requests
from google.oauth2.service_account import Credentials

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("metabase_sync")

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
SCORE_BANDS = [0, 40, 70]  # PLACEHOLDER - edit to real bands

REQUEST_TIMEOUT_S = 1200
FETCH_RETRIES = 3
WRITE_RETRIES = 3
WRITE_CHUNK_ROWS = 5000
# Safety guard: summary tabs are meant to be SMALL. Google Sheets allows 10M cells per spreadsheet.
MAX_TOTAL_CELLS = 5_000_000

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

NEEDED_COLUMNS = [
    "Lead ID",
    "Lead Creation Cohort Month",
    "City",
    "Channel",
    "LRM",
    "Heuristic Score",
    "Score Status",
    "Bucket 2.0",
    "Days Lead to First MS",
    "Days Lead to First MD",
    "MS Instance Key",
    "MS Weekday/Weekend",
    "MS Slot",
    "Schedule Records on MS Date",
    "Valid MD Flag",
    "Days MS to MD",
    "Order Attributed Flag",
    "RTM Flag",
    "MD Before Schedule Data Check",
]

NUMERIC_COLUMNS = [
    "Heuristic Score",
    "Days Lead to First MS",
    "Days Lead to First MD",
    "Schedule Records on MS Date",
    "Valid MD Flag",
    "Days MS to MD",
    "Order Attributed Flag",
    "RTM Flag",
    "MD Before Schedule Data Check",
]

DIMS = ["City", "Channel", "LRM", "Cohort Month"]

AGEING_BINS = [-np.inf, 1, 3, 7, 15, 22, 30, 60, np.inf]
AGEING_LABELS = ["D+1", "D+3", "D+7", "D+15", "D+22", "D+30", "D+60", "60+"]

UNMAPPED_LABEL = "UNMAPPED - REVIEW"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def with_retries(fn, what, retries, base_delay):
    """Run fn(); on failure retry up to `retries` times with exponential backoff."""
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if attempt >= retries:
                log.error("%s failed after %d retries: %s", what, retries, exc)
                raise
            delay = base_delay * (2 ** attempt)
            attempt += 1
            log.warning(
                "%s failed (%s). Retry %d/%d in %ds.", what, exc, attempt, retries, delay
            )
            time.sleep(delay)


def read_config():
    names = [
        "METABASE_URL",
        "METABASE_API_KEY",
        "METABASE_CARD_ID",
        "SPREADSHEET_ID",
        "GCP_SERVICE_ACCOUNT_JSON",
    ]
    cfg = {n: os.environ.get(n, "").strip() for n in names}
    missing = [n for n, v in cfg.items() if not v]
    if missing:
        log.error("Missing environment variables: %s", missing)
        sys.exit(2)
    cfg["METABASE_URL"] = cfg["METABASE_URL"].rstrip("/")
    return cfg


# ----------------------------------------------------------------------------
# Fetch
# ----------------------------------------------------------------------------
def fetch_dataframe(cfg):
    url = f"{cfg['METABASE_URL']}/api/card/{cfg['METABASE_CARD_ID']}/query/csv"
    needed = set(NEEDED_COLUMNS)

    def _once():
        log.info("POST %s (streaming, timeout %ss)", url, REQUEST_TIMEOUT_S)
        with requests.post(
            url,
            headers={"x-api-key": cfg["METABASE_API_KEY"]},
            data={"format_rows": "false"},
            stream=True,
            timeout=REQUEST_TIMEOUT_S,
        ) as resp:
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Metabase returned HTTP {resp.status_code}: {resp.text[:500]}"
                )
            resp.raw.decode_content = True
            return pd.read_csv(
                resp.raw,
                dtype=str,
                usecols=lambda c: c in needed,  # memory-safe: only needed columns
                keep_default_na=False,
                na_values=[""],
                encoding="utf-8-sig",
            )

    df = with_retries(_once, "Metabase fetch", FETCH_RETRIES, base_delay=15)

    missing = [c for c in NEEDED_COLUMNS if c not in df.columns]
    if missing:
        log.error("Required columns missing from Metabase result: %s", missing)
        sys.exit(3)
    return df[NEEDED_COLUMNS].copy()


# ----------------------------------------------------------------------------
# Cleaning and bands
# ----------------------------------------------------------------------------
def clean(df):
    # Blank / whitespace-only strings -> NaN
    for c in df.columns:
        s = df[c].str.strip()
        df[c] = s.mask(s == "")

    for c in NUMERIC_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Cohort Month"] = pd.to_datetime(
        df["Lead Creation Cohort Month"], errors="coerce"
    ).dt.strftime("%Y-%m")

    for c in DIMS:
        df[c] = df[c].fillna("Unknown")
    df["Bucket 2.0"] = df["Bucket 2.0"].fillna("Unknown")
    return df


def score_band_labels():
    labels = []
    for i, lo in enumerate(SCORE_BANDS):
        if i + 1 < len(SCORE_BANDS):
            labels.append(f"{lo}-{SCORE_BANDS[i + 1] - 1}")
        else:
            labels.append(f"{lo}+")
    return labels


def add_score_band(df):
    bins = list(SCORE_BANDS) + [np.inf]
    band = pd.cut(
        df["Heuristic Score"], bins=bins, labels=score_band_labels(), right=False
    ).astype(object)
    unscored = (df["Score Status"] == "Unscored") | df["Heuristic Score"].isna()
    # Scores below the first lower bound also fall out of pd.cut as NaN -> treated as Unscored
    df["Score Band"] = band.where(~unscored & band.notna(), "Unscored")
    return df


def ageing_band(days):
    band = pd.cut(days, bins=AGEING_BINS, labels=AGEING_LABELS, right=True).astype(object)
    return band.where(days.notna(), "Not yet")


def ms_to_md_band(days):
    conditions = [
        days.isna(),
        days <= 0,  # 0 -> Same day (negatives = MD before MS; see MD Before Schedule check)
        days == 1,
        days == 2,
        days == 3,
        days == 4,
        days == 5,
        days >= 6,
    ]
    choices = [
        "Not done",
        "Same day",
        "D+1",
        "D+2",
        "D+3",
        "D+4",
        "D+5",
        "D+6 and above",
    ]
    return pd.Series(np.select(conditions, choices, default="Not done"), index=days.index)


# ----------------------------------------------------------------------------
# Summary tables
# ----------------------------------------------------------------------------
def count_table(frame, group_cols, metric_name):
    return (
        frame.groupby(group_cols, as_index=False, sort=True)
        .size()
        .rename(columns={"size": metric_name})
    )


def build_tables(leads, inst):
    # 1. Agg_Score_Disp
    score_disp = count_table(leads, DIMS + ["Score Band", "Bucket 2.0"], "Leads")

    # 2. Agg_Ageing
    blocks = []
    for measure, col in (("MS", "Days Lead to First MS"), ("MD", "Days Lead to First MD")):
        block = leads[DIMS].copy()
        block["Measure"] = measure
        block["Ageing Band"] = ageing_band(leads[col])
        blocks.append(count_table(block, DIMS + ["Measure", "Ageing Band"], "Leads"))
    ageing = pd.concat(blocks, ignore_index=True)

    # 3. Agg_MS_MD
    inst = inst.copy()
    inst["MS to MD Band"] = ms_to_md_band(inst["Days MS to MD"])
    ms_md = count_table(inst, DIMS + ["Day Type", "MS to MD Band"], "MS Instances")

    # 4. Agg_Slot
    slot = inst.groupby(DIMS + ["MS Slot", "Day Type"], as_index=False, sort=True).agg(
        **{
            "All MS": ("Schedule Records on MS Date", "sum"),
            "Unique MS": ("MS Instance Key", "size"),
            "MD Done": ("Valid MD Flag", "sum"),
            "Orders": ("Order Attributed Flag", "sum"),
            "RTM MS": ("RTM Flag", "sum"),
        }
    )

    tables = {
        "Agg_Score_Disp": score_disp,
        "Agg_Ageing": ageing,
        "Agg_MS_MD": ms_md,
        "Agg_Slot": slot,
    }
    return {name: tidy_numbers(t) for name, t in tables.items()}


def tidy_numbers(table):
    """Store whole-valued float columns as integers so they are written as clean numbers."""
    for c in table.columns:
        if pd.api.types.is_float_dtype(table[c]):
            vals = table[c].dropna()
            if (vals % 1 == 0).all():
                table[c] = table[c].fillna(0).astype("int64")
    return table


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
def validate(df, leads, inst, tables):
    n_leads, n_inst = len(leads), len(inst)
    totals = {
        "raw rows": len(df),
        "distinct leads": n_leads,
        "MS instances": n_inst,
        "Agg_Score_Disp Leads": int(tables["Agg_Score_Disp"]["Leads"].sum()),
        "Agg_Ageing Leads (MS)": int(
            tables["Agg_Ageing"].loc[tables["Agg_Ageing"]["Measure"] == "MS", "Leads"].sum()
        ),
        "Agg_Ageing Leads (MD)": int(
            tables["Agg_Ageing"].loc[tables["Agg_Ageing"]["Measure"] == "MD", "Leads"].sum()
        ),
        "Agg_MS_MD MS Instances": int(tables["Agg_MS_MD"]["MS Instances"].sum()),
        "Agg_Slot Unique MS": int(tables["Agg_Slot"]["Unique MS"].sum()),
    }
    total_cells = sum(t.shape[0] * t.shape[1] for t in tables.values())
    totals["total cells to write"] = total_cells

    log.info("Validation totals:")
    for k, v in totals.items():
        log.info("  %-28s %s", k, f"{v:,}")
        print(f"VALIDATION {k}: {v}")

    failures = []
    if totals["raw rows"] <= 0:
        failures.append("raw rows must be > 0")
    if totals["Agg_Score_Disp Leads"] != n_leads:
        failures.append("sum(Agg_Score_Disp Leads) != len(leads)")
    if totals["Agg_Ageing Leads (MS)"] != n_leads:
        failures.append("sum(Agg_Ageing Leads) for Measure=MS != len(leads)")
    if totals["Agg_Ageing Leads (MD)"] != n_leads:
        failures.append("sum(Agg_Ageing Leads) for Measure=MD != len(leads)")
    if totals["Agg_MS_MD MS Instances"] != n_inst:
        failures.append("sum(Agg_MS_MD MS Instances) != len(inst)")
    if totals["Agg_Slot Unique MS"] != n_inst:
        failures.append("sum(Agg_Slot Unique MS) != len(inst)")
    if total_cells > MAX_TOTAL_CELLS:
        failures.append(
            f"summary tabs too large ({total_cells:,} cells > {MAX_TOTAL_CELLS:,}); "
            "check dimension cardinality"
        )

    if failures:
        for f in failures:
            log.error("VALIDATION FAILED: %s", f)
        log.error("Aborting. Spreadsheet left untouched.")
        sys.exit(1)
    log.info("All validations passed.")


# ----------------------------------------------------------------------------
# Google Sheets
# ----------------------------------------------------------------------------
def py_value(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def frame_to_values(table):
    rows = [[py_value(v) for v in r] for r in table.itertuples(index=False, name=None)]
    return [list(table.columns)] + rows


def write_tab(spreadsheet, name, values):
    try:
        ws = spreadsheet.worksheet(name)
        log.info("Tab '%s' exists; clearing.", name)
    except gspread.WorksheetNotFound:
        log.info("Tab '%s' missing; creating.", name)
        ws = spreadsheet.add_worksheet(title=name, rows=1, cols=1)

    n_rows = max(len(values), 1)
    n_cols = max(max((len(r) for r in values), default=1), 1)

    with_retries(ws.clear, f"clear {name}", WRITE_RETRIES, base_delay=5)
    with_retries(
        lambda: ws.resize(rows=n_rows, cols=n_cols),
        f"resize {name}",
        WRITE_RETRIES,
        base_delay=5,
    )

    for start in range(0, len(values), WRITE_CHUNK_ROWS):
        chunk = values[start : start + WRITE_CHUNK_ROWS]
        with_retries(
            lambda chunk=chunk, start=start: ws.update(
                values=chunk,
                range_name=f"A{start + 1}",
                value_input_option="RAW",
            ),
            f"write {name} rows {start + 1}-{start + len(chunk)}",
            WRITE_RETRIES,
            base_delay=5,
        )
    log.info("Wrote tab '%s': %d rows x %d cols.", name, len(values), n_cols)


def open_spreadsheet(cfg):
    info = json.loads(cfg["GCP_SERVICE_ACCOUNT_JSON"])
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    client = gspread.authorize(creds)
    return with_retries(
        lambda: client.open_by_key(cfg["SPREADSHEET_ID"]),
        "open spreadsheet",
        WRITE_RETRIES,
        base_delay=5,
    )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    cfg = read_config()

    log.info("Step 1/5: fetching data from Metabase")
    df = fetch_dataframe(cfg)
    log.info("Fetched %s rows x %d columns.", f"{len(df):,}", df.shape[1])
    if len(df) == 0:
        log.error("VALIDATION FAILED: raw rows must be > 0. Aborting.")
        sys.exit(1)

    log.info("Step 2/5: cleaning and computing bands")
    df = clean(df)
    df = add_score_band(df)

    leads = df.drop_duplicates("Lead ID").copy()
    inst = df[df["MS Instance Key"].notna()].copy()
    inst = inst.rename(columns={"MS Weekday/Weekend": "Day Type"})
    inst["Day Type"] = inst["Day Type"].fillna("Unknown")
    inst["MS Slot"] = inst["MS Slot"].fillna("Unknown")
    log.info("Distinct leads: %s | MS instances: %s", f"{len(leads):,}", f"{len(inst):,}")

    log.info("Step 3/5: building summary tables")
    tables = build_tables(leads, inst)
    for name, t in tables.items():
        log.info("  %s: %s rows x %d cols", name, f"{len(t):,}", t.shape[1])

    log.info("Step 4/5: validating")
    validate(df, leads, inst, tables)

    refresh_log = [
        ["Last refreshed (IST)", datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")],
        ["Raw rows", int(len(df))],
        ["Distinct leads", int(len(leads))],
        ["MS instances", int(len(inst))],
        ["Unmapped Bucket 2.0 leads", int((leads["Bucket 2.0"] == UNMAPPED_LABEL).sum())],
        [
            "MD before schedule check rows",
            int((inst["MD Before Schedule Data Check"] == 1).sum()),
        ],
        ["Unscored leads", int((leads["Score Band"] == "Unscored").sum())],
    ]

    log.info("Step 5/5: writing to Google Sheets")
    spreadsheet = open_spreadsheet(cfg)
    for name, table in tables.items():
        write_tab(spreadsheet, name, frame_to_values(table))
    write_tab(spreadsheet, "Refresh_Log", refresh_log)

    log.info("Done. Refresh_Log: %s", {k: v for k, v in refresh_log})


if __name__ == "__main__":
    main()
