#!/usr/bin/env python3
"""Daily pipeline: Metabase saved SQL question (month by month) -> pandas aggregation
-> small summary tabs in Google Sheets.

Raw rows are never written to Google Sheets (cell limit). Only the 4 aggregated tabs
and Refresh_Log are written. Configuration comes from environment variables only.
If any month fails to download or any validation fails, the sheet is left untouched.
"""
import json
import logging
import os
import sys
import time
from datetime import date, datetime
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

IST = ZoneInfo("Asia/Kolkata")
DEFAULT_START_MONTH = "2026-03-01"

REQUEST_TIMEOUT_S = 600
FETCH_RETRIES = 3
FETCH_BASE_DELAY_S = 15
WRITE_RETRIES = 3
WRITE_BASE_DELAY_S = 5
WRITE_CHUNK_ROWS = 5000

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
class FatalFetchError(RuntimeError):
    """Client-side error (HTTP 4xx): retrying cannot help."""


def with_retries(fn, what, retries, base_delay):
    """Run fn(); on failure retry up to `retries` times with exponential backoff."""
    attempt = 0
    while True:
        try:
            return fn()
        except FatalFetchError as exc:
            log.error("%s failed (not retryable): %s", what, exc)
            raise
        except Exception as exc:  # noqa: BLE001
            if attempt >= retries:
                log.error("%s failed after %d retries: %s", what, retries, exc)
                raise
            delay = base_delay * (2 ** attempt)
            attempt += 1
            log.warning("%s failed (%s). Retry %d/%d in %ds.", what, exc, attempt, retries, delay)
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
    cfg["START_MONTH"] = os.environ.get("START_MONTH", "").strip() or DEFAULT_START_MONTH
    return cfg


def parse_start_month(text):
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            return datetime.strptime(text, fmt).date().replace(day=1)
        except ValueError:
            continue
    log.error("START_MONTH '%s' is not valid. Use YYYY-MM-DD or YYYY-MM.", text)
    sys.exit(2)


def next_month(d):
    return date(d.year + 1, 1, 1) if d.month == 12 else date(d.year, d.month + 1, 1)


def month_windows(start, today):
    """Yield (from_date inclusive, to_date exclusive) for every month start..current month."""
    current = date(today.year, today.month, 1)
    if start > current:
        log.error("START_MONTH %s is after the current month %s.", start, current)
        sys.exit(2)
    d = start
    while d <= current:
        yield d, next_month(d)
        d = next_month(d)


# ----------------------------------------------------------------------------
# Fetch (month by month)
# ----------------------------------------------------------------------------
def short_error_text(resp):
    if resp.status_code == 524:
        return "Cloudflare timeout (524): this month's query took longer than ~100 s"
    text = " ".join(resp.text.split())
    return text[:300]


def get_tag_ids(cfg):
    """Read the real template-tag ids of the saved question (falls back to the tag name)."""
    url = f"{cfg['METABASE_URL']}/api/card/{cfg['METABASE_CARD_ID']}"
    ids = {"from_date": "from_date", "to_date": "to_date"}
    try:
        resp = requests.get(url, headers={"x-api-key": cfg["METABASE_API_KEY"]}, timeout=60)
        if resp.status_code != 200:
            log.warning("Could not read card definition (HTTP %s); using tag names as ids.", resp.status_code)
            return ids
        tags = resp.json()["dataset_query"]["native"]["template-tags"]
        for name in ids:
            if name not in tags:
                log.error(
                    "Template tag '%s' not found in question %s. Found tags: %s",
                    name, cfg["METABASE_CARD_ID"], sorted(tags),
                )
                sys.exit(7)
            ids[name] = tags[name].get("id") or name
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not parse card definition (%s); using tag names as ids.", exc)
    log.info("Template tag ids: %s", ids)
    return ids


def fetch_month(cfg, from_date, to_date):
    url = f"{cfg['METABASE_URL']}/api/card/{cfg['METABASE_CARD_ID']}/query/csv"
    tag_ids = cfg["TAG_IDS"]
    parameters = [
        {
            "id": tag_ids["from_date"],
            "type": "date/single",
            "value": from_date.strftime("%Y-%m-%d"),
            "target": ["variable", ["template-tag", "from_date"]],
        },
        {
            "id": tag_ids["to_date"],
            "type": "date/single",
            "value": to_date.strftime("%Y-%m-%d"),
            "target": ["variable", ["template-tag", "to_date"]],
        },
    ]
    needed = set(NEEDED_COLUMNS)

    def _once():
        with requests.post(
            url,
            headers={"x-api-key": cfg["METABASE_API_KEY"]},
            data={"format_rows": "false", "parameters": json.dumps(parameters)},
            stream=True,
            timeout=REQUEST_TIMEOUT_S,
        ) as resp:
            if resp.status_code != 200:
                msg = f"Metabase returned HTTP {resp.status_code}: {short_error_text(resp)}"
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    raise FatalFetchError(msg)
                raise RuntimeError(msg)
            resp.raw.decode_content = True
            return pd.read_csv(
                resp.raw,
                dtype=str,
                usecols=lambda c: c in needed,  # memory-safe: only needed columns
                keep_default_na=False,
                na_values=[""],
                encoding="utf-8-sig",
            )

    label = from_date.strftime("%Y-%m")
    df = with_retries(_once, f"Metabase fetch {label}", FETCH_RETRIES, FETCH_BASE_DELAY_S)

    missing = [c for c in NEEDED_COLUMNS if c not in df.columns]
    if missing:
        log.error("Month %s: required columns missing from Metabase result: %s", label, missing)
        sys.exit(3)
    return df[NEEDED_COLUMNS]


def fetch_all(cfg):
    start = parse_start_month(cfg["START_MONTH"])
    cfg["TAG_IDS"] = get_tag_ids(cfg)
    today = datetime.now(IST).date()
    frames = []
    for from_date, to_date in month_windows(start, today):
        label = from_date.strftime("%Y-%m")
        t0 = time.time()
        try:
            part = fetch_month(cfg, from_date, to_date)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            log.error("Month %s could not be fetched (%s). Aborting; sheet untouched.", label, exc)
            sys.exit(4)
        log.info("Month %s: %s rows in %.1f s", label, f"{len(part):,}", time.time() - t0)
        frames.append(part)

    df = pd.concat(frames, ignore_index=True)
    before = len(df)
    # Safety net: a lead belongs to exactly one creation month.
    df = df.drop_duplicates(["Lead ID", "MS Instance Key"], keep="first").reset_index(drop=True)
    if len(df) != before:
        log.warning("Dropped %s duplicate (Lead ID, MS Instance Key) rows.", f"{before - len(df):,}")
    return df


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
    band = pd.cut(df["Heuristic Score"], bins=bins, labels=score_band_labels(), right=False).astype(object)
    unscored = (df["Score Status"] == "Unscored") | df["Heuristic Score"].isna()
    # Scores below the first lower bound fall out of pd.cut as NaN -> also "Unscored"
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
    choices = ["Not done", "Same day", "D+1", "D+2", "D+3", "D+4", "D+5", "D+6 and above"]
    return pd.Series(np.select(conditions, choices, default="Not done"), index=days.index)


# ----------------------------------------------------------------------------
# Summary tables
# ----------------------------------------------------------------------------
def count_table(frame, group_cols, metric_name):
    return (
        frame.groupby(group_cols, as_index=False, sort=True, dropna=False)
        .size()
        .rename(columns={"size": metric_name})
    )


def tidy_numbers(table):
    """Store whole-valued float columns as integers so they are written as clean numbers."""
    for c in table.columns:
        if pd.api.types.is_float_dtype(table[c]):
            vals = table[c].dropna()
            if (vals % 1 == 0).all():
                table[c] = table[c].fillna(0).astype("int64")
    return table


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
    slot = inst.groupby(DIMS + ["MS Slot", "Day Type"], as_index=False, sort=True, dropna=False).agg(
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


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
def validate(df, leads, inst, tables):
    n_leads, n_inst = len(leads), len(inst)
    ageing = tables["Agg_Ageing"]
    totals = {
        "raw rows": len(df),
        "distinct leads": n_leads,
        "MS instances": n_inst,
        "Agg_Score_Disp Leads": int(tables["Agg_Score_Disp"]["Leads"].sum()),
        "Agg_Ageing Leads (MS)": int(ageing.loc[ageing["Measure"] == "MS", "Leads"].sum()),
        "Agg_Ageing Leads (MD)": int(ageing.loc[ageing["Measure"] == "MD", "Leads"].sum()),
        "Agg_MS_MD MS Instances": int(tables["Agg_MS_MD"]["MS Instances"].sum()),
        "Agg_Slot Unique MS": int(tables["Agg_Slot"]["Unique MS"].sum()),
    }

    log.info("Validation totals:")
    for k, v in totals.items():
        log.info("  %-26s %s", k, f"{v:,}")
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


def open_spreadsheet(cfg):
    try:
        info = json.loads(cfg["GCP_SERVICE_ACCOUNT_JSON"])
    except json.JSONDecodeError as exc:
        log.error("GCP_SERVICE_ACCOUNT_JSON is not valid JSON: %s", exc)
        sys.exit(5)
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    client = gspread.authorize(creds)
    return with_retries(
        lambda: client.open_by_key(cfg["SPREADSHEET_ID"]),
        "open spreadsheet",
        WRITE_RETRIES,
        WRITE_BASE_DELAY_S,
    )


def write_tab(spreadsheet, name, values):
    try:
        ws = spreadsheet.worksheet(name)
        log.info("Tab '%s' exists; clearing.", name)
    except gspread.WorksheetNotFound:
        log.info("Tab '%s' missing; creating.", name)
        ws = spreadsheet.add_worksheet(title=name, rows=1, cols=1)

    n_rows = max(len(values), 1)
    n_cols = max(max((len(r) for r in values), default=1), 1)

    with_retries(ws.clear, f"clear {name}", WRITE_RETRIES, WRITE_BASE_DELAY_S)
    with_retries(
        lambda: ws.resize(rows=n_rows, cols=n_cols),
        f"resize {name}",
        WRITE_RETRIES,
        WRITE_BASE_DELAY_S,
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
            WRITE_BASE_DELAY_S,
        )
    log.info("Wrote tab '%s': %d rows x %d cols.", name, len(values), n_cols)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def run():
    cfg = read_config()

    log.info("Step 1/5: fetching data from Metabase month by month (from %s)", cfg["START_MONTH"])
    df = fetch_all(cfg)
    log.info("Fetched %s rows x %d columns in total.", f"{len(df):,}", df.shape[1])
    if len(df) == 0:
        log.error("VALIDATION FAILED: raw rows must be > 0. Aborting; sheet untouched.")
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
        ["Last refreshed (IST)", datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")],
        ["Raw rows", int(len(df))],
        ["Distinct leads", int(len(leads))],
        ["MS instances", int(len(inst))],
        ["Unmapped Bucket 2.0 leads", int((leads["Bucket 2.0"] == UNMAPPED_LABEL).sum())],
        ["MD before schedule check rows", int((inst["MD Before Schedule Data Check"] == 1).sum())],
        ["Unscored leads", int((leads["Score Band"] == "Unscored").sum())],
    ]

    log.info("Step 5/5: writing to Google Sheets")
    spreadsheet = open_spreadsheet(cfg)
    for name, table in tables.items():
        write_tab(spreadsheet, name, frame_to_values(table))
    write_tab(spreadsheet, "Refresh_Log", refresh_log)

    log.info("Done. Refresh_Log: %s", {k: v for k, v in refresh_log})


def main():
    try:
        run()
    except SystemExit:
        raise
    except gspread.exceptions.APIError as exc:
        log.error("Google Sheets API error: %s", exc)
        sys.exit(6)
    except Exception as exc:  # noqa: BLE001
        log.exception("Unexpected failure: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
