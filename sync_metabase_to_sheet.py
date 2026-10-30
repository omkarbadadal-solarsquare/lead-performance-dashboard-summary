#!/usr/bin/env python3
"""Daily pipeline: THREE saved Metabase SQL questions (window by window) -> merge + pandas
aggregation -> small summary tabs in Google Sheets.

Raw rows are never written to Google Sheets (cell limit). Only the 4 aggregated tabs and
Refresh_Log are written. Configuration comes from environment variables only.
If any window fails to download or any validation fails, the sheet is left untouched.
"""
import io
import json
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
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
DEFAULT_START_DATE = "2026-03-01"

REQUEST_TIMEOUT_S = 300
FETCH_MAX_RETRIES = 2
FETCH_BASE_DELAY_S = 10
WRITE_RETRIES = 3
WRITE_BASE_DELAY_S = 5
WRITE_CHUNK_ROWS = 5000

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

TAGS = ["from_date", "to_date"]

CARDS = {
    "LEADS": {
        "env": "METABASE_CARD_ID_LEADS",
        "columns": [
            "Lead ID",
            "Lead Creation Cohort Month",
            "City",
            "Channel",
            "LRM",
            "Heuristic Score",
            "Score Status",
            "Bucket 2.0",
            "Order Timestamp IST",
        ],
    },
    "MEETINGS": {
        "env": "METABASE_CARD_ID_MEETINGS",
        "columns": [
            "Lead ID",
            "Days Lead to First MS",
            "Days Lead to First MD",
            "MS Instance Key",
            "MS Date",
            "MS Timestamp IST",
            "MS Weekday/Weekend",
            "MS Slot",
            "Schedule Records on MS Date",
            "MD Timestamp IST",
            "Days MS to MD",
            "MD Before Schedule Data Check",
        ],
    },
    "CONF": {
        "env": "METABASE_CARD_ID_CONF",
        "columns": ["Lead ID", "Confirmed At IST"],
    },
}

DIMS = ["City", "Channel", "LRM", "Cohort Month"]

AGEING_BINS = [-np.inf, 1, 3, 7, 15, 22, 30, 60, np.inf]
AGEING_LABELS = ["D+1", "D+3", "D+7", "D+15", "D+22", "D+30", "D+60", "60+"]

UNMAPPED_LABEL = "UNMAPPED - REVIEW"


# ----------------------------------------------------------------------------
# Errors and retry helpers
# ----------------------------------------------------------------------------
class FatalFetchError(RuntimeError):
    """Client-side error (HTTP 4xx): retrying cannot help."""


class CloudflareTimeout(RuntimeError):
    """HTTP 524: Cloudflare gave up waiting for Metabase."""


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
            log.warning("%s failed (%s). Retry %d/%d in %ds.", what, exc, attempt, retries, delay)
            time.sleep(delay)


# ----------------------------------------------------------------------------
# Config and windows
# ----------------------------------------------------------------------------
def read_config():
    required = [
        "METABASE_URL",
        "METABASE_API_KEY",
        "METABASE_CARD_ID_LEADS",
        "METABASE_CARD_ID_MEETINGS",
        "METABASE_CARD_ID_CONF",
        "SPREADSHEET_ID",
        "GCP_SERVICE_ACCOUNT_JSON",
    ]
    cfg = {n: os.environ.get(n, "").strip() for n in required}
    missing = [n for n, v in cfg.items() if not v]
    if missing:
        log.error("Missing environment variables: %s", missing)
        sys.exit(2)
    cfg["METABASE_URL"] = cfg["METABASE_URL"].rstrip("/")
    cfg["START_DATE"] = os.environ.get("START_DATE", "").strip() or DEFAULT_START_DATE

    raw_chunk = os.environ.get("CHUNK_DAYS", "").strip()
    cfg["CHUNK_DAYS"] = None
    if raw_chunk:
        try:
            cfg["CHUNK_DAYS"] = int(raw_chunk)
        except ValueError:
            cfg["CHUNK_DAYS"] = 0
        if cfg["CHUNK_DAYS"] <= 0:
            log.error("CHUNK_DAYS '%s' is not a positive whole number.", raw_chunk)
            sys.exit(2)
    return cfg


def parse_start_date(text):
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    log.error("START_DATE '%s' is not valid. Use YYYY-MM-DD or YYYY-MM.", text)
    sys.exit(2)


def next_month(d):
    return date(d.year + 1, 1, 1) if d.month == 12 else date(d.year, d.month + 1, 1)


def build_windows(start, tomorrow, chunk_days):
    """(from inclusive, to exclusive) windows from start up to tomorrow (IST)."""
    if start >= tomorrow:
        log.error("START_DATE %s is not before tomorrow (%s).", start, tomorrow)
        sys.exit(2)
    windows = []
    d = start
    while d < tomorrow:
        if chunk_days:
            end = min(d + timedelta(days=chunk_days), tomorrow)
        else:
            end = min(next_month(d), tomorrow)
        windows.append((d, end))
        d = end
    return windows


# ----------------------------------------------------------------------------
# Fetch
# ----------------------------------------------------------------------------
def short_error_text(resp):
    return " ".join(resp.text.split())[:300]


def get_tag_ids(cfg, card_id):
    """Read the real parameter ids of a saved question (Metabase requires an id per parameter)."""
    ids = {name: name for name in TAGS}
    url = f"{cfg['METABASE_URL']}/api/card/{card_id}"
    try:
        resp = requests.get(url, headers={"x-api-key": cfg["METABASE_API_KEY"]}, timeout=60)
        if resp.status_code != 200:
            log.warning("Card %s: could not read definition (HTTP %s); using tag names as ids.",
                        card_id, resp.status_code)
            return ids
        card = resp.json()
        tags = ((card.get("dataset_query") or {}).get("native") or {}).get("template-tags") or {}
        for name in TAGS:
            if name in tags:
                ids[name] = tags[name].get("id") or name
                continue
            param = next(
                (p for p in (card.get("parameters") or [])
                 if p.get("slug") == name or p.get("name") == name),
                None,
            )
            if param and param.get("id"):
                ids[name] = param["id"]
            elif tags:
                log.error("Card %s: template tag '%s' not found. Found tags: %s",
                          card_id, name, sorted(tags))
                sys.exit(7)
            else:
                log.warning("Card %s: no template tags found in definition; using tag name '%s' as id.",
                            card_id, name)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        log.warning("Card %s: could not parse definition (%s); using tag names as ids.", card_id, exc)
    log.info("Card %s parameter ids: %s", card_id, ids)
    return ids


def blank_to_nan(df):
    """Blank / whitespace-only strings -> NaN."""
    for c in df.columns:
        df[c] = df[c].map(lambda v: (v.strip() or np.nan) if isinstance(v, str) else v)
    return df


def fetch_window(cfg, key, from_d, to_d, tag_ids):
    spec = CARDS[key]
    card_id = cfg[spec["env"]]
    url = f"{cfg['METABASE_URL']}/api/card/{card_id}/query/csv"
    parameters = [
        {
            "id": tag_ids["from_date"],
            "type": "date/single",
            "value": from_d.strftime("%Y-%m-%d"),
            "target": ["variable", ["template-tag", "from_date"]],
        },
        {
            "id": tag_ids["to_date"],
            "type": "date/single",
            "value": to_d.strftime("%Y-%m-%d"),
            "target": ["variable", ["template-tag", "to_date"]],
        },
    ]
    needed = set(spec["columns"])

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
                if resp.status_code == 524:
                    raise CloudflareTimeout(msg)
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    raise FatalFetchError(msg)
                raise RuntimeError(msg)
            # Download the body in 1 MB blocks. A dropped connection raises a clear
            # network error here instead of a confusing "I/O operation on closed file".
            buf = io.BytesIO()
            for block in resp.iter_content(chunk_size=1 << 20):
                buf.write(block)
        buf.seek(0)
        try:
            return pd.read_csv(
                buf,
                dtype=str,
                usecols=lambda c: c in needed,  # memory-safe: only this card's columns
                keep_default_na=False,
                na_values=[""],
                encoding="utf-8-sig",
            )
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=spec["columns"], dtype=str)

    what = f"{key} {from_d} -> {to_d}"
    attempt = 0
    while True:
        try:
            df = _once()
            break
        except FatalFetchError as exc:
            log.error("%s failed (not retryable): %s", what, exc)
            raise
        except CloudflareTimeout as exc:
            log.warning("Cloudflare 100 s limit: set CHUNK_DAYS smaller (for example 7)")
            limit, err = 1, exc
        except Exception as exc:  # noqa: BLE001
            limit, err = FETCH_MAX_RETRIES, exc
        if attempt >= limit:
            log.error("%s failed after %d retries: %s", what, attempt, err)
            raise err
        delay = FETCH_BASE_DELAY_S * (2 ** attempt)
        attempt += 1
        log.warning("%s failed (%s). Retry %d/%d in %ds.", what, err, attempt, limit, delay)
        time.sleep(delay)

    missing = [c for c in spec["columns"] if c not in df.columns]
    if missing:
        log.error("%s: required columns missing from Metabase result: %s", key, missing)
        sys.exit(3)
    return blank_to_nan(df[spec["columns"]].copy())


def dedupe_keep_last(df, subset, name):
    df = df[df["Lead ID"].notna()] if df["Lead ID"].isna().any() else df
    before = len(df)
    out = df.drop_duplicates(subset, keep="last").reset_index(drop=True)
    log.info("%s: %s rows after concat, %s after dedupe on %s", name, f"{before:,}", f"{len(out):,}", subset)
    return out


def fetch_all(cfg):
    start = parse_start_date(cfg["START_DATE"])
    tomorrow = datetime.now(IST).date() + timedelta(days=1)
    windows = build_windows(start, tomorrow, cfg["CHUNK_DAYS"])
    log.info(
        "%d windows (%s) from %s up to %s (exclusive), 3 cards each.",
        len(windows),
        f"{cfg['CHUNK_DAYS']}-day chunks" if cfg["CHUNK_DAYS"] else "calendar months",
        start,
        tomorrow,
    )

    tag_ids = {key: get_tag_ids(cfg, cfg[spec["env"]]) for key, spec in CARDS.items()}
    frames = {key: [] for key in CARDS}
    for from_d, to_d in windows:
        for key in CARDS:
            t0 = time.time()
            try:
                part = fetch_window(cfg, key, from_d, to_d, tag_ids[key])
            except SystemExit:
                raise
            except Exception as exc:  # noqa: BLE001
                log.error("Window %s -> %s, card %s could not be fetched (%s). Aborting; sheet untouched.",
                          from_d, to_d, key, exc)
                sys.exit(4)
            log.info("Window %s -> %s | %-8s %s rows in %.1f s",
                     from_d, to_d, key, f"{len(part):,}", time.time() - t0)
            frames[key].append(part)

    # A later window's row wins (a lead's latest record).
    leads_raw = dedupe_keep_last(pd.concat(frames["LEADS"], ignore_index=True), ["Lead ID"], "LEADS")
    meet_raw = dedupe_keep_last(
        pd.concat(frames["MEETINGS"], ignore_index=True), ["Lead ID", "MS Instance Key"], "MEETINGS"
    )
    conf_raw = dedupe_keep_last(
        pd.concat(frames["CONF"], ignore_index=True), ["Lead ID", "Confirmed At IST"], "CONF"
    )
    return leads_raw, meet_raw, conf_raw


# ----------------------------------------------------------------------------
# Parsing helpers
# ----------------------------------------------------------------------------
def parse_naive_dt(series):
    """Parse to naive datetime64[ns] (source values are already IST)."""
    try:
        dt = pd.to_datetime(series, errors="coerce", format="mixed")
        if dt.dt.tz is not None:
            dt = dt.dt.tz_localize(None)
    except (ValueError, TypeError, AttributeError):
        dt = pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(IST).dt.tz_localize(None)
    return dt.astype("datetime64[ns]")


def to_num(series):
    return pd.to_numeric(series, errors="coerce")


# ----------------------------------------------------------------------------
# Bands
# ----------------------------------------------------------------------------
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
# Build lead and instance frames
# ----------------------------------------------------------------------------
def prepare_meetings(meet_raw):
    meet = meet_raw.copy()
    for c in ["Days Lead to First MS", "Days Lead to First MD", "Schedule Records on MS Date",
              "Days MS to MD", "MD Before Schedule Data Check"]:
        meet[c] = to_num(meet[c])
    for c in ["MS Date", "MS Timestamp IST", "MD Timestamp IST"]:
        meet[c] = parse_naive_dt(meet[c])
    return meet


def prepare_leads(leads_raw, meet):
    leads = leads_raw.copy()
    leads["Heuristic Score"] = to_num(leads["Heuristic Score"])
    leads["Order Timestamp IST"] = parse_naive_dt(leads["Order Timestamp IST"])
    leads["Cohort Month"] = parse_naive_dt(leads["Lead Creation Cohort Month"]).dt.strftime("%Y-%m")
    for c in DIMS:
        leads[c] = leads[c].fillna("Unknown")
    leads["Bucket 2.0"] = leads["Bucket 2.0"].fillna("Unknown")

    lead_days = meet.drop_duplicates("Lead ID", keep="first")[
        ["Lead ID", "Days Lead to First MS", "Days Lead to First MD"]
    ]
    leads = leads.merge(lead_days, on="Lead ID", how="left")
    return add_score_band(leads)


def prepare_instances(meet, leads):
    inst = meet[meet["MS Instance Key"].notna()].copy().reset_index(drop=True)
    lead_dims = leads[["Lead ID"] + DIMS + ["Order Timestamp IST"]]
    inst = inst.merge(lead_dims, on="Lead ID", how="left", indicator=True)
    orphans = int((inst["_merge"] == "left_only").sum())
    inst = inst.drop(columns="_merge").reset_index(drop=True)
    log.info("MS instances whose lead was missing from LEADS: %s", f"{orphans:,}")
    for c in DIMS:
        inst[c] = inst[c].fillna("Unknown")
    inst = inst.rename(columns={"MS Weekday/Weekend": "Day Type"})
    inst["Day Type"] = inst["Day Type"].fillna("Unknown")
    inst["MS Slot"] = inst["MS Slot"].fillna("Unknown")
    return inst


def compute_rtm(inst, conf_raw):
    """RTM Flag per instance using the latest confirmation before the end of the MS date."""
    rtm = pd.Series(0, index=inst.index, dtype="int64")

    conf = conf_raw.copy()
    conf["Confirmed At IST"] = parse_naive_dt(conf["Confirmed At IST"])
    conf = conf.dropna(subset=["Lead ID", "Confirmed At IST"])
    conf = conf.sort_values(["Lead ID", "Confirmed At IST"], kind="mergesort")
    conf["prev_conf"] = conf.groupby("Lead ID")["Confirmed At IST"].shift(1)
    conf["latest_conf"] = conf["Confirmed At IST"]
    right = conf.rename(columns={"Confirmed At IST": "conf_key"})[
        ["Lead ID", "conf_key", "latest_conf", "prev_conf"]
    ].copy()
    right["Lead ID"] = right["Lead ID"].astype(str)
    right = right.sort_values("conf_key", kind="mergesort")

    left = inst.loc[inst["MS Date"].notna() & inst["Lead ID"].notna(), ["Lead ID", "MS Date"]].copy()
    if left.empty or right.empty:
        return rtm
    left["Lead ID"] = left["Lead ID"].astype(str)
    left["_idx"] = left.index
    left["ms_key"] = left["MS Date"] + pd.Timedelta(days=1)
    left = left.sort_values("ms_key", kind="mergesort")

    m = pd.merge_asof(
        left,
        right,
        left_on="ms_key",
        right_on="conf_key",
        by="Lead ID",
        direction="backward",
        allow_exact_matches=False,
    )

    latest, prev = m["latest_conf"], m["prev_conf"]
    has_latest = latest.notna()
    same_day = latest.dt.normalize() == m["MS Date"].dt.normalize()
    prev_was_yesterday = prev.notna() & (
        prev.dt.normalize() == latest.dt.normalize() - pd.Timedelta(days=1)
    )
    flag = (has_latest & same_day & ~prev_was_yesterday).astype("int64")
    rtm.loc[m["_idx"].to_numpy()] = flag.to_numpy()
    return rtm


def compute_order_attributed(inst):
    """Exactly one instance per lead (latest MD, tie: latest MS) gets the order."""
    flag = pd.Series(0, index=inst.index, dtype="int64")
    cand = inst[
        inst["MD Timestamp IST"].notna()
        & inst["Order Timestamp IST"].notna()
        & (inst["Order Timestamp IST"] >= inst["MD Timestamp IST"])
    ]
    if cand.empty:
        return flag
    cand = cand.sort_values(
        ["Lead ID", "MD Timestamp IST", "MS Timestamp IST", "MS Instance Key"],
        na_position="first",
        kind="mergesort",
    )
    chosen = cand.groupby("Lead ID", sort=False).tail(1).index
    flag.loc[chosen] = 1
    return flag


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
def validate(leads, inst, tables):
    n_leads, n_inst = len(leads), len(inst)
    ageing = tables["Agg_Ageing"]
    totals = {
        "leads rows": n_leads,
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
    if n_leads <= 0:
        failures.append("leads rows must be > 0")
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
    if isinstance(v, (np.floating, float)):
        return "" if np.isnan(v) else float(v)
    if v is None or v is pd.NA:
        return ""
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

    log.info("Step 1/6: fetching 3 Metabase questions window by window (from %s)", cfg["START_DATE"])
    leads_raw, meet_raw, conf_raw = fetch_all(cfg)

    log.info("Step 2/6: parsing and merging leads, meetings, confirmations")
    meet = prepare_meetings(meet_raw)
    leads = prepare_leads(leads_raw, meet)
    inst = prepare_instances(meet, leads)
    log.info("Leads: %s | MS instances: %s", f"{len(leads):,}", f"{len(inst):,}")
    if len(leads) == 0:
        log.error("VALIDATION FAILED: leads rows must be > 0. Aborting; sheet untouched.")
        sys.exit(1)

    log.info("Step 3/6: computing RTM, Valid MD and Order Attributed flags")
    inst["RTM Flag"] = compute_rtm(inst, conf_raw)
    inst["Valid MD Flag"] = inst["MD Timestamp IST"].notna().astype("int64")
    inst["Order Attributed Flag"] = compute_order_attributed(inst)
    log.info(
        "RTM instances: %s | MD done: %s | Orders attributed: %s",
        f"{int(inst['RTM Flag'].sum()):,}",
        f"{int(inst['Valid MD Flag'].sum()):,}",
        f"{int(inst['Order Attributed Flag'].sum()):,}",
    )

    log.info("Step 4/6: building summary tables")
    tables = build_tables(leads, inst)
    for name, t in tables.items():
        log.info("  %s: %s rows x %d cols", name, f"{len(t):,}", t.shape[1])

    log.info("Step 5/6: validating")
    validate(leads, inst, tables)

    refresh_log = [
        ["Last refreshed (IST)", datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")],
        ["Raw rows", int(len(leads) + len(meet))],
        ["Distinct leads", int(len(leads))],
        ["MS instances", int(len(inst))],
        ["Unmapped Bucket 2.0 leads", int((leads["Bucket 2.0"] == UNMAPPED_LABEL).sum())],
        ["MD before schedule check rows", int((inst["MD Before Schedule Data Check"] == 1).sum())],
        ["Unscored leads", int((leads["Score Band"] == "Unscored").sum())],
    ]

    log.info("Step 6/6: writing to Google Sheets")
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
