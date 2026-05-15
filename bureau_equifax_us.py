"""
bureau_equifax_us.py
====================
Equifax US Consumer Credit Report — parsing, raw tradeline metrics, and
canonical decisioning features.

All risk-pillar logic and helper utilities are imported directly from
bureau_uw.scoring_api (Dev Artifacts - US/src) so there is no duplication.
The canonical feature values (account counts, utilization buckets, balance
fields) are produced by the EquifaxJsonAdapter + build_features pipeline,
which maps each field to its specific Equifax attribute ID.
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical pipeline bootstrap  (Dev Artifacts - US/src)
# ---------------------------------------------------------------------------
_ARTIFACTS_SRC = Path(__file__).parent / "Dev Artifacts - US" / "src"
if _ARTIFACTS_SRC.is_dir() and str(_ARTIFACTS_SRC) not in sys.path:
    sys.path.insert(0, str(_ARTIFACTS_SRC))

try:
    from bureau_uw.parsers.json_to_dict import EquifaxJsonAdapter as _EquifaxJsonAdapter
    from bureau_uw.features.feature_engineering import build_features as _build_features
    from bureau_uw.scoring_api import (
        TIER_GUIDANCE,
        RISK_TO_NUM,
        _safe_float,
        pillar_credit_quality,
        pillar_exposure_pressure,
        pillar_conduct,
        pillar_stability,
    )
    _EQX_ADAPTER: Any = _EquifaxJsonAdapter()
    _CANONICAL_PIPELINE_OK = True
    print("[bureau_equifax_us] Canonical pipeline loaded OK.", file=sys.stderr)
except Exception as _canon_err:
    # Log the real reason so it appears in Render / server logs.
    print(
        f"[bureau_equifax_us] Canonical pipeline import FAILED — canonical metrics will be blank.\n"
        f"  Reason: {_canon_err}\n"
        f"  Traceback:\n{traceback.format_exc()}",
        file=sys.stderr,
    )
    # Graceful fallback — dashboard still runs but uses raw tradeline values.
    _EQX_ADAPTER = None
    _build_features = None
    _CANONICAL_PIPELINE_OK = False

    TIER_GUIDANCE: Dict[str, str] = {
        "Standard Review": "Bureau profile is consistent with expectations. Verify key details.",
        "Enhanced Review": "Some bureau signals warrant closer attention. Review flagged areas; refer if necessary.",
        "Detailed Review": "Multiple bureau signals require thorough assessment. Review all pillars and tradelines; refer after checks.",
    }
    RISK_TO_NUM: Dict[str, int] = {"Unknown": 0, "Low": 1, "Moderate": 2, "Elevated": 3}

    def _safe_float(x):
        try:
            return None if x is None else float(str(x).strip())
        except Exception:
            return None

    def _stub_pillar(*_):
        return "Unknown", ["Canonical pipeline unavailable"]

    pillar_credit_quality = pillar_exposure_pressure = pillar_conduct = pillar_stability = _stub_pillar


# ---------------------------------------------------------------------------
# Date helper
# ---------------------------------------------------------------------------

def _parse_date(raw: str) -> pd.Timestamp | None:
    if not raw or not isinstance(raw, str):
        return None
    raw = raw.strip()
    if len(raw) == 8:
        month, day, year = raw[:2], raw[2:4], raw[4:]
        if day == "00":
            try:
                return pd.Timestamp(datetime.strptime(f"{month}{year}", "%m%Y"))
            except ValueError:
                pass
        else:
            try:
                return pd.Timestamp(datetime.strptime(raw, "%m%d%Y"))
            except ValueError:
                pass
    if len(raw) == 6:
        try:
            return pd.Timestamp(datetime.strptime(raw, "%m%Y"))
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# Portfolio / narrative constants
# ---------------------------------------------------------------------------

DEBT_PORTFOLIO_CODES = {"I", "M", "C", "O", "R"}
REVOLVING_CODES      = {"R", "C", "O"}
CLOSED_CODES         = {"FA", "CF"}

ADVERSE_RATE_MAP = {"B": "Lost or Stolen Card"}
ADVERSE_NARRATIVE_MAP = {
    "RP": "Returned Payment",
    "CO": "Charge-Off",
    "CL": "Collection",
    "BC": "Bankruptcy",
    "CF": "Closed Adverse",
}


# ---------------------------------------------------------------------------
# Trade classifiers
# ---------------------------------------------------------------------------

def _is_active(trade: dict) -> bool:
    codes = {n["code"] for n in trade.get("narrativeCodes", [])}
    return len(codes & CLOSED_CODES) == 0


def _is_paying_as_agreed(trade: dict) -> bool:
    return trade.get("rate", {}).get("code") == 1


def _is_adverse(trade: dict) -> bool:
    codes = {n["code"] for n in trade.get("narrativeCodes", [])}
    rate  = trade.get("rateStatusCode", {}).get("code", "")
    return bool(codes & set(ADVERSE_NARRATIVE_MAP)) or rate in ADVERSE_RATE_MAP


def _classify_trade(trade: dict) -> dict:
    p = trade.get("portfolioTypeCode", {}).get("code", "")
    return {
        "is_debt":        p in DEBT_PORTFOLIO_CODES,
        "is_revolving":   p in REVOLVING_CODES,
        "is_mortgage":    p == "M",
        "is_installment": p == "I",
        "is_active":      _is_active(trade),
        "is_adverse":     _is_adverse(trade),
        "pays_as_agreed": _is_paying_as_agreed(trade),
    }


# ---------------------------------------------------------------------------
# Tradeline deduplication
# ---------------------------------------------------------------------------

_DEDUP_COLS = ["name", "account_type", "portfolio_code", "date_opened", "high_credit"]


def _dedup_trades(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    subset = [c for c in _DEDUP_COLS if c in df.columns]
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(
            f"[bureau_equifax_us] Deduplication removed {removed} duplicate "
            f"tradeline(s). Before: {before}, after: {len(df)}.",
            file=sys.stderr,
        )
    return df


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_bureau_json(source) -> dict:
    """
    Parse an Equifax US ResponseEssentials JSON.

    Returns: report, raw_payload, trades_df, inquiries_df, identity, model_score
    """
    if isinstance(source, str):
        with open(source) as f:
            raw = json.load(f)
    else:
        raw = source

    report = raw["consumers"]["equifaxUSConsumerCreditReport"][0]

    rows = []
    for trade in report.get("trades", []):
        flags = _classify_trade(trade)
        rows.append({
            "name":              trade.get("customerName", ""),
            "account_type":      trade.get("accountTypeCode", {}).get("description", ""),
            "portfolio_code":    trade.get("portfolioTypeCode", {}).get("code", ""),
            "designator":        trade.get("accountDesignator", {}).get("description", ""),
            "narratives":        [n["code"] for n in trade.get("narrativeCodes", [])],
            "date_opened":       _parse_date(trade.get("dateOpened", "")),
            "date_reported":     _parse_date(trade.get("dateReported", "")),
            "balance":           trade.get("balance") or 0,
            "high_credit":       trade.get("highCredit") or 0,
            "credit_limit":      trade.get("creditLimit") or 0,
            "scheduled_payment": trade.get("scheduledPaymentAmount") or 0,
            "actual_payment":    trade.get("actualPaymentAmount") or 0,
            "months_reviewed":   int(trade.get("monthsReviewed") or 0),
            **flags,
        })

    trades_df    = _dedup_trades(pd.DataFrame(rows))
    inquiries_df = pd.DataFrame([
        {
            "date":          _parse_date(inq.get("inquiryDate", "")),
            "name":          inq.get("customerName", ""),
            "type":          inq.get("type", ""),
            "industry_code": inq.get("industryCode", ""),
        }
        for inq in report.get("inquiries", [])
    ])

    name = report.get("subjectName", {})
    identity = {
        "first_name":          name.get("firstName", ""),
        "last_name":           name.get("lastName", ""),
        "full_name":           f"{name.get('firstName','')} {name.get('lastName','')}".strip(),
        "dob":                 _parse_date(report.get("birthDate", "")),
        "ssn":                 report.get("subjectSocialNum", ""),
        "report_date":         _parse_date(report.get("reportDate", "")),
        "file_since":          _parse_date(report.get("fileSinceDate", "")),
        "last_activity":       _parse_date(report.get("lastActivityDate", "")),
        "hit_code":            report.get("hitCode", {}).get("description", ""),
        "address_discrepancy": report.get("addressDiscrepancyIndicator", "N") == "Y",
        "addresses":           report.get("addresses", []),
        "employments":         report.get("employments", []),
    }

    model_score = None
    for m in report.get("models", []):
        model_score = m.get("score")
        break

    return {
        "report":       report,
        "raw_payload":  raw,
        "trades_df":    trades_df,
        "inquiries_df": inquiries_df,
        "identity":     identity,
        "model_score":  model_score,
    }


# ---------------------------------------------------------------------------
# Adverse accounts
# ---------------------------------------------------------------------------

def get_bounced_equivalents(parsed: dict) -> pd.DataFrame:
    rows = []
    for trade in parsed["report"].get("trades", []):
        categories = []
        if trade.get("rateStatusCode", {}).get("code", "") in ADVERSE_RATE_MAP:
            categories.append(ADVERSE_RATE_MAP[trade["rateStatusCode"]["code"]])
        for n in trade.get("narrativeCodes", []):
            if n["code"] in ADVERSE_NARRATIVE_MAP:
                cat = ADVERSE_NARRATIVE_MAP[n["code"]]
                if cat not in categories:
                    categories.append(cat)
        if categories:
            rows.append({
                "name":            trade.get("customerName", ""),
                "account_type":    trade.get("accountTypeCode", {}).get("description", ""),
                "date_reported":   _parse_date(trade.get("dateReported", "")),
                "balance":         trade.get("balance", 0),
                "Bounce Category": ", ".join(categories),
                "narratives":      [n["description"] for n in trade.get("narrativeCodes", [])],
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Raw tradeline metrics  (supporting analysis only — not used for decisioning)
# ---------------------------------------------------------------------------

def calculate_metrics(parsed: dict) -> dict:
    trades    = parsed["trades_df"]
    inquiries = parsed["inquiries_df"]
    identity  = parsed["identity"]

    if trades.empty:
        return {}

    debt = trades[trades["is_debt"]]

    total_balance      = round(trades["balance"].sum(), 2)
    total_credit_limit = round(trades["credit_limit"].sum(), 2)
    total_high_credit  = round(trades["high_credit"].sum(), 2)
    unused_credit      = round(total_credit_limit - total_balance, 2)

    tw_limit         = trades[trades["credit_limit"] > 0]
    utilisation_rate = round(
        tw_limit["balance"].sum() / tw_limit["credit_limit"].sum() * 100
        if not tw_limit.empty else 0, 2,
    )

    total_debt           = round(debt["balance"].sum(), 2)
    total_scheduled_pmts = round(trades["scheduled_payment"].sum(), 2)
    total_actual_pmts    = round(trades["actual_payment"].sum(), 2)
    dscr = round(
        total_actual_pmts / total_scheduled_pmts if total_scheduled_pmts > 0 else 0, 2,
    )

    n_total       = len(trades)
    n_active      = int(trades["is_active"].sum())
    n_closed      = n_total - n_active
    n_mortgage    = int(trades["is_mortgage"].sum())
    n_installment = int(trades["is_installment"].sum())
    n_revolving   = int(trades["is_revolving"].sum())
    n_adverse     = int(trades["is_adverse"].sum())
    n_pays_agreed = int(trades["pays_as_agreed"].sum())

    valid_dates       = trades["date_opened"].dropna()
    credit_age_months = 0
    oldest_account    = None
    newest_account    = None
    if not valid_dates.empty:
        ref = identity.get("report_date") or pd.Timestamp.now()
        oldest_account    = valid_dates.min()
        newest_account    = valid_dates.max()
        credit_age_months = int((ref - oldest_account).days / 30.44)

    report_date_ref = identity.get("report_date") or pd.Timestamp.now()
    n_inq_total = len(inquiries)
    n_inq_3m = n_inq_6m = n_inq_12m = 0
    if not inquiries.empty and "date" in inquiries.columns:
        vi = inquiries.dropna(subset=["date"])
        n_inq_3m  = int((vi["date"] >= report_date_ref - pd.DateOffset(months=3)).sum())
        n_inq_6m  = int((vi["date"] >= report_date_ref - pd.DateOffset(months=6)).sum())
        n_inq_12m = int((vi["date"] >= report_date_ref - pd.DateOffset(months=12)).sum())

    bounced_df = get_bounced_equivalents(parsed)

    return {
        "Total Balance (All Accounts)":     total_balance,
        "Total Credit Limit":               total_credit_limit,
        "Total High Credit (Peak/Loans)":   total_high_credit,
        "Unused Credit":                    unused_credit,
        "Credit Utilisation (%)":           utilisation_rate,
        "Total Active Debt":                total_debt,
        "Total Scheduled Monthly Payments": total_scheduled_pmts,
        "Total Actual Payments Made":       total_actual_pmts,
        "Debt Service Coverage Ratio":      dscr,
        "Average Monthly Obligation":       round(total_scheduled_pmts, 2),
        "Total Accounts":                   n_total,
        "Active Accounts":                  n_active,
        "Closed Accounts":                  n_closed,
        "Mortgage Accounts":                n_mortgage,
        "Installment Accounts":             n_installment,
        "Revolving Accounts":               n_revolving,
        "Accounts Paying as Agreed":        n_pays_agreed,
        "Adverse Accounts":                 n_adverse,
        "Credit Age (Months)":              credit_age_months,
        "Oldest Account Opened":            str(oldest_account.date()) if oldest_account else None,
        "Newest Account Opened":            str(newest_account.date()) if newest_account else None,
        "Max Months of History":            int(trades["months_reviewed"].max()),
        "Total Inquiries":                  n_inq_total,
        "Inquiries (Last 3 Months)":        n_inq_3m,
        "Inquiries (Last 6 Months)":        n_inq_6m,
        "Inquiries (Last 12 Months)":       n_inq_12m,
        "Adverse Accounts (Bounced equiv)": len(bounced_df),
        "Address Discrepancy Flag":         identity.get("address_discrepancy", False),
        "Bureau Model Score":               parsed.get("model_score"),
    }


# ---------------------------------------------------------------------------
# Supporting tradeline helpers
# ---------------------------------------------------------------------------

def count_credit_sources(parsed: dict) -> int:
    trades = parsed["trades_df"]
    return trades[trades["is_active"]]["name"].nunique()


def check_lender_repayments(parsed: dict) -> tuple:
    trades  = parsed["trades_df"]
    active  = trades[trades["is_active"]].copy()
    with_pmts, missing = {}, []
    for _, row in active.iterrows():
        label = f"{row['name']} ({row['account_type']})" if row["account_type"] else row["name"]
        if row["scheduled_payment"] > 0:
            with_pmts[label] = round(row["scheduled_payment"], 2)
        else:
            missing.append(label)
    return with_pmts, missing


def process_balance_report(parsed: dict) -> pd.DataFrame:
    trades = parsed["trades_df"].copy()
    if trades.empty:
        return pd.DataFrame()
    active = trades[trades["is_active"]].copy()

    def _util(row):
        if row["credit_limit"] > 0:
            return round(row["balance"] / row["credit_limit"] * 100, 2)
        if row["high_credit"] > 0:
            return round(row["balance"] / row["high_credit"] * 100, 2)
        return 0.0

    active["utilisation_pct"] = active.apply(_util, axis=1)
    active["account_status"]  = active["pays_as_agreed"].map(
        {True: "Pays as agreed", False: "Issue flagged"}
    )
    out = active[[
        "name", "account_type", "portfolio_code",
        "balance", "credit_limit", "high_credit",
        "scheduled_payment", "actual_payment",
        "utilisation_pct", "account_status",
        "date_opened", "date_reported", "months_reviewed",
    ]].copy()
    out.columns = [
        "Lender", "Account Type", "Portfolio",
        "Current Balance", "Credit Limit", "High Credit / Loan Amount",
        "Scheduled Payment", "Actual Payment",
        "Utilisation (%)", "Status",
        "Date Opened", "Last Reported", "Months Reviewed",
    ]
    return out.sort_values("Current Balance", ascending=False).reset_index(drop=True)


def inquiry_velocity(parsed: dict) -> dict:
    inquiries = parsed["inquiries_df"].copy()
    if inquiries.empty:
        return {"clustered_inquiry_dates": [], "total": 0, "by_date": {}}
    inquiries = inquiries.dropna(subset=["date"]).sort_values("date")
    inquiries["date_only"] = inquiries["date"].dt.date
    daily     = inquiries.groupby("date_only").size().reset_index(name="count")
    clustered = daily[daily["count"] >= 3]["date_only"].tolist()
    return {
        "clustered_inquiry_dates": [str(d) for d in clustered],
        "total":   len(inquiries),
        "by_date": daily.set_index("date_only")["count"].to_dict(),
    }


# ---------------------------------------------------------------------------
# Canonical feature pipeline
# ---------------------------------------------------------------------------

def _build_canonical_feats(raw_payload: dict, model_score: Any = None) -> dict:
    """
    Route the raw JSON through EquifaxJsonAdapter → build_features() so every
    canonical key (totalaccounts from attr 4140, utilizationbankcard3months from
    attr 4932, etc.) comes from its specific Equifax attribute ID rather than
    from raw tradeline rollups.
    """
    result = _EQX_ADAPTER.parse_payload(raw_payload, source_name="upload.json")
    feats  = _build_features(result)
    if model_score is not None:
        feats.setdefault("attr__creditscore.class_10.score", float(model_score))
        feats.setdefault("attr__creditscore.class_11.score", float(model_score))
    return feats


def _build_feats(parsed: dict, raw_metrics: dict) -> dict:
    """Return feats dict for risk pillars. Primary: canonical pipeline. Fallback: raw approximations."""
    if _CANONICAL_PIPELINE_OK and parsed.get("raw_payload"):
        try:
            feats = _build_canonical_feats(parsed["raw_payload"], model_score=parsed.get("model_score"))
            print(f"[bureau_equifax_us] Canonical feats built OK — {len(feats)} keys.", file=sys.stderr)
            return feats
        except Exception as _rt_err:
            print(
                f"[bureau_equifax_us] Canonical pipeline RUNTIME ERROR — falling back to raw.\n"
                f"  Reason: {_rt_err}\n{traceback.format_exc()}",
                file=sys.stderr,
            )

    # Fallback — raw tradeline approximations (canonical keys, not contaminated across buckets)
    print(f"[bureau_equifax_us] Using raw fallback. _CANONICAL_PIPELINE_OK={_CANONICAL_PIPELINE_OK}", file=sys.stderr)
    trades   = parsed["trades_df"]
    identity = parsed["identity"]
    if trades.empty:
        return {"attr__summary.us.foreclosures12months": 0}

    active      = trades[trades["is_active"]]
    rev_active  = trades[trades["is_revolving"] & trades["is_active"]]
    inst_active = trades[trades["is_installment"] & trades["is_active"]]

    rev_bal   = float(rev_active["balance"].sum())
    rev_lim   = float(rev_active["credit_limit"].sum())
    rev_ratio = round(rev_bal / rev_lim * 100, 2) if rev_lim > 0 else 0.0
    n_adverse = int(trades["is_adverse"].sum())
    bk_flag   = int(any("BC" in (r or []) for r in trades["narratives"].tolist()))

    valid_dates = trades["date_opened"].dropna()
    avg_age = 0
    if not valid_dates.empty:
        ref  = identity.get("report_date") or pd.Timestamp.now()
        avg_age = int(sum((ref - d).days / 30.44 for d in valid_dates) / len(valid_dates))

    ms = parsed.get("model_score")
    return {
        "attr__summary.share.totalaccounts":             raw_metrics.get("Total Accounts"),
        "attr__summary.share.totalactiveaccs":           raw_metrics.get("Active Accounts"),
        "attr__summary.share.totalsettledaccs":          raw_metrics.get("Closed Accounts"),
        "attr__summary.indebt.totalbalancesactive":      float(active["balance"].sum()),
        "attr__summary.indebt.totalbalancesrevolve":     rev_bal,
        "attr__summary.indebt.totallimitsrevolve":       rev_lim,
        "attr__summary.indebt.balancelimitratiorevolve": rev_ratio,
        "attr__summary.indebt.totalbalancesloans":       float(inst_active["balance"].sum()),
        "attr__summary.searches.totalsearches3months":   raw_metrics.get("Inquiries (Last 3 Months)"),
        "attr__summary.searches.totalsearches12months":  raw_metrics.get("Inquiries (Last 12 Months)"),
        "attr__creditscore.class_10.score":              ms,
        "attr__creditscore.class_11.score":              ms,
        "attr__summary.ich.impairedcredit":              bk_flag,
        "attr__summary.us.foreclosures12months":         int((trades["is_mortgage"] & trades["is_adverse"]).sum()),
        "attr__summary.us.bankruptcyflagever":           bk_flag,
        "attr__summary.us.majorderogatoryever":          n_adverse,
        "attr__summary.us.tradespastdueever":            n_adverse,
        "attr__summary.us.tradeswithpaymentmadeever":    raw_metrics.get("Accounts Paying as Agreed"),
        "attr__summary.us.inquiries6months":             raw_metrics.get("Inquiries (Last 6 Months)"),
        "attr__summary.us.monthsoncreditfile":           raw_metrics.get("Credit Age (Months)"),
        "attr__summary.us.ageoldesttrade":               raw_metrics.get("Credit Age (Months)"),
        "attr__summary.us.averageagetrades":             avg_age,
        # Utilization buckets kept separate (not copied from rev_ratio)
        "attr__summary.us.utilizationrevolving6months":       None,
        "attr__summary.us.utilizationbankcard3months":        None,
        "attr__summary.us.utilizationbankcard6months":        None,
        "attr__summary.us.utilizationretailrevolving3months": None,
        "attr__summary.us.maxutilizationrevolving3months":    None,
    }


# ---------------------------------------------------------------------------
# Canonical metrics overlay
# ---------------------------------------------------------------------------

# Account counts: replaced by canonical Equifax attr values (4140, 4173).
_CANONICAL_COUNTS = [
    ("Total Accounts",  "attr__summary.share.totalaccounts"),
    ("Active Accounts", "attr__summary.share.totalactiveaccs"),
    ("Closed Accounts", "attr__summary.share.totalsettledaccs"),
]

# Decisioning fields appended to metrics so the frontend can display them
# separately from raw tradeline rollups.
_CANONICAL_DECISIONING = [
    ("Total Balance on Open Trades",        "attr__summary.indebt.totalbalancesactive"),
    ("Revolving Balance",                   "attr__summary.indebt.totalbalancesrevolve"),
    ("Revolving Credit Limit",              "attr__summary.indebt.totallimitsrevolve"),
    ("Revolving Utilization 3m (%)",        "attr__summary.indebt.balancelimitratiorevolve"),
    ("Revolving Utilization 6m (%)",        "attr__summary.us.utilizationrevolving6months"),
    ("Bankcard Utilization 3m (%)",         "attr__summary.us.utilizationbankcard3months"),
    ("Bankcard Utilization 6m (%)",         "attr__summary.us.utilizationbankcard6months"),
    ("Retail Revolving Utilization 3m (%)", "attr__summary.us.utilizationretailrevolving3months"),
    ("Max Revolving Utilization 3m (%)",    "attr__summary.us.maxutilizationrevolving3months"),
    ("Trades Past Due Ever",                "attr__summary.us.tradespastdueever"),
    ("Major Derogatory Trades Ever",        "attr__summary.us.majorderogatoryever"),
    ("Trades with Payment Made Ever",       "attr__summary.us.tradeswithpaymentmadeever"),
    ("Months on Credit File",               "attr__summary.us.monthsoncreditfile"),
    ("Oldest Trade Age (Months)",           "attr__summary.us.ageoldesttrade"),
    ("Average Trade Age (Months)",          "attr__summary.us.averageagetrades"),
]


def _overlay_canonical(metrics: dict, feats: dict) -> dict:
    m = dict(metrics)
    for m_key, f_key in _CANONICAL_COUNTS:
        v = _safe_float(feats.get(f_key))
        if v is not None:
            m[m_key] = int(v)
    for m_key, f_key in _CANONICAL_DECISIONING:
        v = _safe_float(feats.get(f_key))
        m[m_key] = round(v, 2) if v is not None else None
    return m


# ---------------------------------------------------------------------------
# Risk pillars  (logic lives in bureau_uw.scoring_api)
# ---------------------------------------------------------------------------

def build_pillars(feats: dict) -> List[dict]:
    rows = []
    for name, fn in [
        ("Credit Quality",    pillar_credit_quality),
        ("Exposure Pressure", pillar_exposure_pressure),
        ("Conduct",           pillar_conduct),
        ("Stability",         pillar_stability),
    ]:
        level, notes = fn(feats)
        rows.append({
            "pillar":     name,
            "risk":       level,
            "risk_score": RISK_TO_NUM.get(level, 0),
            "notes":      notes,
        })
    return rows


def tier_from_pillars(pillars: List[dict]) -> str:
    levels = [p["risk"] for p in pillars]
    if any(r == "Elevated" for r in levels):
        return "Detailed Review"
    if any(r == "Moderate" for r in levels):
        return "Enhanced Review"
    return "Standard Review"


# ---------------------------------------------------------------------------
# Top-level summary
# ---------------------------------------------------------------------------

def summary(parsed: dict) -> dict:
    raw_metrics = calculate_metrics(parsed)
    bounced     = get_bounced_equivalents(parsed)
    velocity    = inquiry_velocity(parsed)
    balance_r   = process_balance_report(parsed)
    n_sources   = count_credit_sources(parsed)
    lenders_w, lenders_missing = check_lender_repayments(parsed)

    feats   = _build_feats(parsed, raw_metrics)
    pillars = build_pillars(feats)
    tier    = tier_from_pillars(pillars)
    metrics = _overlay_canonical(raw_metrics, feats)

    return {
        "identity":                 parsed["identity"],
        "metrics":                  metrics,
        "adverse_accounts":         bounced.to_dict("records") if not bounced.empty else [],
        "inquiry_velocity":         velocity,
        "monthly_balance_report":   balance_r.to_dict("records") if not balance_r.empty else [],
        "n_distinct_lenders":       n_sources,
        "lenders_with_payments":    lenders_w,
        "lenders_missing_payments": lenders_missing,
        "risk_pillars":             pillars,
        "tier":                     tier,
        "tier_guidance":            TIER_GUIDANCE.get(tier, ""),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    path   = sys.argv[1] if len(sys.argv) > 1 else "ResponseEssentials.json"
    parsed = parse_bureau_json(path)
    result = summary(parsed)

    print("=== IDENTITY ===")
    print(f"  Name        : {result['identity']['full_name']}")
    print(f"  Report Date : {result['identity']['report_date']}")

    print("\n=== KEY METRICS ===")
    for k, v in result["metrics"].items():
        print(f"  {k:<45}: {v}")

    print("\n=== ADVERSE ACCOUNTS ===")
    for a in result["adverse_accounts"]:
        print(f"  {a['name']} — {a['Bounce Category']}")

    print("\n=== RISK PILLARS ===")
    for p in result["risk_pillars"]:
        print(f"  {p['pillar']}: {p['risk']}")
        for n in p["notes"]:
            print(f"    - {n}")

    print(f"\nTier: {result['tier']}")
