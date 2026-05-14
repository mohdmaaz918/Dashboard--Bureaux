"""
bureau_equifax_us.py
====================
Equifax US Consumer Credit Report — parsing and metrics.

Extracted from a-1.ipynb and corrected for four identified issues:

  1. Tradeline deduplication  — drop exact duplicates before any counting.
  2. Adverse consistency      — CF (Closed Adverse) now treated as adverse in
                                _is_adverse() to match the written specification.
  3. Total Active Debt scope  — Revolving (R) portfolios now included so that
                                revolving balances are not silently excluded.
  4. Inquiry recency anchor   — windows now measured from the bureau report date,
                                not from pd.Timestamp.now(), so the same file
                                always yields the same counts regardless of when
                                it is processed.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Date helper
# ---------------------------------------------------------------------------

def _parse_date(raw: str, fmt_hint: str = "MMDDYYYY") -> pd.Timestamp | None:
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
# Portfolio / narrative code constants
# ---------------------------------------------------------------------------

# FIX 3: "R" (Revolving) added so that revolving balances are captured in
#         Total Active Debt.  Previously only I/M/C/O were classed as debt,
#         which silently excluded revolving accounts with outstanding balances.
DEBT_PORTFOLIO_CODES = {"I", "M", "C", "O", "R"}  # Installment, Mortgage, Line-of-Credit, Open, Revolving

REVOLVING_CODES = {"R", "C", "O"}   # Revolving, Line of Credit, Open

# Narrative codes that mark an account as closed (not adverse on their own)
CLOSED_CODES = {"FA", "CF"}          # FA = Closed/Paid, CF = Closed Account

# Consolidated adverse status maps to guarantee _is_adverse() and
# get_bounced_equivalents() are perfectly aligned.
ADVERSE_RATE_MAP = {
    "B": "Lost or Stolen Card",
}

ADVERSE_NARRATIVE_MAP = {
    "RP": "Returned Payment",
    "CO": "Charge-Off",
    "CL": "Collection",
    "BC": "Bankruptcy",
    "CF": "Closed Adverse",
}

# ---------------------------------------------------------------------------
# Review tiering  (from scoring_api.py)
# ---------------------------------------------------------------------------

TIER_BANDS = [
    ("Standard Review", 75.0, 100.0001),
    ("Enhanced Review", 60.0, 75.0),
    ("Detailed Review", -1e9, 60.0),
]

TIER_GUIDANCE = {
    "Standard Review": "Bureau profile is consistent with expectations. Verify key details.",
    "Enhanced Review": "Some bureau signals warrant closer attention. Review flagged areas; refer if necessary.",
    "Detailed Review": "Multiple bureau signals require thorough assessment. Review all pillars and tradelines; refer after checks.",
}

RISK_TO_NUM: Dict[str, int] = {"Unknown": 0, "Low": 1, "Moderate": 2, "Elevated": 3}


# ---------------------------------------------------------------------------
# Trade classifiers
# ---------------------------------------------------------------------------

def _is_active(trade: dict) -> bool:
    """Account is active if no closed narrative codes are present."""
    codes = {n["code"] for n in trade.get("narrativeCodes", [])}
    return len(codes & CLOSED_CODES) == 0


def _is_paying_as_agreed(trade: dict) -> bool:
    rate = trade.get("rate", {})
    return rate.get("code") == 1


def _is_adverse(trade: dict) -> bool:
    """
    Account has a derogatory marker.

    Checks against the unified ADVERSE_NARRATIVE_MAP and ADVERSE_RATE_MAP
    to assure alignment with get_bounced_equivalents().
    """
    codes = {n["code"] for n in trade.get("narrativeCodes", [])}
    rate_status = trade.get("rateStatusCode", {}).get("code", "")
    
    has_adverse_narrative = bool(codes & set(ADVERSE_NARRATIVE_MAP.keys()))
    has_adverse_rate = rate_status in ADVERSE_RATE_MAP
    
    return has_adverse_narrative or has_adverse_rate


def _classify_trade(trade: dict) -> dict:
    portfolio = trade.get("portfolioTypeCode", {}).get("code", "")
    return {
        "is_debt":        portfolio in DEBT_PORTFOLIO_CODES,
        "is_revolving":   portfolio in REVOLVING_CODES,
        "is_mortgage":    portfolio == "M",
        "is_installment": portfolio == "I",
        "is_active":      _is_active(trade),
        "is_adverse":     _is_adverse(trade),
        "pays_as_agreed": _is_paying_as_agreed(trade),
    }


# ---------------------------------------------------------------------------
# FIX 1: Tradeline deduplication
# ---------------------------------------------------------------------------

# Natural key that identifies a unique account.  Two rows that share all of
# these fields are treated as duplicates; only the first occurrence is kept.
#
# Rationale for each field:
#   name          — lender / creditor name
#   account_type  — e.g. "Credit Card", "Auto Loan" (breaks ties for same lender)
#   portfolio_code — I / M / R / C / O
#   date_opened   — opening date of the account
#   high_credit   — original loan amount or peak balance (stable identifier)
#
# balance is intentionally excluded from the key because it fluctuates month
# to month and would fail to detect a duplicate that was re-pulled at a
# different point in time.
_DEDUP_COLS = ["name", "account_type", "portfolio_code", "date_opened", "high_credit"]


def _dedup_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop duplicate tradelines using the natural account key.

    Returns a de-duplicated DataFrame with a reset index.
    Rows removed are logged to stderr so underwriters can audit if needed.
    """
    if df.empty:
        return df

    subset = [c for c in _DEDUP_COLS if c in df.columns]
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    removed = before - len(df)
    if removed:
        import sys
        print(
            f"[bureau_equifax_us] Deduplication removed {removed} duplicate "
            f"tradeline(s). Total tradelines before: {before}, after: {len(df)}.",
            file=sys.stderr,
        )
    return df


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_bureau_json(source) -> dict:
    """
    Parse an Equifax US Consumer Credit Report JSON (ResponseEssentials format).

    Returns a dict with keys:
        report, trades_df, inquiries_df, identity, model_score
    """
    if isinstance(source, str):
        with open(source) as f:
            raw = json.load(f)
    else:
        raw = source

    report = raw["consumers"]["equifaxUSConsumerCreditReport"][0]

    # --- Build trades DataFrame ---
    rows = []
    for trade in report.get("trades", []):
        flags          = _classify_trade(trade)
        balance        = trade.get("balance") or 0
        high_credit    = trade.get("highCredit") or 0
        credit_limit   = trade.get("creditLimit") or 0
        scheduled_pmt  = trade.get("scheduledPaymentAmount") or 0
        actual_pmt     = trade.get("actualPaymentAmount") or 0
        months_reviewed = int(trade.get("monthsReviewed") or 0)

        date_opened   = _parse_date(trade.get("dateOpened", ""))
        date_reported = _parse_date(trade.get("dateReported", ""))

        rows.append({
            # Identity
            "name":           trade.get("customerName", ""),
            "account_type":   trade.get("accountTypeCode", {}).get("description", ""),
            "portfolio_code": trade.get("portfolioTypeCode", {}).get("code", ""),
            "designator":     trade.get("accountDesignator", {}).get("description", ""),
            "narratives":     [n["code"] for n in trade.get("narrativeCodes", [])],

            # Dates
            "date_opened":    date_opened,
            "date_reported":  date_reported,

            # Amounts
            "balance":           balance,
            "high_credit":       high_credit,
            "credit_limit":      credit_limit,
            "scheduled_payment": scheduled_pmt,
            "actual_payment":    actual_pmt,

            **flags,

            "months_reviewed": months_reviewed,
        })

    trades_df = pd.DataFrame(rows)

    # FIX 1: deduplicate before any downstream counting
    trades_df = _dedup_trades(trades_df)

    # --- Build inquiries DataFrame ---
    inq_rows = []
    for inq in report.get("inquiries", []):
        inq_rows.append({
            "date":          _parse_date(inq.get("inquiryDate", "")),
            "name":          inq.get("customerName", ""),
            "type":          inq.get("type", ""),
            "industry_code": inq.get("industryCode", ""),
        })
    inquiries_df = pd.DataFrame(inq_rows)

    # --- Identity ---
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

    # --- Model score ---
    model_score = None
    for m in report.get("models", []):
        model_score = m.get("score")
        break

    return {
        "report":       report,
        "trades_df":    trades_df,
        "inquiries_df": inquiries_df,
        "identity":     identity,
        "model_score":  model_score,
    }


# ---------------------------------------------------------------------------
# Adverse / bounced-equivalent accounts
# ---------------------------------------------------------------------------

def get_bounced_equivalents(parsed: dict) -> pd.DataFrame:
    """
    Return a DataFrame of trades that carry an adverse/bounced-equivalent marker.

    Guaranteed to be aligned with _is_adverse() by relying on the shared
    ADVERSE_NARRATIVE_MAP and ADVERSE_RATE_MAP constants.
    """
    report = parsed["report"]
    adverse_rows = []

    for trade in report.get("trades", []):
        categories = []

        rate_status_code = trade.get("rateStatusCode", {}).get("code", "")
        if rate_status_code in ADVERSE_RATE_MAP:
            categories.append(ADVERSE_RATE_MAP[rate_status_code])

        for n in trade.get("narrativeCodes", []):
            code = n["code"]
            if code in ADVERSE_NARRATIVE_MAP:
                cat = ADVERSE_NARRATIVE_MAP[code]
                if cat not in categories:
                    categories.append(cat)

        if categories:
            adverse_rows.append({
                "name":            trade.get("customerName", ""),
                "account_type":    trade.get("accountTypeCode", {}).get("description", ""),
                "date_reported":   _parse_date(trade.get("dateReported", "")),
                "balance":         trade.get("balance", 0),
                "Bounce Category": ", ".join(categories),
                "narratives":      [n["description"] for n in trade.get("narrativeCodes", [])],
            })

    if not adverse_rows:
        return pd.DataFrame()

    return pd.DataFrame(adverse_rows)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def calculate_metrics(parsed: dict) -> dict:
    trades    = parsed["trades_df"]
    inquiries = parsed["inquiries_df"]
    identity  = parsed["identity"]

    if trades.empty:
        return {}

    active_trades  = trades[trades["is_active"]]
    adverse_trades = trades[trades["is_adverse"]]

    # --- Core balance metrics ---
    total_balance      = round(trades["balance"].sum(), 2)
    total_credit_limit = round(trades["credit_limit"].sum(), 2)   # revolving limits only
    total_high_credit  = round(trades["high_credit"].sum(), 2)    # peak/loan amounts
    unused_credit      = round(total_credit_limit - total_balance, 2)

    # Utilisation: balance / credit_limit (only where limit is known)
    trades_with_limit = trades[trades["credit_limit"] > 0]
    utilisation_rate  = round(
        trades_with_limit["balance"].sum() / trades_with_limit["credit_limit"].sum() * 100
        if not trades_with_limit.empty else 0,
        2,
    )

    # --- Debt metrics ---
    # FIX 3: debt_trades now includes revolving (R) accounts because R is in
    #         DEBT_PORTFOLIO_CODES.  This prevents understating Total Active Debt
    #         for customers with revolving balances.
    debt_trades          = trades[trades["is_debt"]]
    total_debt           = round(debt_trades["balance"].sum(), 2)
    total_scheduled_pmts = round(trades["scheduled_payment"].sum(), 2)
    total_actual_pmts    = round(trades["actual_payment"].sum(), 2)

    dscr = round(
        total_actual_pmts / total_scheduled_pmts
        if total_scheduled_pmts > 0 else 0,
        2,
    )
    avg_monthly_obligation = round(total_scheduled_pmts, 2)

    # --- Account mix ---
    n_total       = len(trades)
    n_active      = int(trades["is_active"].sum())
    n_closed      = n_total - n_active
    n_mortgage    = int(trades["is_mortgage"].sum())
    n_installment = int(trades["is_installment"].sum())
    n_revolving   = int(trades["is_revolving"].sum())
    n_adverse     = int(trades["is_adverse"].sum())
    n_pays_agreed = int(trades["pays_as_agreed"].sum())

    # --- Credit age ---
    valid_dates = trades["date_opened"].dropna()
    credit_age_months = 0
    oldest_account    = None
    newest_account    = None
    if not valid_dates.empty:
        report_date    = identity.get("report_date") or pd.Timestamp.now()
        oldest_account = valid_dates.min()
        newest_account = valid_dates.max()
        credit_age_months = int((report_date - oldest_account).days / 30.44)

    # --- Inquiry metrics ---
    # FIX 4: anchor recency windows to the bureau report date rather than
    #         pd.Timestamp.now(), so the same file always yields identical
    #         inquiry counts regardless of when it is processed.
    n_inquiries_total = len(inquiries)

    # Use report date as reference; fall back to now only if missing.
    report_date_ref = identity.get("report_date") or pd.Timestamp.now()

    n_inquiries_3m  = 0
    n_inquiries_6m  = 0
    n_inquiries_12m = 0
    if not inquiries.empty and "date" in inquiries.columns:
        valid_inq = inquiries.dropna(subset=["date"])
        n_inquiries_3m  = int((valid_inq["date"] >= report_date_ref - pd.DateOffset(months=3)).sum())
        n_inquiries_6m  = int((valid_inq["date"] >= report_date_ref - pd.DateOffset(months=6)).sum())
        n_inquiries_12m = int((valid_inq["date"] >= report_date_ref - pd.DateOffset(months=12)).sum())

    # --- Adverse / bounced equivalents ---
    bounced_df         = get_bounced_equivalents(parsed)
    n_adverse_accounts = len(bounced_df)

    # --- Address discrepancy ---
    address_discrepancy = identity.get("address_discrepancy", False)

    # --- Months of history ---
    max_months_history = int(trades["months_reviewed"].max()) if not trades.empty else 0

    return {
        # Balance metrics
        "Total Balance (All Accounts)":     total_balance,
        "Total Credit Limit":               total_credit_limit,
        "Total High Credit (Peak/Loans)":   total_high_credit,
        "Unused Credit":                    unused_credit,
        "Credit Utilisation (%)":           utilisation_rate,

        # Debt metrics
        "Total Active Debt":                total_debt,
        "Total Scheduled Monthly Payments": total_scheduled_pmts,
        "Total Actual Payments Made":       total_actual_pmts,
        "Debt Service Coverage Ratio":      dscr,
        "Average Monthly Obligation":       avg_monthly_obligation,

        # Account mix
        "Total Accounts":                   n_total,
        "Active Accounts":                  n_active,
        "Closed Accounts":                  n_closed,
        "Mortgage Accounts":                n_mortgage,
        "Installment Accounts":             n_installment,
        "Revolving Accounts":               n_revolving,
        "Accounts Paying as Agreed":        n_pays_agreed,
        "Adverse Accounts":                 n_adverse,

        # Stability
        "Credit Age (Months)":              credit_age_months,
        "Oldest Account Opened":            str(oldest_account.date()) if oldest_account else None,
        "Newest Account Opened":            str(newest_account.date()) if newest_account else None,
        "Max Months of History":            max_months_history,

        # Conduct / Inquiry
        "Total Inquiries":                  n_inquiries_total,
        "Inquiries (Last 3 Months)":        n_inquiries_3m,
        "Inquiries (Last 6 Months)":        n_inquiries_6m,
        "Inquiries (Last 12 Months)":       n_inquiries_12m,

        # Risk flags
        "Adverse Accounts (Bounced equiv)": n_adverse_accounts,
        "Address Discrepancy Flag":         address_discrepancy,

        # Bureau model score (raw)
        "Bureau Model Score":               parsed.get("model_score"),
    }


# ---------------------------------------------------------------------------
# Supporting helpers
# ---------------------------------------------------------------------------

def count_credit_sources(parsed: dict) -> int:
    trades = parsed["trades_df"]
    active = trades[trades["is_active"]]
    return active["name"].nunique()


def _clean_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = re.sub(r'www\.|\\.co\\.uk|\.com|\.org|\.net', '', name)
    name = re.sub(r'\bltd\b|\blimited\b|\bt/a\b', '', name)
    name = re.sub(r'\b(repayment|payment|finance|loan|loans|transfer)\b', '', name)
    name = re.sub(r'\b[a-z]*\d+[a-z\d]*\b', '', name)
    name = re.sub(r'[^a-z0-9\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def check_lender_repayments(parsed: dict, threshold: int = 80) -> tuple:
    trades     = parsed["trades_df"]
    all_active = trades[trades["is_active"]].copy()

    lenders_with_payments    = {}
    lenders_missing_payments = []

    for _, row in all_active.iterrows():
        label = f"{row['name']} ({row['account_type']})" if row["account_type"] else row["name"]
        pmt   = row["scheduled_payment"]

        if pmt > 0:
            lenders_with_payments[label] = round(pmt, 2)
        else:
            lenders_missing_payments.append(label)

    return lenders_with_payments, lenders_missing_payments


def process_balance_report(parsed: dict) -> pd.DataFrame:
    trades = parsed["trades_df"].copy()

    if trades.empty:
        return pd.DataFrame()

    active = trades[trades["is_active"]].copy()

    def _utilisation(row):
        if row["credit_limit"] > 0:
            return round(row["balance"] / row["credit_limit"] * 100, 2)
        elif row["high_credit"] > 0:
            return round(row["balance"] / row["high_credit"] * 100, 2)
        return 0.0

    active["utilisation_pct"] = active.apply(_utilisation, axis=1)
    active["account_status"]  = active["pays_as_agreed"].map(
        {True: "Pays as agreed", False: "Issue flagged"}
    )

    report = active[[
        "name", "account_type", "portfolio_code",
        "balance", "credit_limit", "high_credit",
        "scheduled_payment", "actual_payment",
        "utilisation_pct", "account_status",
        "date_opened", "date_reported", "months_reviewed",
    ]].copy()

    report.columns = [
        "Lender", "Account Type", "Portfolio",
        "Current Balance", "Credit Limit", "High Credit / Loan Amount",
        "Scheduled Payment", "Actual Payment",
        "Utilisation (%)", "Status",
        "Date Opened", "Last Reported", "Months Reviewed",
    ]

    return report.sort_values("Current Balance", ascending=False).reset_index(drop=True)


def inquiry_velocity(parsed: dict) -> dict:
    inquiries = parsed["inquiries_df"].copy()
    if inquiries.empty:
        return {"clustered_inquiry_dates": [], "counts": {}}

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
# Risk pillar helpers  (from scoring_api.py)
# ---------------------------------------------------------------------------

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float, np.number)):
            return float(x)
        s = str(x).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def _get(feats: Dict[str, Any], key: str) -> Optional[float]:
    return _safe_float(feats.get(key))


def _fmt_val(x: Any) -> str:
    if x is None or x == "":
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    if isinstance(x, (int, float, np.number)):
        fv = float(x)
        return f"{int(fv)}" if fv.is_integer() else f"{fv:,.2f}"
    return str(x)


def _is_us_equifax(feats: Dict[str, Any]) -> bool:
    return any(
        _get(feats, k) is not None
        for k in (
            "attr__summary.us.foreclosures12months",
            "attr__summary.us.foreclosures24months",
            "attr__summary.us.foreclosuresever",
            "attr__summary.us.bankruptcyflagever",
            "attr__summary.us.majorderogatoryever",
            "attr__summary.us.inquiries6months",
        )
    )


def _build_feats_from_bureau(parsed: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Bridge: maps bureau_equifax_us parsed output → feats dict consumed by pillar functions.

    US-specific attributes (foreclosures, bankruptcy, etc.) are approximated from the
    trade-level data already parsed. Values of None mean the signal is genuinely
    unavailable; the pillar functions skip None values gracefully.
    """
    trades   = parsed["trades_df"]
    identity = parsed["identity"]

    if trades.empty:
        return {"attr__summary.us.foreclosures12months": 0}  # ensure _is_us_equifax → True

    active      = trades[trades["is_active"]]
    rev_active  = trades[trades["is_revolving"] & trades["is_active"]]
    inst_active = trades[trades["is_installment"] & trades["is_active"]]

    rev_balance = float(rev_active["balance"].sum())
    rev_limit   = float(rev_active["credit_limit"].sum())
    rev_ratio   = round(rev_balance / rev_limit * 100, 2) if rev_limit > 0 else 0.0
    active_bal  = float(active["balance"].sum())
    loan_bal    = float(inst_active["balance"].sum())

    n_adverse = int(trades["is_adverse"].sum())

    # Bankruptcy proxy: narrative code BC present on any trade
    bk_flag = int(any("BC" in (row or []) for row in trades["narratives"].tolist()))

    # Foreclosure proxy: adverse mortgage trade
    mort_adverse = int((trades["is_mortgage"] & trades["is_adverse"]).sum())

    # Average age of trades in months
    valid_dates = trades["date_opened"].dropna()
    avg_age = 0
    if not valid_dates.empty:
        ref  = identity.get("report_date") or pd.Timestamp.now()
        ages = [(ref - d).days / 30.44 for d in valid_dates]
        avg_age = int(sum(ages) / len(ages))

    model_score = parsed.get("model_score")

    return {
        # Summary share
        "attr__summary.share.totalaccounts":        metrics.get("Total Accounts"),
        "attr__summary.share.totalactiveaccs":      metrics.get("Active Accounts"),
        "attr__summary.share.totalsettledaccs":     metrics.get("Closed Accounts"),
        "attr__summary.share.totalopened6months":   None,
        "attr__summary.share.totaldefaults36months": n_adverse,
        "attr__summary.share.totaldefaults12months": None,
        "attr__summary.share.totaldelinqs12months":  None,

        # Balance / debt
        "attr__summary.indebt.totalbalancesactive":      active_bal,
        "attr__summary.indebt.totalbalancesrevolve":     rev_balance,
        "attr__summary.indebt.totallimitsrevolve":       rev_limit,
        "attr__summary.indebt.balancelimitratiorevolve": rev_ratio,
        "attr__summary.indebt.totalbalancesloans":       loan_bal,

        # Searches / inquiries
        "attr__summary.searches.totalsearches3months":  metrics.get("Inquiries (Last 3 Months)"),
        "attr__summary.searches.totalsearches12months": metrics.get("Inquiries (Last 12 Months)"),

        # Credit score (bureau model score used as proxy)
        "attr__creditscore.class_10.score": model_score,
        "attr__creditscore.class_11.score": model_score,

        # ICH
        "attr__summary.ich.impairedcredit": bk_flag,
        "attr__summary.ich.unsecured":      None,

        # US-specific signals
        "attr__summary.us.foreclosures12months":              mort_adverse,
        "attr__summary.us.foreclosures24months":              mort_adverse,
        "attr__summary.us.foreclosuresever":                  mort_adverse,
        "attr__summary.us.bankruptcyflagever":                bk_flag,
        "attr__summary.us.publicrecordbankruptciesever":      bk_flag,
        "attr__summary.us.monthssincebankruptcyfiled":        None,
        "attr__summary.us.majorderogatoryever":               n_adverse,
        "attr__summary.us.monthssincemajorderogatory":        None,
        "attr__summary.us.tradespastdueever":                 n_adverse,
        "attr__summary.us.totalpastdueever":                  None,
        "attr__summary.us.pctpastduetobalanceever":           None,
        "attr__summary.us.pctpastduetobalanceexstudentever":  None,
        "attr__summary.us.tradeswithpaymentmadeever":         metrics.get("Accounts Paying as Agreed"),
        "attr__summary.us.inquiries6months":                  metrics.get("Inquiries (Last 6 Months)"),
        "attr__summary.us.inquiries30days":                   None,
        "attr__summary.us.monthsoncreditfile":                metrics.get("Credit Age (Months)"),
        "attr__summary.us.ageoldesttrade":                    metrics.get("Credit Age (Months)"),
        "attr__summary.us.averageagetrades":                  avg_age,
        "attr__summary.us.utilizationrevolving6months":       rev_ratio,
        "attr__summary.us.utilizationbankcard3months":        rev_ratio,
        "attr__summary.us.utilizationbankcard6months":        rev_ratio,
        "attr__summary.us.utilizationretailrevolving3months": None,
        "attr__summary.us.maxutilizationrevolving3months":    rev_ratio,
        "attr__summary.us.totalbalanceopentrades3months":     active_bal,
    }


# ---------------------------------------------------------------------------
# Risk pillars  (from scoring_api.py — verbatim logic, US Equifax path active)
# ---------------------------------------------------------------------------

def pillar_credit_quality(feats: Dict[str, Any]) -> Tuple[str, List[str]]:
    notes: List[str] = []
    level = "Low"
    if _is_us_equifax(feats):
        f12      = _get(feats, "attr__summary.us.foreclosures12months") or 0.0
        f24      = _get(feats, "attr__summary.us.foreclosures24months") or 0.0
        fev      = _get(feats, "attr__summary.us.foreclosuresever") or 0.0
        bk       = _get(feats, "attr__summary.us.bankruptcyflagever") or 0.0
        bk_cnt   = _get(feats, "attr__summary.us.publicrecordbankruptciesever") or 0.0
        bk_mos   = _get(feats, "attr__summary.us.monthssincebankruptcyfiled")
        md_ever  = _get(feats, "attr__summary.us.majorderogatoryever") or 0.0
        md_mos   = _get(feats, "attr__summary.us.monthssincemajorderogatory")
        oldest_trade = _get(feats, "attr__summary.us.ageoldesttrade")
        months_file  = _get(feats, "attr__summary.us.monthsoncreditfile")
        if f12 > 0:
            notes.append(f"US foreclosure trades (12m): {int(f12)}")
            level = "Elevated"
        if f24 > 0:
            notes.append(f"US foreclosure trades (24m): {int(f24)}")
            level = "Elevated"
        if fev > 0:
            notes.append(f"US foreclosure trades (ever): {int(fev)}")
            level = "Elevated"
        if bk > 0:
            notes.append("US bankruptcy flag (ever): 1")
            level = "Elevated"
        if bk_cnt > 0:
            notes.append(f"US public record bankruptcies (ever): {int(bk_cnt)}")
            level = "Elevated"
        if bk_mos is not None:
            notes.append(f"US months since bankruptcy filed: {int(bk_mos)}")
            if bk_mos <= 24:
                level = "Elevated"
        if md_ever > 0:
            notes.append(f"US major derogatory trades (ever): {int(md_ever)}")
            if level != "Elevated":
                level = "Moderate"
        if md_mos is not None:
            notes.append(f"US months since major derogatory: {int(md_mos)}")
            if md_mos <= 24 and level != "Elevated":
                level = "Moderate"
        if oldest_trade is not None:
            notes.append(f"US age oldest trade: {int(oldest_trade)}")
            if oldest_trade < 12 and level == "Low":
                level = "Moderate"
        if months_file is not None:
            notes.append(f"US months on file: {int(months_file)}")
            if months_file < 18 and level == "Low":
                level = "Moderate"
        if not notes:
            notes.append("No US foreclosure/bankruptcy adverse flags")
        return level, notes

    lqb = _get(feats, "attr__LQB")
    if lqb is not None and lqb > 0:
        notes.append("LQB: DECEASED INDICATOR PRESENT")
        level = "Elevated"
    pd_  = _get(feats, "attr__PD") or 0.0
    nqb  = _get(feats, "attr__NQB") or 0.0
    gqb  = _get(feats, "attr__GQB") or 0.0
    sd   = _get(feats, "attr__SD") or 0.0
    db   = _get(feats, "attr__DB") or 0.0
    ubc  = _get(feats, "attr__UBC")
    rq   = _get(feats, "attr__RQ")
    if pd_ > 0:
        notes.append("PD: Insolvency flag present")
        level = "Elevated"
    if nqb > 0:
        notes.append("NQB: Bankruptcy/sequestration in last 36m")
        level = "Elevated"
    if gqb > 0:
        notes.append("GQB: IVA in last 36m")
        level = "Elevated"
    if sd > 0:
        notes.append("SD: Currently restricted")
        level = "Elevated"
    if db > 0:
        notes.append(f"DB: Unsatisfied CCJs = {int(db)}")
        level = "Elevated"
    if ubc is not None and ubc > 0:
        notes.append(f"UBC: Worst pay last 3m = {_fmt_val(ubc)}")
        if level != "Elevated":
            level = "Moderate"
    if rq is not None and rq > 0:
        notes.append(f"RQ: Worst pay last month = {_fmt_val(rq)}")
        if level != "Elevated":
            level = "Moderate"
    iscore = _get(feats, "attr__internalscore")
    if iscore is not None:
        notes.append(f"Internal score: {iscore:.1f}")
    s10 = _get(feats, "attr__creditscore.class_10.score")
    s11 = _get(feats, "attr__creditscore.class_11.score")
    if s10 is not None:
        notes.append(f"Credit score (class 10): {int(s10)}")
    if s11 is not None:
        notes.append(f"Credit score (class 11): {int(s11)}")
    defaults36 = _get(feats, "attr__summary.share.totaldefaults36months")
    defaults12 = _get(feats, "attr__summary.share.totaldefaults12months")
    if defaults36 is not None and defaults36 > 0:
        notes.append(f"Summary defaults (36m): {int(defaults36)}")
        if level != "Elevated":
            level = "Moderate"
    if defaults12 is not None and defaults12 > 0:
        notes.append(f"Summary defaults (12m): {int(defaults12)}")
    ich_impaired = _get(feats, "attr__summary.ich.impairedcredit")
    if ich_impaired is not None and ich_impaired > 0:
        notes.append(f"ICH impaired credit flag: {int(ich_impaired)}")
    judgments_active = _get(feats, "attr__summary.judgments.totalactive")
    if judgments_active is not None and judgments_active > 0:
        notes.append(f"Active judgments: {int(judgments_active)}")
    if not notes:
        bsb_present = any(_get(feats, f"attr__{c}") is not None for c in ("NQB", "GQB", "SD", "PD", "DB"))
        if bsb_present:
            notes.append("No adverse credit quality flags")
        elif s10 is not None or s11 is not None:
            avg = float(np.mean([v for v in [s10, s11] if v is not None]))
            notes.append(f"Credit score avg: {avg:.0f}")
            if avg >= 700:
                level = "Low"
            elif avg >= 620:
                level = "Moderate"
            else:
                level = "Elevated"
        else:
            return "Unknown", ["No credit quality data available"]
    return level, notes


def pillar_exposure_pressure(feats: Dict[str, Any]) -> Tuple[str, List[str]]:
    notes: List[str] = []
    level = "Low"
    if _is_us_equifax(feats):
        bal_active   = _get(feats, "attr__summary.indebt.totalbalancesactive")
        open_bal_3m  = _get(feats, "attr__summary.us.totalbalanceopentrades3months")
        lim_rev      = _get(feats, "attr__summary.indebt.totallimitsrevolve")
        util3        = _get(feats, "attr__summary.indebt.balancelimitratiorevolve")
        util6        = _get(feats, "attr__summary.us.utilizationrevolving6months")
        util_bc3     = _get(feats, "attr__summary.us.utilizationbankcard3months")
        util_bc6     = _get(feats, "attr__summary.us.utilizationbankcard6months")
        util_retail3 = _get(feats, "attr__summary.us.utilizationretailrevolving3months")
        util_max3    = _get(feats, "attr__summary.us.maxutilizationrevolving3months")
        total_accs   = _get(feats, "attr__summary.share.totalaccounts")
        active_accs  = _get(feats, "attr__summary.share.totalactiveaccs")
        if bal_active is not None:
            notes.append(f"US active balances: ${bal_active:,.0f}")
        if open_bal_3m is not None:
            notes.append(f"US open-trade balances (3m): ${open_bal_3m:,.0f}")
        if lim_rev is not None:
            notes.append(f"US revolving limits: ${lim_rev:,.0f}")
        if util3 is not None:
            notes.append(f"US revolving utilisation (3m): {util3:.0f}%")
        if util6 is not None:
            notes.append(f"US revolving utilisation (6m): {util6:.0f}%")
        if util_bc3 is not None:
            notes.append(f"US bankcard utilisation (3m): {util_bc3:.0f}%")
        if util_bc6 is not None:
            notes.append(f"US bankcard utilisation (6m): {util_bc6:.0f}%")
        if util_retail3 is not None:
            notes.append(f"US retail revolving utilisation (3m): {util_retail3:.0f}%")
        if util_max3 is not None:
            notes.append(f"US max revolving utilisation (3m): {util_max3:.0f}%")
        if total_accs is not None:
            notes.append(f"US total trades: {int(total_accs)}")
        if active_accs is not None:
            notes.append(f"US open trades: {int(active_accs)}")
        peak_util = max(
            [u for u in (util3, util6, util_bc3, util_bc6, util_retail3, util_max3) if u is not None],
            default=None,
        )
        if peak_util is not None and peak_util >= 90:
            level = "Elevated"
        elif peak_util is not None and peak_util >= 70:
            level = "Moderate"
        elif bal_active is not None and bal_active >= 50000:
            level = "Moderate"
        if not notes:
            return "Unknown", ["No US exposure indicators available"]
        return level, notes

    wg    = _get(feats, "attr__WG")
    zhc   = _get(feats, "attr__ZHC")
    bic   = _get(feats, "attr__BIC")
    eb    = _get(feats, "attr__EB")
    fb    = _get(feats, "attr__FB")
    ih    = _get(feats, "attr__IH")
    ohc   = _get(feats, "attr__OHC")
    if wg is not None:
        notes.append(f"WG (opened 12m): {_fmt_val(wg)}")
    if zhc is not None:
        notes.append(f"ZHC (active STL): {_fmt_val(zhc)}")
    if bic is not None and bic > 0:
        notes.append(f"BIC (STL opened last mth): {_fmt_val(bic)}")
    if eb is not None:
        notes.append(f"EB (SHARE records): {_fmt_val(eb)}")
    if fb is not None:
        notes.append(f"FB (active SHARE): {_fmt_val(fb)}")
    if ih is not None:
        notes.append(f"IH (sector-1 active): {_fmt_val(ih)}")
    if ohc is not None:
        notes.append(f"OHC (opened sector-1 12m): {_fmt_val(ohc)}")
    if zhc is not None and zhc >= 2:
        level = "Elevated"
    elif wg is not None and wg >= 8:
        level = "Elevated"
    elif (zhc is not None and zhc >= 1) or (wg is not None and wg >= 4) or (eb is not None and eb >= 40):
        level = "Moderate"
    bal_active  = _get(feats, "attr__summary.indebt.totalbalancesactive")
    bal_rev     = _get(feats, "attr__summary.indebt.totalbalancesrevolve")
    lim_rev     = _get(feats, "attr__summary.indebt.totallimitsrevolve")
    ratio       = _get(feats, "attr__summary.indebt.balancelimitratiorevolve")
    bal_loans   = _get(feats, "attr__summary.indebt.totalbalancesloans")
    total_accs  = _get(feats, "attr__summary.share.totalaccounts")
    active_accs = _get(feats, "attr__summary.share.totalactiveaccs")
    if bal_active is not None:
        notes.append(f"Total active balances: £{bal_active:,.0f}")
    if bal_rev is not None:
        notes.append(f"Revolving balances: £{bal_rev:,.0f}")
    if lim_rev is not None:
        notes.append(f"Revolving limits: £{lim_rev:,.0f}")
    if ratio is not None:
        notes.append(f"Revolving utilisation: {ratio:.0f}%")
    if bal_loans is not None:
        notes.append(f"Loan balances: £{bal_loans:,.0f}")
    if total_accs is not None:
        notes.append(f"Total accounts: {int(total_accs)}")
    if active_accs is not None:
        notes.append(f"Active accounts: {int(active_accs)}")
    if level == "Low" and bal_active is not None:
        if bal_active >= 50000:
            level = "Elevated"
        elif bal_active >= 20000:
            level = "Moderate"
    if not notes:
        return "Unknown", ["No exposure data available"]
    return level, notes


def pillar_conduct(feats: Dict[str, Any]) -> Tuple[str, List[str]]:
    notes: List[str] = []
    level = "Low"
    if _is_us_equifax(feats):
        past_due_cnt     = _get(feats, "attr__summary.us.tradespastdueever")
        past_due_amt     = _get(feats, "attr__summary.us.totalpastdueever")
        pct_past_due     = _get(feats, "attr__summary.us.pctpastduetobalanceever")
        pct_past_due_exs = _get(feats, "attr__summary.us.pctpastduetobalanceexstudentever")
        md_ever          = _get(feats, "attr__summary.us.majorderogatoryever")
        trades_paid_ever = _get(feats, "attr__summary.us.tradeswithpaymentmadeever")
        defaults12       = _get(feats, "attr__summary.share.totaldefaults12months")
        defaults36       = _get(feats, "attr__summary.share.totaldefaults36months")
        delinqs12        = _get(feats, "attr__summary.share.totaldelinqs12months")
        if past_due_cnt is not None:
            notes.append(f"US trades past due (ever): {int(past_due_cnt)}")
            if past_due_cnt > 0:
                level = "Moderate"
        if past_due_amt is not None:
            notes.append(f"US total past due amount (ever): ${past_due_amt:,.0f}")
            if past_due_amt > 0 and level != "Elevated":
                level = "Moderate"
        if pct_past_due is not None:
            notes.append(f"US pct past due to balance (ever): {pct_past_due:.2f}")
            if pct_past_due > 0.25 and level != "Elevated":
                level = "Moderate"
        if pct_past_due_exs is not None:
            notes.append(f"US pct past due to balance ex-student (ever): {pct_past_due_exs:.2f}")
            if pct_past_due_exs > 0.25 and level != "Elevated":
                level = "Moderate"
        if md_ever is not None:
            notes.append(f"US major derogatory trades (ever): {int(md_ever)}")
            if md_ever > 0:
                level = "Elevated"
        if trades_paid_ever is not None:
            notes.append(f"US trades with payment made (ever): {int(trades_paid_ever)}")
            if trades_paid_ever <= 1 and level == "Low":
                level = "Moderate"
        if defaults12 is not None and defaults12 > 0:
            notes.append(f"US 30+ DPD/major derogatory opened (12m): {int(defaults12)}")
            level = "Elevated"
        if defaults36 is not None and defaults36 > 0:
            notes.append(f"US 30+ DPD/major derogatory reported (ever proxy): {int(defaults36)}")
            if level != "Elevated":
                level = "Moderate"
        if delinqs12 is not None and delinqs12 > 0:
            notes.append(f"US 30 DPD delinquency (12m): {int(delinqs12)}")
            if level == "Low":
                level = "Moderate"
        if not notes:
            return "Unknown", ["No US conduct indicators available"]
        return level, notes

    pp   = _get(feats, "attr__PP")
    qp   = _get(feats, "attr__QP")
    tg   = _get(feats, "attr__TG")
    ubc  = _get(feats, "attr__UBC")
    rq   = _get(feats, "attr__RQ")
    pg   = _get(feats, "attr__PG")
    jrb  = _get(feats, "attr__JRB")
    if pp is not None:
        notes.append(f"PP (defaults last 6m): {_fmt_val(pp)}")
        if pp > 0:
            level = "Elevated"
    if qp is not None:
        notes.append(f"QP (defaults 7-12m): {_fmt_val(qp)}")
        if qp > 0 and level != "Elevated":
            level = "Moderate"
    if tg is not None:
        notes.append(f"TG (repayment plans): {_fmt_val(tg)}")
        if tg > 0:
            level = "Elevated"
    if ubc is not None and ubc > 0:
        notes.append(f"UBC (worst pay last 3m): {_fmt_val(ubc)}")
        if level != "Elevated":
            level = "Moderate"
    if rq is not None and rq > 0:
        notes.append(f"RQ (worst pay last mth): {_fmt_val(rq)}")
        if level != "Elevated":
            level = "Moderate"
    if pg is not None:
        notes.append(f"PG (DM value 12m): {_fmt_val(pg)}")
    if jrb is not None:
        notes.append(f"JRB (active status '1'): {_fmt_val(jrb)}")
    delinqs12  = _get(feats, "attr__summary.share.totaldelinqs12months")
    worse12    = _get(feats, "attr__summary.share.worsepaystatus12months")
    worse36    = _get(feats, "attr__summary.share.worsepaystatus36months")
    defaults36 = _get(feats, "attr__summary.share.totaldefaults36months")
    unsecured  = _get(feats, "attr__summary.ich.unsecured")
    if delinqs12 is not None and delinqs12 > 0:
        notes.append(f"Delinquencies (12m): {int(delinqs12)}")
    if worse12 is not None and worse12 > 0:
        notes.append(f"Worst pay status (12m): {int(worse12)}")
    if worse36 is not None and worse36 > 0:
        notes.append(f"Worst pay status (36m): {int(worse36)}")
    if defaults36 is not None and defaults36 > 0:
        notes.append(f"Summary defaults (36m): {int(defaults36)}")
        if level != "Elevated":
            level = "Moderate"
    if unsecured is not None and unsecured > 0:
        notes.append(f"ICH unsecured: {int(unsecured)}")
    if not notes:
        return "Unknown", ["No conduct indicators available"]
    return level, notes


def pillar_stability(feats: Dict[str, Any]) -> Tuple[str, List[str]]:
    notes: List[str] = []
    level = "Low"
    if _is_us_equifax(feats):
        inq_30d      = _get(feats, "attr__summary.us.inquiries30days")
        inq_6m       = _get(feats, "attr__summary.us.inquiries6months")
        inq_12m      = _get(feats, "attr__summary.searches.totalsearches12months")
        opened6      = _get(feats, "attr__summary.share.totalopened6months")
        md_mos       = _get(feats, "attr__summary.us.monthssincemajorderogatory")
        avg_age      = _get(feats, "attr__summary.us.averageagetrades")
        oldest_trade = _get(feats, "attr__summary.us.ageoldesttrade")
        months_file  = _get(feats, "attr__summary.us.monthsoncreditfile")
        if inq_30d is not None:
            notes.append(f"US inquiries (30d): {int(inq_30d)}")
            if inq_30d >= 4:
                level = "Elevated"
            elif inq_30d >= 2:
                level = "Moderate"
        if inq_6m is not None:
            notes.append(f"US inquiries (6m): {int(inq_6m)}")
            if inq_6m >= 8:
                level = "Elevated"
            elif inq_6m >= 4 and level != "Elevated":
                level = "Moderate"
        if inq_12m is not None:
            notes.append(f"US inquiries (12m): {int(inq_12m)}")
        if opened6 is not None:
            notes.append(f"US trades opened (6m): {int(opened6)}")
            if opened6 >= 6 and level != "Elevated":
                level = "Moderate"
        if md_mos is not None:
            notes.append(f"US months since major derogatory: {int(md_mos)}")
            if md_mos <= 12:
                level = "Elevated"
            elif md_mos <= 24 and level != "Elevated":
                level = "Moderate"
        if avg_age is not None:
            notes.append(f"US average age of trades: {int(avg_age)}")
            if avg_age < 12 and level == "Low":
                level = "Moderate"
        if oldest_trade is not None:
            notes.append(f"US oldest trade age: {int(oldest_trade)}")
        if months_file is not None:
            notes.append(f"US months on file: {int(months_file)}")
            if months_file < 18 and level == "Low":
                level = "Moderate"
        if not notes:
            return "Unknown", ["No US stability indicators available"]
        return level, notes

    tr  = _get(feats, "attr__TR")
    bh  = _get(feats, "attr__BH")
    wg  = _get(feats, "attr__WG")
    efc = _get(feats, "attr__EFC")
    jgc = _get(feats, "attr__JGC")
    if tr is not None:
        if tr >= 99:
            notes.append("TR = 99 (no default recorded)")
        else:
            notes.append(f"TR (months since last default): {_fmt_val(tr)}")
            if tr < 6:
                level = "Elevated"
            elif tr < 12:
                if level != "Elevated":
                    level = "Moderate"
    if bh is not None:
        notes.append(f"BH (closed 12m): {_fmt_val(bh)}")
    if wg is not None:
        notes.append(f"WG (opened 12m): {_fmt_val(wg)}")
    if efc is not None:
        notes.append(f"EFC (settled sector-1): {_fmt_val(efc)}")
    if jgc is not None:
        notes.append(f"JGC (settled sector-1 12m): {_fmt_val(jgc)}")
    churn = (bh or 0) + (wg or 0)
    if churn >= 12 and level != "Elevated":
        level = "Moderate"
    s3        = _get(feats, "attr__summary.searches.totalsearches3months")
    s12       = _get(feats, "attr__summary.searches.totalsearches12months")
    opened6   = _get(feats, "attr__summary.share.totalopened6months")
    settled   = _get(feats, "attr__summary.share.totalsettledaccs")
    undec_addr = _get(feats, "attr__summary.links.totalundecaddressesunsearched")
    paf       = _get(feats, "attr__summary.summaryaddress.pafvalid")
    if s3 is not None:
        notes.append(f"Searches (3m): {int(s3)}")
        if s3 >= 6:
            level = "Elevated"
        elif s3 >= 3 and level != "Elevated":
            level = "Moderate"
    if s12 is not None:
        notes.append(f"Searches (12m): {int(s12)}")
        if s12 >= 10 and level != "Elevated":
            level = "Moderate"
    if opened6 is not None:
        notes.append(f"Opened last 6m: {int(opened6)}")
    if settled is not None:
        notes.append(f"Settled accounts: {int(settled)}")
    if undec_addr is not None and undec_addr >= 1:
        notes.append(f"Undeclared/unsearched addresses: {int(undec_addr)}")
        if level != "Elevated":
            level = "Moderate"
    if paf is not None:
        notes.append(f"PAF valid: {int(paf)}")
    if not notes:
        return "Unknown", ["No stability indicators available"]
    return level, notes


def build_pillars(feats: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Assess four risk pillars and return a JSON-serialisable list of dicts."""
    rows = []
    for pillar_name, fn in [
        ("Credit Quality",    pillar_credit_quality),
        ("Exposure Pressure", pillar_exposure_pressure),
        ("Conduct",           pillar_conduct),
        ("Stability",         pillar_stability),
    ]:
        level, notes = fn(feats)
        rows.append({
            "pillar":      pillar_name,
            "risk":        level,
            "risk_score":  RISK_TO_NUM.get(level, 0),
            "notes":       notes,
        })
    return rows


def tier_from_pillars(pillars: List[Dict[str, Any]]) -> str:
    """Assign review tier from pillar risk levels (rule-based; used when no ML score is available)."""
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
    metrics   = calculate_metrics(parsed)
    bounced   = get_bounced_equivalents(parsed)
    velocity  = inquiry_velocity(parsed)
    balance_r = process_balance_report(parsed)
    n_sources = count_credit_sources(parsed)
    lenders_w, lenders_missing = check_lender_repayments(parsed)

    feats   = _build_feats_from_bureau(parsed, metrics)
    pillars = build_pillars(feats)
    tier    = tier_from_pillars(pillars)

    return {
        "identity":                parsed["identity"],
        "metrics":                 metrics,
        "adverse_accounts":        bounced.to_dict("records") if not bounced.empty else [],
        "inquiry_velocity":        velocity,
        "monthly_balance_report":  balance_r.to_dict("records") if not balance_r.empty else [],
        "n_distinct_lenders":      n_sources,
        "lenders_with_payments":   lenders_w,
        "lenders_missing_payments": lenders_missing,
        "risk_pillars":            pillars,
        "tier":                    tier,
        "tier_guidance":           TIER_GUIDANCE.get(tier, ""),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    path   = sys.argv[1] if len(sys.argv) > 1 else "ResponseEssentials.json"
    parsed = parse_bureau_json(path)
    result = summary(parsed)

    print("=== IDENTITY ===")
    print(f"  Name        : {result['identity']['full_name']}")
    print(f"  Report Date : {result['identity']['report_date']}")
    print(f"  Addr Flag   : {result['identity']['address_discrepancy']}")

    print("\n=== KEY METRICS ===")
    for k, v in result["metrics"].items():
        print(f"  {k:<45}: {v}")

    print("\n=== ADVERSE ACCOUNTS ===")
    for a in result["adverse_accounts"]:
        print(f"  {a['name']} — {a['Bounce Category']}")

    print("\n=== INQUIRY VELOCITY ===")
    vel = result["inquiry_velocity"]
    print(f"  Total inquiries : {vel.get('total', 0)}")
    print(f"  Clustered dates : {vel['clustered_inquiry_dates']}")

    print("\n=== LENDERS WITHOUT SCHEDULED PAYMENTS ===")
    print(result["lenders_missing_payments"])
