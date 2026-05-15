"""
Headless scoring API for Bureau XML underwriting.

Use this module to integrate scoring into your loan management platform:
no Streamlit or UI dependencies. Load artifacts once, then call
score_bureau_xml() per application.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from bureau_uw.parsers.json_to_dict import EquifaxJsonAdapter
from bureau_uw.parsers.xml_to_dict import TransUnionAdapter
from bureau_uw.features.feature_engineering import build_features


# ---------------------------------------------------------------------------
# BSB24 attribute descriptions
# ---------------------------------------------------------------------------
BSB24_DESCRIPTIONS = {
    "NQB": "Any bankruptcy order or Scottish sequestration recorded in the last 36 months",
    "GQB": "Any IVA recorded in the last 36 months",
    "SD": "Currently Restricted",
    "QP": "Number of accounts entering default in the last 7-12 months",
    "TG": "Number of Accounts on Repayment Plans",
    "PD": "Currently insolvent?",
    "LQB": "Deceased Indicator",
    "TR": "Months since last default",
    "BH": "Number of accounts closed last 12 months",
    "PP": "Number of accounts entering default in the last 6 months",
    "WG": "Number of accounts opened last 12 months",
    "JRB": "Number of active accounts with a current status of '1'",
    "FB": "Number of active SHARE records",
    "ZHC": "Number of active short term loan accounts",
    "EB": "Number of SHARE records",
    "BIC": "Number of short term loan accounts opened in last month",
    "DB": "Number of unsatisfied CCJs",
    "PG": "Total value of accounts in code DM last 12 months",
    "UBC": "Worst payment status last 3 months on active accounts excluding historic defaults",
    "IH": "Number of active accounts in sector 1",
    "JGC": "Number of settled accounts in sector 1 (Loan and Instalment credit) in last 12 months",
    "OHC": "Number of opened accounts in sector 1 (Loan and Instalment credit) in last 12 months",
    "RQ": "Worst payment status in last month on active accounts in Sector 1 or Sector 3 created in last 36 months; excluding historic defaults",
    "EFC": "Number of settled accounts in sector 1 (Loan and Instalment credit)",
}

BSB24_CODES: List[str] = list(BSB24_DESCRIPTIONS.keys())

LEGACY_METRIC_KEYS: List[Tuple[str, str]] = [
    ("Internal score", "attr__internalscore"),
    ("Credit score class 11", "attr__creditscore.class_11.score"),
    ("Credit score class 10", "attr__creditscore.class_10.score"),
    ("Total accounts", "attr__summary.share.totalaccounts"),
    ("Active accounts", "attr__summary.share.totalactiveaccs"),
    ("Settled accounts", "attr__summary.share.totalsettledaccs"),
    ("Opened 6 months", "attr__summary.share.totalopened6months"),
    ("Defaults 36 months", "attr__summary.share.totaldefaults36months"),
    ("Searches 3 months", "attr__summary.searches.totalsearches3months"),
    ("Searches 12 months", "attr__summary.searches.totalsearches12months"),
    ("Total balances active", "attr__summary.indebt.totalbalancesactive"),
    ("Total balances revolve", "attr__summary.indebt.totalbalancesrevolve"),
    ("Total limits revolve", "attr__summary.indebt.totallimitsrevolve"),
    ("Balance/limit ratio revolve", "attr__summary.indebt.balancelimitratiorevolve"),
    ("Total balances loans", "attr__summary.indebt.totalbalancesloans"),
    ("ICH impaired credit", "attr__summary.ich.impairedcredit"),
    ("ICH unsecured", "attr__summary.ich.unsecured"),
    ("US foreclosures (12m)", "attr__summary.us.foreclosures12months"),
    ("US foreclosures (24m)", "attr__summary.us.foreclosures24months"),
    ("US foreclosures (ever)", "attr__summary.us.foreclosuresever"),
    ("US bankruptcy flag (ever)", "attr__summary.us.bankruptcyflagever"),
    ("US trades past due (ever)", "attr__summary.us.tradespastdueever"),
    ("US total past due amount (ever)", "attr__summary.us.totalpastdueever"),
    ("US major derogatory trades (ever)", "attr__summary.us.majorderogatoryever"),
    ("US months since major derogatory", "attr__summary.us.monthssincemajorderogatory"),
    ("US public record bankruptcies (ever)", "attr__summary.us.publicrecordbankruptciesever"),
    ("US months since bankruptcy filed", "attr__summary.us.monthssincebankruptcyfiled"),
    ("US inquiries (6m)", "attr__summary.us.inquiries6months"),
    ("US inquiries (30d)", "attr__summary.us.inquiries30days"),
    ("US revolving utilization (6m)", "attr__summary.us.utilizationrevolving6months"),
    ("US total open-trade balance (3m)", "attr__summary.us.totalbalanceopentrades3months"),
    ("US months on credit file", "attr__summary.us.monthsoncreditfile"),
    ("US age oldest trade", "attr__summary.us.ageoldesttrade"),
    ("US average age trades", "attr__summary.us.averageagetrades"),
    ("US trades with payment made (ever)", "attr__summary.us.tradeswithpaymentmadeever"),
    ("US pct past due to balance (ever)", "attr__summary.us.pctpastduetobalanceever"),
    ("US pct past due to balance ex-student (ever)", "attr__summary.us.pctpastduetobalanceexstudentever"),
    ("US bankcard utilization (3m)", "attr__summary.us.utilizationbankcard3months"),
    ("US bankcard utilization (6m)", "attr__summary.us.utilizationbankcard6months"),
    ("US retail revolving utilization (3m)", "attr__summary.us.utilizationretailrevolving3months"),
    ("US max revolving utilization (3m)", "attr__summary.us.maxutilizationrevolving3months"),
    ("Undeclared addresses unsearched", "attr__summary.links.totalundecaddressesunsearched"),
    ("Address message code", "attr__summary.summaryaddress.messagecode"),
]

LEGACY_DESCRIPTIONS: Dict[str, str] = {
    "attr__internalscore": "Internal pre-purchase score (0-100, cutoff 65)",
    "attr__creditscore.class_11.score": "Credit score (class 11) if present in the XML feed",
    "attr__creditscore.class_10.score": "Credit score (class 10) if present in the XML feed",
    "attr__summary.share.totalaccounts": "Total SHARE accounts (all accounts in bureau summary)",
    "attr__summary.share.totalactiveaccs": "Total active SHARE accounts",
    "attr__summary.share.totalsettledaccs": "Total settled SHARE accounts",
    "attr__summary.share.totalopened6months": "Accounts opened in last 6 months",
    "attr__summary.share.totaldefaults36months": "Defaults recorded within last 36 months (summary-level)",
    "attr__summary.searches.totalsearches3months": "Total bureau searches in last 3 months",
    "attr__summary.searches.totalsearches12months": "Total bureau searches in last 12 months",
    "attr__summary.indebt.totalbalancesactive": "Total balances across active accounts",
    "attr__summary.indebt.totalbalancesrevolve": "Total balances on revolving accounts",
    "attr__summary.indebt.totallimitsrevolve": "Total limits on revolving accounts",
    "attr__summary.indebt.balancelimitratiorevolve": "Revolving balance/limit utilisation ratio",
    "attr__summary.indebt.totalbalancesloans": "Total balances on loan/instalment accounts",
    "attr__summary.ich.impairedcredit": "Impaired credit indicator/value (ICH)",
    "attr__summary.ich.unsecured": "Unsecured indicator/value (ICH)",
    "attr__summary.us.foreclosures12months": "US: Number of foreclosure trades in last 12 months",
    "attr__summary.us.foreclosures24months": "US: Number of foreclosure trades in last 24 months",
    "attr__summary.us.foreclosuresever": "US: Number of foreclosure trades ever",
    "attr__summary.us.bankruptcyflagever": "US: Bankruptcy flag ever (0/1)",
    "attr__summary.us.tradespastdueever": "US: Number of trades with past due amount > 0 (ever)",
    "attr__summary.us.totalpastdueever": "US: Total past due amount on trades (ever)",
    "attr__summary.us.majorderogatoryever": "US: Number of trades with major derogatory history (ever)",
    "attr__summary.us.monthssincemajorderogatory": "US: Months since most recent major derogatory trade",
    "attr__summary.us.publicrecordbankruptciesever": "US: Number of public record bankruptcies ever",
    "attr__summary.us.monthssincebankruptcyfiled": "US: Months since most recent public record bankruptcy filed",
    "attr__summary.us.inquiries6months": "US: Number of inquiries in last 6 months",
    "attr__summary.us.inquiries30days": "US: Number of inquiries in last 30 days",
    "attr__summary.us.utilizationrevolving6months": "US: Revolving utilization on open revolving trades (6 months)",
    "attr__summary.us.totalbalanceopentrades3months": "US: Total balance on open trades reported in last 3 months",
    "attr__summary.us.monthsoncreditfile": "US: Months on credit file",
    "attr__summary.us.ageoldesttrade": "US: Age of oldest trade",
    "attr__summary.us.averageagetrades": "US: Average age of trades",
    "attr__summary.us.tradeswithpaymentmadeever": "US: Number of trades with payment made (ever)",
    "attr__summary.us.pctpastduetobalanceever": "US: Percent of past due amount to balance (ever)",
    "attr__summary.us.pctpastduetobalanceexstudentever": "US: Percent of past due amount to balance excluding student loans (ever)",
    "attr__summary.us.utilizationbankcard3months": "US: Overall bankcard utilization (3 months)",
    "attr__summary.us.utilizationbankcard6months": "US: Overall bankcard utilization (6 months)",
    "attr__summary.us.utilizationretailrevolving3months": "US: Retail revolving utilization (3 months)",
    "attr__summary.us.maxutilizationrevolving3months": "US: Maximum revolving utilization (3 months)",
    "attr__summary.links.totalundecaddressesunsearched": "Undeclared/unsearched addresses count",
    "attr__summary.summaryaddress.messagecode": "Address match/message code from bureau summary",
}

US_DASHBOARD_DESCRIPTION_OVERRIDES: Dict[str, str] = {
    "attr__summary.share.totalaccounts": "Total trades (Equifax summary)",
    "attr__summary.share.totalactiveaccs": "Open trades (Equifax summary)",
    "attr__summary.share.totalsettledaccs": "Closed/settled trades",
    "attr__summary.share.totalopened6months": "Trades opened in last 6 months",
    "attr__summary.share.totaldefaults36months": "Trades 30+ DPD or major derogatory (Equifax summary)",
    "attr__summary.share.totaldefaults12months": "Trades 30+ DPD or major derogatory opened in last 12 months",
    "attr__summary.share.totaldelinqs12months": "Trades with 30 DPD delinquency in last 12 months",
    "attr__summary.searches.totalsearches3months": "Inquiries in last 3 months",
    "attr__summary.searches.totalsearches12months": "Inquiries in last 12 months",
    "attr__summary.indebt.totalbalancesactive": "Total balance on open trades",
    "attr__summary.indebt.totalbalancesrevolve": "Computed revolving balance (trade-level)",
    "attr__summary.indebt.totallimitsrevolve": "Computed revolving credit limit (trade-level)",
    "attr__summary.indebt.balancelimitratiorevolve": "Equifax revolving utilization summary",
    "attr__summary.ich.impairedcredit": "US bankruptcy-related indicator (Equifax attr 5795)",
    "attr__summary.ich.unsecured": "US unsecured derogatory percentage",
}

# ---------------------------------------------------------------------------
# Tiering (review intensity)
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


def _safe_float(x: Any) -> float | None:
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


def _get(feats: Dict[str, Any], key: str) -> float | None:
    return _safe_float(feats.get(key))


def _norm_missing_to_blank(x: Any) -> Any:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    if str(x).strip() == "{ND}":
        return ""
    return x


CURRENCY_KEYS = {
    "attr__PG",
    "attr__summary.indebt.totalbalancesactive",
    "attr__summary.indebt.totalbalancesrevolve",
    "attr__summary.indebt.totalbalancesloans",
    "attr__summary.indebt.totallimitsrevolve",
}


def format_value(k: str, v: Any, *, currency_symbol: str = "\u00a3") -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    if isinstance(v, str) and (v.startswith("\u00a3") or v.startswith("$")):
        return v
    if k in CURRENCY_KEYS:
        fv = _safe_float(v)
        if fv is not None:
            return f"{currency_symbol}{fv:,.0f}"
        return str(v)
    if isinstance(v, (int, float, np.number)):
        fv = float(v)
        if fv.is_integer():
            return f"{int(fv)}"
        return f"{fv:,.2f}"
    return str(v)


def _fmt_val(x: Any) -> str:
    return format_value("", _norm_missing_to_blank(x))


def tier_from_score(score_0_100: float) -> str:
    for name, lo, hi in TIER_BANDS:
        if lo <= score_0_100 < hi:
            return name
    return "Detailed Review"


def get_policy_decision(score_0_100: float, policy: Dict[str, Any] | None) -> str:
    """Map score to APPROVE / REFER / DECLINE using policy bands. Returns 'UNKNOWN' if no policy."""
    if not policy or "bands" not in policy:
        return "UNKNOWN"
    bands = policy["bands"]
    if score_0_100 >= _band_min(bands.get("APPROVE")) and score_0_100 <= _band_max(bands.get("APPROVE")):
        return "APPROVE"
    if score_0_100 >= _band_min(bands.get("REFER")) and score_0_100 <= _band_max(bands.get("REFER")):
        return "REFER"
    if score_0_100 >= _band_min(bands.get("DECLINE")) and score_0_100 <= _band_max(bands.get("DECLINE")):
        return "DECLINE"
    return "UNKNOWN"


def _band_min(b: Dict[str, Any] | None) -> float:
    if not b:
        return -1e9
    return float(b.get("min_score", -1e9))


def _band_max(b: Dict[str, Any] | None) -> float:
    if not b:
        return 1e9
    return float(b.get("max_score", 1e9))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
@dataclass
class ScoredApplicant:
    parsed: Any
    feats: Dict[str, Any]
    p_paid: float
    score_0_100: float
    tier: str


def load_artifacts(artifact_dir: Path) -> Tuple[Any, List[str], pd.DataFrame | None, Dict[str, Any] | None]:
    """
    Load model pipeline, feature list, coefficients, and policy from artifact directory.

    Returns:
        (pipe, feature_list, coef_df, policy_dict)
        coef_df and policy_dict may be None if files are missing.
    """
    artifact_dir = Path(artifact_dir)
    model_path = artifact_dir / "model_pipeline.joblib"
    feature_list_path = artifact_dir / "feature_list.json"
    coef_path = artifact_dir / "coefficients.csv"
    policy_path = artifact_dir / "policy.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model_pipeline.joblib in {artifact_dir}")
    if not feature_list_path.exists():
        raise FileNotFoundError(f"Missing feature_list.json in {artifact_dir}")
    pipe = joblib.load(model_path)
    with open(feature_list_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    feature_list = raw if isinstance(raw, list) else list(raw)
    coef_df = pd.read_csv(coef_path) if coef_path.exists() else None
    policy = None
    if policy_path.exists():
        with open(policy_path, "r", encoding="utf-8") as f:
            policy = json.load(f)
    return pipe, feature_list, coef_df, policy


def score_xml(upload_bytes: bytes, pipe: Any, feature_list: List[str]) -> ScoredApplicant:
    adapter = TransUnionAdapter()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "upload.xml"
        p.write_bytes(upload_bytes)
        parsed = adapter.parse_file(p, strict=False)
    feats = build_features(parsed)
    row = {f: feats.get(f) for f in feature_list}
    X = pd.DataFrame([row])
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    p_paid = float(pipe.predict_proba(X)[:, 1][0])
    score_0_100 = round(100.0 * p_paid, 2)
    tier = tier_from_score(score_0_100)
    return ScoredApplicant(parsed=parsed, feats=feats, p_paid=p_paid, score_0_100=score_0_100, tier=tier)


def score_input(
    upload_bytes: bytes,
    pipe: Any,
    feature_list: List[str],
    *,
    input_format: str = "xml",
    bureau: str = "transunion",
) -> ScoredApplicant:
    fmt = (input_format or "").strip().lower()
    br = (bureau or "").strip().lower()

    if fmt == "xml":
        return score_xml(upload_bytes, pipe, feature_list)

    if fmt != "json":
        raise ValueError(f"Unsupported input format '{input_format}'. Expected 'xml' or 'json'.")
    if br not in {"equifax", "equifax_us", "us"}:
        raise ValueError(
            f"JSON scoring is currently configured for Equifax US only. Received bureau '{bureau}'."
        )

    adapter = EquifaxJsonAdapter()
    try:
        payload = json.loads(upload_bytes.decode("utf-8"))
    except Exception:
        payload = json.loads(upload_bytes.decode("utf-8", errors="ignore"))
    parsed = adapter.parse_payload(payload, source_name="upload.json")
    feats = build_features(parsed)
    row = {f: feats.get(f) for f in feature_list}
    X = pd.DataFrame([row])
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    p_paid = float(pipe.predict_proba(X)[:, 1][0])
    score_0_100 = round(100.0 * p_paid, 2)
    tier = tier_from_score(score_0_100)
    return ScoredApplicant(parsed=parsed, feats=feats, p_paid=p_paid, score_0_100=score_0_100, tier=tier)


# ---------------------------------------------------------------------------
# Risk Pillars
# ---------------------------------------------------------------------------
RISK_TO_NUM = {"Unknown": 0, "Low": 1, "Moderate": 2, "Elevated": 3}


def _bsb(code: str, feats: Dict[str, Any]) -> float | None:
    return _get(feats, f"attr__{code}")


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


def pillar_credit_quality(feats: Dict[str, Any]) -> Tuple[str, List[str]]:
    notes: List[str] = []
    level = "Low"
    if _is_us_equifax(feats):
        f12 = _get(feats, "attr__summary.us.foreclosures12months") or 0.0
        f24 = _get(feats, "attr__summary.us.foreclosures24months") or 0.0
        fev = _get(feats, "attr__summary.us.foreclosuresever") or 0.0
        bk = _get(feats, "attr__summary.us.bankruptcyflagever") or 0.0
        bk_cnt = _get(feats, "attr__summary.us.publicrecordbankruptciesever") or 0.0
        bk_mos = _get(feats, "attr__summary.us.monthssincebankruptcyfiled")
        md_ever = _get(feats, "attr__summary.us.majorderogatoryever") or 0.0
        md_mos = _get(feats, "attr__summary.us.monthssincemajorderogatory")
        oldest_trade = _get(feats, "attr__summary.us.ageoldesttrade")
        months_file = _get(feats, "attr__summary.us.monthsoncreditfile")
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

    lqb = _bsb("LQB", feats)
    if lqb is not None and lqb > 0:
        notes.append("LQB: DECEASED INDICATOR PRESENT")
        level = "Elevated"
    pd_ = _bsb("PD", feats) or 0.0
    nqb = _bsb("NQB", feats) or 0.0
    gqb = _bsb("GQB", feats) or 0.0
    sd = _bsb("SD", feats) or 0.0
    db = _bsb("DB", feats) or 0.0
    ubc = _bsb("UBC", feats)
    rq = _bsb("RQ", feats)
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
        bsb_present = any(_bsb(c, feats) is not None for c in ("NQB", "GQB", "SD", "PD", "DB"))
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
        bal_active = _get(feats, "attr__summary.indebt.totalbalancesactive")
        open_bal_3m = _get(feats, "attr__summary.us.totalbalanceopentrades3months")
        lim_rev = _get(feats, "attr__summary.indebt.totallimitsrevolve")
        util3 = _get(feats, "attr__summary.indebt.balancelimitratiorevolve")
        util6 = _get(feats, "attr__summary.us.utilizationrevolving6months")
        util_bc3 = _get(feats, "attr__summary.us.utilizationbankcard3months")
        util_bc6 = _get(feats, "attr__summary.us.utilizationbankcard6months")
        util_retail3 = _get(feats, "attr__summary.us.utilizationretailrevolving3months")
        util_max3 = _get(feats, "attr__summary.us.maxutilizationrevolving3months")
        total_accs = _get(feats, "attr__summary.share.totalaccounts")
        active_accs = _get(feats, "attr__summary.share.totalactiveaccs")
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

    wg = _bsb("WG", feats)
    zhc = _bsb("ZHC", feats)
    bic = _bsb("BIC", feats)
    eb = _bsb("EB", feats)
    fb = _bsb("FB", feats)
    ih = _bsb("IH", feats)
    ohc = _bsb("OHC", feats)
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
    bal_active = _get(feats, "attr__summary.indebt.totalbalancesactive")
    bal_rev = _get(feats, "attr__summary.indebt.totalbalancesrevolve")
    lim_rev = _get(feats, "attr__summary.indebt.totallimitsrevolve")
    ratio = _get(feats, "attr__summary.indebt.balancelimitratiorevolve")
    bal_loans = _get(feats, "attr__summary.indebt.totalbalancesloans")
    total_accs = _get(feats, "attr__summary.share.totalaccounts")
    active_accs = _get(feats, "attr__summary.share.totalactiveaccs")
    if bal_active is not None:
        notes.append(f"Total active balances: \u00a3{bal_active:,.0f}")
    if bal_rev is not None:
        notes.append(f"Revolving balances: \u00a3{bal_rev:,.0f}")
    if lim_rev is not None:
        notes.append(f"Revolving limits: \u00a3{lim_rev:,.0f}")
    if ratio is not None:
        notes.append(f"Revolving utilisation: {ratio:.0f}%")
    if bal_loans is not None:
        notes.append(f"Loan balances: \u00a3{bal_loans:,.0f}")
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
        past_due_cnt = _get(feats, "attr__summary.us.tradespastdueever")
        past_due_amt = _get(feats, "attr__summary.us.totalpastdueever")
        pct_past_due = _get(feats, "attr__summary.us.pctpastduetobalanceever")
        pct_past_due_exs = _get(feats, "attr__summary.us.pctpastduetobalanceexstudentever")
        md_ever = _get(feats, "attr__summary.us.majorderogatoryever")
        trades_paid_ever = _get(feats, "attr__summary.us.tradeswithpaymentmadeever")
        defaults12 = _get(feats, "attr__summary.share.totaldefaults12months")
        defaults36 = _get(feats, "attr__summary.share.totaldefaults36months")
        delinqs12 = _get(feats, "attr__summary.share.totaldelinqs12months")
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

    pp = _bsb("PP", feats)
    qp = _bsb("QP", feats)
    tg = _bsb("TG", feats)
    ubc = _bsb("UBC", feats)
    rq = _bsb("RQ", feats)
    pg = _bsb("PG", feats)
    jrb = _bsb("JRB", feats)
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
    delinqs12 = _get(feats, "attr__summary.share.totaldelinqs12months")
    worse12 = _get(feats, "attr__summary.share.worsepaystatus12months")
    worse36 = _get(feats, "attr__summary.share.worsepaystatus36months")
    defaults36 = _get(feats, "attr__summary.share.totaldefaults36months")
    unsecured = _get(feats, "attr__summary.ich.unsecured")
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
        inq_30d = _get(feats, "attr__summary.us.inquiries30days")
        inq_6m = _get(feats, "attr__summary.us.inquiries6months")
        inq_12m = _get(feats, "attr__summary.searches.totalsearches12months")
        opened6 = _get(feats, "attr__summary.share.totalopened6months")
        md_mos = _get(feats, "attr__summary.us.monthssincemajorderogatory")
        avg_age = _get(feats, "attr__summary.us.averageagetrades")
        oldest_trade = _get(feats, "attr__summary.us.ageoldesttrade")
        months_file = _get(feats, "attr__summary.us.monthsoncreditfile")
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

    tr = _bsb("TR", feats)
    bh = _bsb("BH", feats)
    wg = _bsb("WG", feats)
    efc = _bsb("EFC", feats)
    jgc = _bsb("JGC", feats)
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
    s3 = _get(feats, "attr__summary.searches.totalsearches3months")
    s12 = _get(feats, "attr__summary.searches.totalsearches12months")
    opened6 = _get(feats, "attr__summary.share.totalopened6months")
    settled = _get(feats, "attr__summary.share.totalsettledaccs")
    undec_addr = _get(feats, "attr__summary.links.totalundecaddressesunsearched")
    paf = _get(feats, "attr__summary.summaryaddress.pafvalid")
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


def build_pillars_df(feats: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for pillar_name, fn in [
        ("Credit Quality", pillar_credit_quality),
        ("Exposure Pressure", pillar_exposure_pressure),
        ("Conduct", pillar_conduct),
        ("Stability", pillar_stability),
    ]:
        level, notes = fn(feats)
        rows.append({
            "Pillar": pillar_name,
            "Risk": level,
            "RiskScore": RISK_TO_NUM.get(level, 0),
            "Notes": " | ".join(notes),
        })
    return pd.DataFrame(rows)


def model_feature_table(
    feature_list: List[str],
    coef_df: pd.DataFrame | None,
    feats: Dict[str, Any],
) -> pd.DataFrame:
    is_us_equifax = _is_us_equifax(feats)
    currency_symbol = "$" if is_us_equifax else "\u00a3"
    wanted_features: List[str] = []
    for c in BSB24_CODES:
        wanted_features.append(f"attr__{c}")
    for _, k in LEGACY_METRIC_KEYS:
        wanted_features.append(k)
    seen = set()
    rows: List[Dict[str, Any]] = []
    for fk in wanted_features:
        if fk in seen:
            continue
        seen.add(fk)
        raw = feats.get(fk, None)
        raw = _norm_missing_to_blank(raw)
        desc = ""
        if fk.startswith("attr__"):
            code = fk.replace("attr__", "", 1)
            desc = BSB24_DESCRIPTIONS.get(code, "") or LEGACY_DESCRIPTIONS.get(fk, "")
        else:
            desc = LEGACY_DESCRIPTIONS.get(fk, "")
        if is_us_equifax:
            desc = US_DASHBOARD_DESCRIPTION_OVERRIDES.get(fk, desc)
        rows.append({"Metric": fk, "Description": desc, "Value": format_value(fk, raw, currency_symbol=currency_symbol)})
    df = pd.DataFrame(rows)
    df["_grp"] = df["Metric"].map(
        lambda x: 0 if str(x).startswith("attr__") and str(x).replace("attr__", "", 1) in BSB24_DESCRIPTIONS else 1
    )
    df = df.sort_values(by=["_grp", "Metric"]).drop(columns=["_grp"])
    return df


def key_metrics_df(feature_list: List[str], coef_df: pd.DataFrame | None, feats: Dict[str, Any]) -> pd.DataFrame:
    return model_feature_table(feature_list=feature_list, coef_df=coef_df, feats=feats)


def build_export_payload(
    applicant: ScoredApplicant,
    customer_name: str,
    loan_number: str,
    uw_notes: str,
    pillars: pd.DataFrame,
    metrics: pd.DataFrame,
    drivers: pd.DataFrame,
) -> Dict[str, Any]:
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "exported_utc": now,
        "customer_name": customer_name,
        "loan_number": loan_number,
        "underwriter_notes": uw_notes,
        "review": {
            "review_level": applicant.tier,
            "guidance": TIER_GUIDANCE.get(applicant.tier, ""),
        },
        "risk_pillars": pillars.drop(columns=["RiskScore"], errors="ignore").to_dict(orient="records"),
        "key_metrics": metrics.to_dict(orient="records"),
        "top_drivers": drivers.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_bureau_xml(
    xml_bytes: bytes,
    artifact_dir: Path | str,
    *,
    customer_name: str = "",
    loan_number: str = "",
    uw_notes: str = "",
    pipe: Any = None,
    feature_list: List[str] | None = None,
    coef_df: pd.DataFrame | None = None,
    policy: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Score a single TransUnion bureau XML and return a JSON-serializable result.

    Use this from your loan management platform: pass XML bytes and (optionally)
    pre-loaded artifacts. If pipe/feature_list are not provided, they are loaded
    from artifact_dir.

    Args:
        xml_bytes: Raw XML file content (TransUnion bureau response).
        artifact_dir: Path to artifact bundle (used if pipe/feature_list not provided).
        customer_name: Optional customer name for export payload.
        loan_number: Optional loan reference for export payload.
        uw_notes: Optional underwriter notes for export payload.
        pipe, feature_list, coef_df, policy: Optional pre-loaded artifacts (avoids repeated disk I/O).

    Returns:
        Dict with keys:
          - score_0_100: float
          - p_paid: float (probability)
          - tier: str (e.g. "Standard Review")
          - tier_guidance: str
          - policy_decision: "APPROVE" | "REFER" | "DECLINE" | "UNKNOWN"
          - risk_pillars: list of {Pillar, Risk, Notes}
          - key_metrics: list of {Metric, Description, Value}
          - top_drivers: list of {Metric, Description, Value} (same as key_metrics)
          - export_payload: full payload for JSON/PDF/CSV export (customer_name, loan_number, etc.)
    """
    artifact_dir = Path(artifact_dir)
    if pipe is None or feature_list is None:
        loaded = load_artifacts(artifact_dir)
        pipe = loaded[0]
        feature_list = loaded[1]
        coef_df = coef_df or loaded[2]
        policy = policy or loaded[3]
    else:
        if policy is None and (artifact_dir / "policy.json").exists():
            with open(artifact_dir / "policy.json", "r", encoding="utf-8") as f:
                policy = json.load(f)

    applicant = score_input(
        xml_bytes,
        pipe,
        feature_list,
        input_format="xml",
        bureau="transunion",
    )
    pillars_df = build_pillars_df(applicant.feats)
    drivers_df = model_feature_table(feature_list, coef_df, applicant.feats)
    metrics_df = key_metrics_df(feature_list, coef_df, applicant.feats)

    export_payload = build_export_payload(
        applicant=applicant,
        customer_name=customer_name,
        loan_number=loan_number,
        uw_notes=uw_notes,
        pillars=pillars_df,
        metrics=metrics_df,
        drivers=drivers_df,
    )

    policy_decision = get_policy_decision(applicant.score_0_100, policy)

    return {
        "score_0_100": applicant.score_0_100,
        "p_paid": applicant.p_paid,
        "tier": applicant.tier,
        "tier_guidance": TIER_GUIDANCE.get(applicant.tier, ""),
        "policy_decision": policy_decision,
        "risk_pillars": pillars_df.drop(columns=["RiskScore"], errors="ignore").to_dict(orient="records"),
        "key_metrics": metrics_df.to_dict(orient="records"),
        "top_drivers": drivers_df.to_dict(orient="records"),
        "export_payload": export_payload,
    }


def score_bureau_input(
    upload_bytes: bytes,
    artifact_dir: Path | str,
    *,
    input_format: str,
    bureau: str,
    customer_name: str = "",
    loan_number: str = "",
    uw_notes: str = "",
    pipe: Any = None,
    feature_list: List[str] | None = None,
    coef_df: pd.DataFrame | None = None,
    policy: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Score XML/JSON bureau payloads and return a JSON-serializable result."""
    artifact_dir = Path(artifact_dir)
    if pipe is None or feature_list is None:
        loaded = load_artifacts(artifact_dir)
        pipe = loaded[0]
        feature_list = loaded[1]
        coef_df = coef_df or loaded[2]
        policy = policy or loaded[3]
    else:
        if policy is None and (artifact_dir / "policy.json").exists():
            with open(artifact_dir / "policy.json", "r", encoding="utf-8") as f:
                policy = json.load(f)

    applicant = score_input(
        upload_bytes,
        pipe,
        feature_list,
        input_format=input_format,
        bureau=bureau,
    )
    pillars_df = build_pillars_df(applicant.feats)
    drivers_df = model_feature_table(feature_list, coef_df, applicant.feats)
    metrics_df = key_metrics_df(feature_list, coef_df, applicant.feats)

    export_payload = build_export_payload(
        applicant=applicant,
        customer_name=customer_name,
        loan_number=loan_number,
        uw_notes=uw_notes,
        pillars=pillars_df,
        metrics=metrics_df,
        drivers=drivers_df,
    )

    policy_decision = get_policy_decision(applicant.score_0_100, policy)

    return {
        "score_0_100": applicant.score_0_100,
        "p_paid": applicant.p_paid,
        "tier": applicant.tier,
        "tier_guidance": TIER_GUIDANCE.get(applicant.tier, ""),
        "policy_decision": policy_decision,
        "risk_pillars": pillars_df.drop(columns=["RiskScore"], errors="ignore").to_dict(orient="records"),
        "key_metrics": metrics_df.to_dict(orient="records"),
        "top_drivers": drivers_df.to_dict(orient="records"),
        "export_payload": export_payload,
    }
