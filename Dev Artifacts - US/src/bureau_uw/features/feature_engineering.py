from __future__ import annotations

import math
import re
from typing import Any, Dict, List

from bureau_uw.parsers.xml_to_dict import ParseResult


_NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")

BSB_24_CODES: List[str] = [
    "NQB", "GQB", "SD", "QP", "TG", "PD", "LQB", "TR", "BH", "PP", "WG", "JRB",
    "FB", "ZHC", "EB", "BIC", "DB", "PG", "UBC", "IH", "JGC", "OHC", "RQ", "EFC",
]

LEGACY_SUMMARY_KEYS: List[str] = [
    "summary.share.totalsettledaccs",
    "summary.share.totalaccounts",
    "summary.share.totalactiveaccs",
    "summary.share.totalopened6months",
    "summary.share.totaldefaults36months",
    "summary.share.totaldefaults12months",
    "summary.share.totaldelinqs12months",
    "summary.share.worsepaystatus12months",
    "summary.share.worsepaystatus36months",
    "summary.searches.totalsearches3months",
    "summary.searches.totalsearches12months",
    "summary.searches.totalhomecreditsearches3months",
    "summary.indebt.totalbalancesactive",
    "summary.indebt.totalbalancesrevolve",
    "summary.indebt.totallimitsrevolve",
    "summary.indebt.balancelimitratiorevolve",
    "summary.indebt.totalbalancesloans",
    "summary.ich.impairedcredit",
    "summary.ich.unsecured",
    "summary.ich.secured",
    "summary.ich.judgment",
    "summary.ich.iva",
    "summary.ich.boss",
    "summary.bais.currentlyinsolvent",
    "summary.bais.restricted",
    "summary.bais.totaldischarged",
    "summary.judgments.totalactive",
    "summary.judgments.totalsatisfied",
    "summary.links.totalundecaddressesunsearched",
    "summary.links.totalundecaddressessearched",
    "summary.links.totalundecaliases",
    "summary.links.totalundecassociates",
    "summary.summaryaddress.messagecode",
    "summary.summaryaddress.pafvalid",
    "summary.notices.nocflag",
    "summary.notices.totaldisputes",
    "summary.us.foreclosures12months",
    "summary.us.foreclosures24months",
    "summary.us.foreclosuresever",
    "summary.us.bankruptcyflagever",
    "summary.us.tradespastdueever",
    "summary.us.totalpastdueever",
    "summary.us.majorderogatoryever",
    "summary.us.monthssincemajorderogatory",
    "summary.us.publicrecordbankruptciesever",
    "summary.us.monthssincebankruptcyfiled",
    "summary.us.inquiries6months",
    "summary.us.inquiries30days",
    "summary.us.utilizationrevolving6months",
    "summary.us.totalbalanceopentrades3months",
    "summary.us.monthsoncreditfile",
    "summary.us.ageoldesttrade",
    "summary.us.averageagetrades",
    "summary.us.tradeswithpaymentmadeever",
    "summary.us.pctpastduetobalanceever",
    "summary.us.pctpastduetobalanceexstudentever",
    "summary.us.utilizationbankcard3months",
    "summary.us.utilizationbankcard6months",
    "summary.us.utilizationretailrevolving3months",
    "summary.us.maxutilizationrevolving3months",
    "creditscore.class_10.score",
    "creditscore.class_11.score",
    "internalscore",
]


def _to_float(x: str) -> float | None:
    x = (x or "").strip()
    if not x:
        return None
    x = x.replace(",", "")
    if x.lower() in {"na", "n/a", "null", "none", "unknown", "{nd}"}:
        return None
    if _NUM_RE.match(x):
        try:
            return float(x)
        except Exception:
            return None
    return None


def build_features(parsed: ParseResult) -> Dict[str, Any]:
    """Build an engineered feature row from ParseResult.

    Extracts:
    - Structural counts (cnt__*)
    - BSB24 engineered bureau attributes (attr__CODE)
    - Legacy summary-level metrics (attr__summary.*, attr__creditscore.*)
    - Thin-file / missingness indicators (meta__*)
    """
    feats: Dict[str, Any] = {}
    attrs = parsed.attrs or {}

    # --- structural counts (stable) ---
    for k, v in (parsed.counts or {}).items():
        feats[f"cnt__{k}"] = int(v)

    # --- BSB 24 core attributes ---
    for code in BSB_24_CODES:
        raw_val = attrs.get(code)

        if raw_val is None:
            feats[f"attr__{code}"] = None
            continue

        val_str = str(raw_val).strip()

        if not val_str or val_str == "{ND}":
            feats[f"attr__{code}"] = None
            continue

        f_num = _to_float(val_str)
        if f_num is not None and math.isfinite(f_num):
            feats[f"attr__{code}"] = f_num
        else:
            feats[f"attr__{code}"] = None

    # --- Legacy summary / creditscore attributes ---
    for key in LEGACY_SUMMARY_KEYS:
        raw_val = attrs.get(key)

        if raw_val is None:
            feats[f"attr__{key}"] = None
            continue

        val_str = str(raw_val).strip()

        if not val_str or val_str.lower() in {"{nd}", "na", "n/a", "null", "none", "unknown"}:
            feats[f"attr__{key}"] = None
            continue

        f_num = _to_float(val_str)
        if f_num is not None and math.isfinite(f_num):
            feats[f"attr__{key}"] = f_num
        else:
            feats[f"attr__{key}"] = val_str

    # --- thin-file / missingness indicators ---
    feats["meta__n_attr_items"] = int(parsed.meta.get("attr_items", 0) if parsed.meta else 0)
    feats["meta__n_flat_items"] = int(parsed.meta.get("flat_items", 0) if parsed.meta else 0)
    feats["meta__has_namespace"] = int(1 if (parsed.meta or {}).get("has_namespace") else 0)

    return feats
