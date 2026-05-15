from __future__ import annotations

from typing import Any, Dict, List


# Canonical keys already consumed by feature_engineering/scoring.
# Values are Equifax Core Essentials "Attribute Number" identifiers.
EQUIFAX_CANONICAL_ATTRIBUTE_MAP: Dict[str, List[str]] = {
    "summary.share.totalaccounts": ["4140"],  # Number of Trades
    "summary.share.totalactiveaccs": ["4173"],  # Number of Open Trades
    "summary.share.totalopened6months": ["3967"],  # Number of Trades Opened in 6 Months
    "summary.share.totaldefaults36months": ["3843"],  # Trades currently 30+ DPD or major derogatory (ever)
    "summary.share.totaldefaults12months": ["3885"],  # Trades currently 30+ DPD/major derog opened in 12m
    "summary.share.totaldelinqs12months": ["4244"],  # Trades worst rating 30 DPD delinquency in 12m
    "summary.searches.totalsearches3months": ["5710"],  # Number of inquiries in 3m with dedupe
    "summary.searches.totalsearches12months": ["5712"],  # Number of inquiries in 12m with dedupe
    "summary.indebt.totalbalancesactive": ["4749"],  # Total balance on open trades (ever)
    "summary.indebt.totallimitsrevolve": ["4628"],  # Total credit limit/high credit of open trades (6m)
    "summary.indebt.balancelimitratiorevolve": ["4945"],  # Overall utilization on open revolving trades (3m)
    "summary.ich.impairedcredit": ["5795"],  # Flag of bankruptcies ever
    "summary.ich.unsecured": ["5531"],  # % unsecured trades 30+ DPD or major derogatory (ever)
    # US adverse indicators used in dashboards and risk pillar logic.
    "summary.us.foreclosures12months": ["4562"],
    "summary.us.foreclosures24months": ["4563"],
    "summary.us.foreclosuresever": ["4564"],
    "summary.us.bankruptcyflagever": ["5795"],
    # Next US mapping batch (conduct/exposure/recency).
    "summary.us.tradespastdueever": ["4202"],
    "summary.us.totalpastdueever": ["4841"],
    "summary.us.majorderogatoryever": ["3542"],
    "summary.us.monthssincemajorderogatory": ["3190"],
    "summary.us.publicrecordbankruptciesever": ["5776"],
    "summary.us.monthssincebankruptcyfiled": ["5787"],
    "summary.us.inquiries6months": ["5711"],
    "summary.us.inquiries30days": ["5749"],
    "summary.us.utilizationrevolving6months": ["4946"],
    "summary.us.totalbalanceopentrades3months": ["4747"],
    # Top-10 additive mapping batch.
    "summary.us.monthsoncreditfile": ["5798"],
    "summary.us.ageoldesttrade": ["3001"],
    "summary.us.averageagetrades": ["3108"],
    "summary.us.tradeswithpaymentmadeever": ["3497"],
    "summary.us.pctpastduetobalanceever": ["5105"],
    "summary.us.pctpastduetobalanceexstudentever": ["5106"],
    "summary.us.utilizationbankcard3months": ["4932"],
    "summary.us.utilizationbankcard6months": ["4933"],
    "summary.us.utilizationretailrevolving3months": ["4938"],
    "summary.us.maxutilizationrevolving3months": ["5048"],
}


def _normalize_attr_id(v: Any) -> str:
    s = str(v or "").strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def apply_equifax_canonical_mapping(
    equifax_attrs: Dict[str, str],
    *,
    preserve_existing: Dict[str, str] | None = None,
) -> Dict[str, str]:
    """Map Equifax attribute IDs into canonical summary keys."""
    out: Dict[str, str] = dict(preserve_existing or {})

    for canonical_key, candidate_ids in EQUIFAX_CANONICAL_ATTRIBUTE_MAP.items():
        for attr_id in candidate_ids:
            norm_id = _normalize_attr_id(attr_id)
            lookup_key = f"eqx.attr.{norm_id}"
            value = equifax_attrs.get(lookup_key)
            if value is None or str(value).strip() == "":
                continue
            out[canonical_key] = str(value).strip()
            break

    return out
