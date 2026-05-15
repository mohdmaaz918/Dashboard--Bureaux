from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Tuple

from bureau_uw.parsers.equifax_feature_mapping import (
    EQUIFAX_CANONICAL_ATTRIBUTE_MAP,
    apply_equifax_canonical_mapping,
)
from bureau_uw.parsers.xml_to_dict import BureauAdapter, ParseResult

_PLACEHOLDER_NUMBERS = {
    Decimal("9.9996"),
    Decimal("9.9997"),
    Decimal("9.9998"),
    Decimal("9.9999"),
    Decimal("96"),
    Decimal("97"),
    Decimal("98"),
    Decimal("99"),
    Decimal("99.9996"),
    Decimal("99.9997"),
    Decimal("99.9998"),
    Decimal("99.9999"),
    Decimal("996"),
    Decimal("997"),
    Decimal("998"),
    Decimal("999"),
    Decimal("9999.9996"),
    Decimal("9999.9997"),
    Decimal("9999.9998"),
    Decimal("9999.9999"),
    Decimal("99999.9996"),
    Decimal("99999.9997"),
    Decimal("99999.9998"),
    Decimal("99999.9999"),
    Decimal("999999996"),
    Decimal("999999997"),
    Decimal("999999998"),
    Decimal("999999999"),
    Decimal("9995"),
    Decimal("995"),
    Decimal("95"),
    Decimal("9.95"),
    Decimal("999.95"),
    Decimal("99995"),
    Decimal("99.95"),
    Decimal("999995"),
    Decimal("9999.95"),
    Decimal("9.9995"),
    Decimal("9999.9995"),
    Decimal("99.9995"),
    Decimal("9996"),
    Decimal("9997"),
    Decimal("9998"),
    Decimal("9999"),
    Decimal("99996"),
    Decimal("99997"),
    Decimal("99998"),
    Decimal("99999"),
}

# Ratio/percentage-style attributes that should never carry sentinel magnitudes.
_RATIO_ATTR_IDS = {
    "4932", "4933", "4935", "4936", "4938", "4939",
    "4941", "4942", "4943", "4944", "4945", "4946",
    "5040", "5041", "5044", "5046", "5047", "5048", "5050",
    "5105", "5106", "5113", "5348", "5364", "5396", "5689",
}


def _flatten_json(
    obj: Any,
    out: Dict[str, str],
    *,
    prefix: str = "",
    max_items: int = 50000,
) -> None:
    if len(out) >= max_items:
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            nxt = f"{prefix}.{k}" if prefix else str(k)
            _flatten_json(v, out, prefix=nxt, max_items=max_items)
            if len(out) >= max_items:
                return
        return

    if isinstance(obj, list):
        for i, v in enumerate(obj):
            nxt = f"{prefix}[{i}]"
            _flatten_json(v, out, prefix=nxt, max_items=max_items)
            if len(out) >= max_items:
                return
        return

    if obj is None:
        return
    if isinstance(obj, bool):
        out[prefix] = "1" if obj else "0"
        return
    out[prefix] = str(obj).strip()


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        raw = str(v).replace(",", "").strip()
        if _is_placeholder_numeric(raw):
            return None
        return float(raw)
    except Exception:
        return None


def _is_placeholder_numeric(v: Any) -> bool:
    s = str(v or "").strip()
    if not s:
        return False
    try:
        d = Decimal(s)
    except (InvalidOperation, ValueError):
        return False
    return d in _PLACEHOLDER_NUMBERS


def _normalize_equifax_value(v: Any, *, attr_id: str = "") -> str:
    s = str(v or "").strip()
    if not s:
        return ""
    if _is_placeholder_numeric(s):
        return "0"

    fv = _safe_float(s)
    aid = str(attr_id or "").strip()
    if fv is not None and aid in _RATIO_ATTR_IDS:
        # Ratios/percentages in this feed are expected on a bounded scale.
        # Very large magnitudes are sentinel placeholders, not real values.
        if fv >= 1000:
            return "0"
        # Some Equifax ratio attributes are delivered as implied hundredths.
        # Example: 421 means 4.21%, not 421%.
        if fv > 100:
            scaled = fv / 100.0
            return f"{scaled:.2f}".rstrip("0").rstrip(".")
        # Additional sentinel bands that frequently leak through as pseudo-values.
        if fv in {95.0, 96.0, 97.0, 98.0, 99.0, 995.0, 996.0, 997.0, 998.0, 999.0}:
            return "0"

    if s.lower() in {"nan", "inf", "-inf"}:
        return "0"
    return s


def _is_placeholder_or_sentinel(v: Any, *, attr_id: str = "") -> bool:
    s = str(v or "").strip()
    if not s:
        return False
    if _is_placeholder_numeric(s):
        return True
    fv = _safe_float(s)
    aid = str(attr_id or "").strip()
    if fv is not None and aid in _RATIO_ATTR_IDS:
        if fv >= 1000:
            return True
        if fv in {95.0, 96.0, 97.0, 98.0, 99.0, 995.0, 996.0, 997.0, 998.0, 999.0}:
            return True
    return s.lower() in {"nan", "inf", "-inf"}


def _is_nonzero_numeric(v: Any) -> bool:
    fv = _safe_float(v)
    return fv is not None and abs(fv) > 1e-9


def _get_equifax_consumer_root(payload: Dict[str, Any]) -> Dict[str, Any]:
    consumers = payload.get("consumers", {})
    rows = consumers.get("equifaxUSConsumerCreditReport", [])
    if isinstance(rows, list) and rows:
        first = rows[0]
        if isinstance(first, dict):
            return first
    return {}


class EquifaxJsonAdapter(BureauAdapter):
    bureau_name = "equifax_us"

    def parse_file(self, json_path: Path, strict: bool = True) -> ParseResult:
        json_path = Path(json_path)
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            if strict:
                raise
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
            except Exception as exc:
                return ParseResult(
                    flat={},
                    attrs={},
                    counts={},
                    meta={"file": json_path.name, "bureau": self.bureau_name, "error": str(exc)},
                )

        return self.parse_payload(payload, source_name=json_path.name)

    def parse_payload(self, payload: Dict[str, Any], source_name: str = "upload.json") -> ParseResult:
        flat: Dict[str, str] = {}
        _flatten_json(payload, flat)

        consumer = _get_equifax_consumer_root(payload)
        attrs, mapping_diag = self._extract_attrs(payload, consumer)
        counts = self._structural_counts(consumer)
        meta = {
            "file": source_name,
            "bureau": self.bureau_name,
            "flat_items": len(flat),
            "attr_items": len(attrs),
            "has_namespace": False,
            "equifax_mapping": mapping_diag,
        }
        return ParseResult(flat=flat, attrs=attrs, counts=counts, meta=meta)

    def _extract_attrs(self, payload: Dict[str, Any], consumer: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, Any]]:
        attrs: Dict[str, str] = {}
        trades = consumer.get("trades", []) if isinstance(consumer.get("trades"), list) else []
        inquiries = consumer.get("inquiries", []) if isinstance(consumer.get("inquiries"), list) else []

        total_accounts = len(trades)
        total_active = 0
        total_settled = 0
        bal_active = 0.0
        bal_revolve = 0.0
        lim_revolve = 0.0
        opened_recent = 0

        for t in trades:
            if not isinstance(t, dict):
                continue
            balance = _safe_float(t.get("balance")) or 0.0
            high_credit = _safe_float(t.get("highCredit")) or 0.0
            credit_limit = _safe_float(t.get("creditLimit")) or 0.0
            bal_active += max(balance, 0.0)

            rate = t.get("rate", {})
            rate_code = str(rate.get("code", "")).strip() if isinstance(rate, dict) else ""
            if rate_code in {"1", "01"}:
                total_active += 1
            if balance <= 0 and high_credit > 0:
                total_settled += 1

            ptype = t.get("portfolioTypeCode", {})
            pcode = str(ptype.get("code", "")).strip().upper() if isinstance(ptype, dict) else ""
            if pcode in {"R", "C", "O"}:
                bal_revolve += max(balance, 0.0)
                lim_revolve += max(credit_limit, 0.0)

            if str(t.get("dateOpened", "")).strip():
                opened_recent += 1

        # Baseline computed rollups from the payload.
        attrs["summary.share.totalaccounts"] = str(total_accounts)
        attrs["summary.share.totalactiveaccs"] = str(total_active)
        attrs["summary.share.totalsettledaccs"] = str(total_settled)
        attrs["summary.share.totalopened6months"] = str(opened_recent)
        attrs["summary.searches.totalsearches12months"] = str(len(inquiries))
        attrs["summary.searches.totalsearches3months"] = str(len(inquiries))
        attrs["summary.indebt.totalbalancesactive"] = str(round(bal_active, 2))
        attrs["summary.indebt.totalbalancesrevolve"] = str(round(bal_revolve, 2))
        attrs["summary.indebt.totallimitsrevolve"] = str(round(lim_revolve, 2))
        if lim_revolve > 0:
            attrs["summary.indebt.balancelimitratiorevolve"] = str(round((bal_revolve / lim_revolve) * 100.0, 2))

        # Keep Equifax attributes and apply explicit dictionary mapping IDs -> canonical keys.
        equifax_attrs: Dict[str, str] = {}
        placeholder_attr_ids: set[str] = set()
        attr_groups = consumer.get("attributes", []) if isinstance(consumer.get("attributes"), list) else []
        for grp in attr_groups:
            if not isinstance(grp, dict):
                continue
            for a in grp.get("attributes", []) if isinstance(grp.get("attributes"), list) else []:
                if not isinstance(a, dict):
                    continue
                identifier = str(a.get("identifier", "")).strip()
                raw_value = a.get("value", "")
                if identifier and _is_placeholder_or_sentinel(raw_value, attr_id=identifier):
                    placeholder_attr_ids.add(identifier)
                value = _normalize_equifax_value(raw_value, attr_id=identifier)
                if identifier and value:
                    equifax_attrs[f"eqx.attr.{identifier}"] = value

        attrs.update(equifax_attrs)
        mapped = apply_equifax_canonical_mapping(equifax_attrs, preserve_existing=attrs)
        guarded_keys: set[str] = set()
        forced_fallback_keys: set[str] = set()
        for canonical_key, candidate_ids in EQUIFAX_CANONICAL_ATTRIBUTE_MAP.items():
            matched_id = None
            for attr_id in candidate_ids:
                candidate = str(attr_id).strip()
                lookup = f"eqx.attr.{candidate}"
                if lookup in equifax_attrs and str(equifax_attrs[lookup]).strip():
                    matched_id = candidate
                    break

            if matched_id is None:
                continue
            if matched_id not in placeholder_attr_ids:
                continue

            computed_value = attrs.get(canonical_key)
            mapped_value = mapped.get(canonical_key)
            if _is_nonzero_numeric(computed_value) and not _is_nonzero_numeric(mapped_value):
                mapped[canonical_key] = str(computed_value)
                guarded_keys.add(canonical_key)

        us_bk_flag = _safe_float(mapped.get("summary.us.bankruptcyflagever"))
        if us_bk_flag is not None:
            mapped["summary.us.bankruptcyflagever"] = "1" if us_bk_flag > 0 else "0"

        # For the US dashboard/runtime, use the actual trade-level revolving credit limits
        # rather than the broader Equifax summary attribute 4628.
        if lim_revolve >= 0:
            mapped["summary.indebt.totallimitsrevolve"] = str(round(lim_revolve, 2))
            forced_fallback_keys.add("summary.indebt.totallimitsrevolve")
        attrs.update(mapped)

        sources: Dict[str, Dict[str, str]] = {}
        for canonical_key, candidate_ids in EQUIFAX_CANONICAL_ATTRIBUTE_MAP.items():
            matched_id = None
            for attr_id in candidate_ids:
                candidate = str(attr_id).strip()
                lookup = f"eqx.attr.{candidate}"
                if lookup in equifax_attrs and str(equifax_attrs[lookup]).strip():
                    matched_id = candidate
                    break

            if canonical_key in forced_fallback_keys or canonical_key in guarded_keys:
                sources[canonical_key] = {
                    "source": "fallback_computed",
                    "equifax_attribute_id": "",
                }
            elif matched_id is not None and canonical_key in attrs:
                sources[canonical_key] = {
                    "source": "equifax_attribute",
                    "equifax_attribute_id": matched_id,
                }
            elif canonical_key in attrs and str(attrs.get(canonical_key, "")).strip():
                sources[canonical_key] = {
                    "source": "fallback_computed",
                    "equifax_attribute_id": "",
                }
            else:
                sources[canonical_key] = {
                    "source": "missing",
                    "equifax_attribute_id": "",
                }

        mapping_diag = {
            "canonical_sources": sources,
            "mapped_keys": [k for k, v in sources.items() if v["source"] == "equifax_attribute"],
            "fallback_keys": [k for k, v in sources.items() if v["source"] == "fallback_computed"],
            "missing_keys": [k for k, v in sources.items() if v["source"] == "missing"],
        }
        return attrs, mapping_diag

    def _structural_counts(self, consumer: Dict[str, Any]) -> Dict[str, int]:
        def _count_list(k: str) -> int:
            v = consumer.get(k)
            return len(v) if isinstance(v, list) else 0

        return {
            "n_elements": 0,
            "n_trade_like": _count_list("trades"),
            "n_enquiry_like": _count_list("inquiries"),
            "n_public_record_like": _count_list("publicRecords"),
            "n_collection_like": _count_list("collections"),
        }
