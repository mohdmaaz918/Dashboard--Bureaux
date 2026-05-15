from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from lxml import etree

def _xp_text(root, xpath: str) -> str | None:
    """
    Return stripped text for a single xpath, or None if missing/blank.
    """
    try:
        nodes = root.xpath(xpath)
        if not nodes:
            return None
        # node may be element or string depending on xpath
        if hasattr(nodes[0], "text"):
            val = nodes[0].text
        else:
            val = str(nodes[0])
        if val is None:
            return None
        val = str(val).strip()
        return val if val != "" else None
    except Exception:
        return None


def _xp_attr(root, xpath: str, attr_name: str) -> str | None:
    """
    Return an attribute value for the first matched node, or None.
    """
    try:
        nodes = root.xpath(xpath)
        if not nodes:
            return None
        val = nodes[0].get(attr_name)
        if val is None:
            return None
        val = str(val).strip()
        return val if val != "" else None
    except Exception:
        return None


@dataclass(frozen=True)
class ParseResult:
    """Normalised representation of a bureau XML file.

    - flat: flattened leaf-text fields keyed by an XML-ish path (for debug / discovery)
    - attrs: attribute-code map (best-effort) -> str value
    - counts: structural counts (trades/enquiries/public records/etc.)
    - meta: metadata like filename, namespace presence, parse warnings
    """
    flat: Dict[str, str]
    attrs: Dict[str, str]
    counts: Dict[str, int]
    meta: Dict[str, Any]


class BureauAdapter:
    bureau_name: str = "unknown"

    def parse_file(self, xml_path: Path, strict: bool = True) -> ParseResult:
        raise NotImplementedError


class TransUnionAdapter(BureauAdapter):
    bureau_name = "transunion"

    # Heuristics: common field labels seen in bureau XML "attribute" structures
    _CODE_KEYS = {"code", "attribute_code", "attributecode", "id", "attributeid", "number", "key", "name"}
    _VALUE_KEYS = {"value", "val", "amount", "score", "result", "text", "data"}

    def parse_file(self, xml_path: Path, strict: bool = True) -> ParseResult:
        xml_path = Path(xml_path)

        parser = etree.XMLParser(
            recover=not strict,
            resolve_entities=False,
            no_network=True,
            remove_blank_text=True,
            huge_tree=True,
        )

        raw = None
        try:
            raw = xml_path.read_bytes()
            root = etree.fromstring(raw, parser)
        except Exception as exc:
            if strict:
                raise

            # Retry: remove/ignore invalid bytes and try again
            try:
                if raw is None:
                    raw = xml_path.read_bytes()
                fixed = raw.decode("utf-8", errors="ignore").encode("utf-8")
                root = etree.fromstring(fixed, parser)
            except Exception as exc2:
                return ParseResult(
                    flat={},
                    attrs={},
                    counts={},
                    meta={"file": xml_path.name, "error": str(exc2)},
                )

        nsmap = {k if k is not None else "ns": v for k, v in (root.nsmap or {}).items()}
        has_ns = bool(root.nsmap)

        # 1) Flatten leaf texts (discovery/debug)
        flat: Dict[str, str] = {}
        self._flatten_leaf_texts(root, flat, max_items=25_000)

        # 2) Extract numeric/boolean-like report metrics (summary/score blocks)
        attrs: Dict[str, str] = {}

        # Extract summary + credit score metrics from TU UK SOAP XML
        self._extract_report_metrics(root, attrs)

        # 2b) Extract generic Code/Value (or code="" value="") attribute pairs.
        # Many bureau feeds expose engineered counts as short concept codes (e.g., DB, NQB).
        # We merge them into attrs for discovery/scoring.
        code_pairs: Dict[str, str] = {}
        self._extract_code_value_pairs(root, code_pairs)
        for k, v in code_pairs.items():
            if k not in attrs:
                attrs[k] = v
            elif attrs[k] != v:
                attrs[f"codepair.{k}"] = v

        # 2c) BSB-style attribute codes (attribute NAME is the code, e.g. EB="23")
        bsb_pairs: Dict[str, str] = {}
        self._extract_bsb_attr_codes(root, bsb_pairs)
        for k, v in bsb_pairs.items():
            if k not in attrs:
                attrs[k] = v
            elif attrs[k] != v:
                attrs[f"bsb.{k}"] = v

        # 2c) Extract short-code attributes like DB="0", FB="10", LQB="0" (BSB-style).
        shortcode_attrs: Dict[str, str] = {}
        self._extract_shortcode_attributes(root, shortcode_attrs)
        for k, v in shortcode_attrs.items():
            if k not in attrs:
                attrs[k] = v
            elif attrs[k] != v:
                attrs[f"shortcode.{k}"] = v

        # 3) Structural counts (namespaced-safe)
        counts = self._structural_counts(root)

        meta: Dict[str, Any] = {
            "file": xml_path.name,
            "bureau": self.bureau_name,
            "has_namespace": has_ns,
            "flat_items": len(flat),
            "attr_items": len(attrs),
        }

        return ParseResult(flat=flat, attrs=attrs, counts=counts, meta=meta)

    def _extract_bsb_attr_codes(self, root: etree._Element, out: Dict[str, str]) -> None:
        """
        BSB-style TU XML stores engineered metrics as XML attributes where the
        attribute NAME is the code (e.g. EB="23", FB="10", JF="542").
        Extract CODE->value for attributes that look like bureau codes.
        """

        def local(tag: str) -> str:
            return tag.split("}", 1)[1] if "}" in tag else tag

        # Restrict to uppercase code-like attributes to avoid pulling in noise like app_no="1"
        def looks_like_code(k: str) -> bool:
            k = (k or "").strip()
            return 2 <= len(k) <= 4 and k.isupper() and k.isalpha()

        # Prefer to start under the <BSB> node if present
        bsb_root = None
        for el in root.iter():
            if isinstance(el.tag, str) and local(el.tag).upper() == "BSB":
                bsb_root = el
                break

        scan_root = bsb_root if bsb_root is not None else root

        for el in scan_root.iter():
            if not isinstance(el.tag, str):
                continue
            for k, v in (el.attrib or {}).items():
                if looks_like_code(k):
                    vv = (v or "").strip()
                    if vv == "":
                        continue
                    out[self._norm_code(k)] = vv

    def _flatten_leaf_texts(self, root: etree._Element, out: Dict[str, str], max_items: int = 50_000) -> None:
        """Store leaf node texts keyed by a simple path: Tag/ChildTag[2]/LeafTag"""
        def tag_name(el: etree._Element) -> str:
            # Strip namespace
            if el.tag is None:
                return "None"
            if "}" in el.tag:
                return el.tag.split("}", 1)[1]
            return str(el.tag)

        def iter_children(el: etree._Element):
            # children elements only (skip comments/PI)
            for c in el:
                if isinstance(c.tag, str):
                    yield c

        # Build parent->counts to index siblings of same tag
        sibling_index: Dict[Tuple[int, str], int] = {}

        stack: list[Tuple[etree._Element, str]] = [(root, tag_name(root))]
        while stack and len(out) < max_items:
            el, path = stack.pop()

            # NEW: capture element attributes as path.@ATTR -> value
            if el.attrib:
                for attr_name, attr_val in el.attrib.items():
                    if attr_val is None:
                        continue
                    s = str(attr_val).strip()
                    if not s:
                        continue
                    if len(out) >= max_items:
                        break
                    out[f"{path}.@{attr_name}"] = s

            children = list(iter_children(el))
            text = (el.text or "").strip()

            if not children and text:
                out[path] = text
                continue

            # push children
            for child in reversed(children):
                t = tag_name(child)
                key = (id(el), t)
                sibling_index[key] = sibling_index.get(key, 0) + 1
                idx = sibling_index[key]
                child_path = f"{path}/{t}[{idx}]"
                stack.append((child, child_path))

    def _extract_report_metrics(self, root: etree._Element, out: Dict[str, str]) -> None:
        """
        Extract summary + credit score metrics from Callcredit/TransUnion UK SOAP XML.

        Produces keys like:
          summary.searches.totalsearches3months
          summary.indebt.totalbalancesactive
          creditscore.class_10.score
        """

        def local(tag: str) -> str:
            return tag.split("}", 1)[1] if "}" in tag else tag

        def _summary_relative_path(el: etree._Element, anchor: etree._Element) -> str:
            """Build a dotted path relative to the anchor (summary) element."""
            parts = []
            cur = el
            while cur is not None and cur is not anchor:
                if isinstance(cur.tag, str):
                    parts.append(local(cur.tag).lower())
                cur = cur.getparent()
            parts.reverse()
            return ".".join(parts)

        # Locate creditreport node (root may already be <creditreport>)
        creditreport = root if (isinstance(root.tag, str) and local(root.tag).lower() == "creditreport") else None
        if creditreport is None:
            for el in root.iter():
                if isinstance(el.tag, str) and local(el.tag).lower() == "creditreport":
                    creditreport = el
                    break

        if creditreport is None:
            return

        # --- Summary block ---
        summary = None
        for el in creditreport.iter():
            if not isinstance(el.tag, str):
                continue
            if local(el.tag).lower() == "summary":
                summary = el
                break

        if summary is not None:
            for el in summary.iter():
                if not isinstance(el.tag, str):
                    continue

                rel = _summary_relative_path(el, summary)
                base_key = f"summary.{rel}" if rel else "summary"

                # 1) Capture attributes on any summary descendant
                if el.attrib:
                    for ak, av in el.attrib.items():
                        if av is None:
                            continue
                        avs = str(av).strip()
                        if not avs or avs.lower() in {"{nd}", "na", "n/a", "null", "none", "unknown"}:
                            continue
                        out[f"{base_key}.{ak.lower()}"] = avs

                # 2) Capture leaf text values
                children = [c for c in el if isinstance(c.tag, str)]
                if children:
                    continue

                txt = (el.text or "").strip()
                if not txt or txt.lower() in {"{nd}"}:
                    continue

                out[base_key] = txt

        # --- Credit scores ---
        for cs in creditreport.iter():
            if not isinstance(cs.tag, str):
                continue
            if local(cs.tag).lower() != "creditscore":
                continue

            score_el = None
            for child in cs:
                if isinstance(child.tag, str) and local(child.tag).lower() == "score":
                    score_el = child
                    break

            score_class = None
            if score_el is not None:
                score_class = (score_el.get("class") or "").strip() or None
                score_val = (score_el.text or "").strip()
                if score_val and score_class:
                    out[f"creditscore.class_{score_class}.score"] = score_val
                elif score_val:
                    out["creditscore.score"] = score_val

            # Reason codes
            for r in cs.iter():
                if not isinstance(r.tag, str):
                    continue
                if local(r.tag).lower() != "code":
                    continue
                code_txt = (r.text or "").strip()
                if not code_txt or code_txt == "0":
                    continue
                if score_class:
                    out[f"creditscore.class_{score_class}.reason_{code_txt}"] = "1"
                else:
                    out[f"creditscore.reason_{code_txt}"] = "1"

        # --- InternalScore (company pre-purchase score) ---
        for el in root.iter():
            if not isinstance(el.tag, str):
                continue
            if local(el.tag).lower() == "internalscore":
                val = (el.text or "").strip()
                if val:
                    out["internalscore"] = val
                break


    def _extract_code_value_pairs(self, root: etree._Element, out: Dict[str, str]) -> None:
        """Find repeated structures that look like:
        <Something>
            <Code>XYZ</Code>
            <Value>123</Value>
        </Something>

        or attributes stored as:
        <Attribute code="XYZ" value="123" />
        """
        def local(tag: str) -> str:
            return tag.split("}", 1)[1].lower() if "}" in tag else tag.lower()

        def text_or_none(el: Optional[etree._Element]) -> Optional[str]:
            if el is None:
                return None
            t = (el.text or "").strip()
            return t if t else None

        # (A) Attribute-like elements with code/value attributes
        for el in root.iter():
            if not isinstance(el.tag, str):
                continue

            # Look for attribute names on element
            attrib = {k.lower(): v for k, v in (el.attrib or {}).items()}
            code = None
            value = None

            for ck in self._CODE_KEYS:
                if ck in attrib and attrib[ck]:
                    code = attrib[ck].strip()
                    break
            for vk in self._VALUE_KEYS:
                if vk in attrib and attrib[vk]:
                    value = attrib[vk].strip()
                    break

            if code and value:
                # normalise feature key
                out[self._norm_code(code)] = value

        # (B) Elements with child <Code>/<Value> patterns
        # For each element, check if it has a child that looks like code and value.
        for el in root.iter():
            if not isinstance(el.tag, str):
                continue

            children = [c for c in el if isinstance(c.tag, str)]
            if len(children) < 2:
                continue

            child_map: Dict[str, etree._Element] = {}
            for c in children:
                child_map[local(c.tag)] = c

            code_el = None
            value_el = None

            # Find any child whose local name matches code-ish keys
            for ck in self._CODE_KEYS:
                if ck in child_map:
                    code_el = child_map[ck]
                    break

            for vk in self._VALUE_KEYS:
                if vk in child_map:
                    value_el = child_map[vk]
                    break

            code = text_or_none(code_el)
            value = text_or_none(value_el)

            if code and value:
                out[self._norm_code(code)] = value

    def _extract_shortcode_attributes(self, root: etree._Element, out: Dict[str, str]) -> None:
        """Extract attributes where the attribute NAME is a short concept code, e.g. DB="0", FB="10", LQB="0".

        Heuristic:
        - attribute name length 2..4
        - attribute name consists of A-Z and digits only
        - value is non-empty
        - if the same code appears multiple times, keep the most informative value:
            prefer non-{ND} over {ND}, and prefer non-zero over zero.
        """
        def is_nd(x: str) -> bool:
            return x == "{ND}"

        def is_zero(x: str) -> bool:
            return x == "0"

        for el in root.iter():
            if not isinstance(el.tag, str):
                continue
            if not el.attrib:
                continue

            for k, v in el.attrib.items():
                if v is None:
                    continue

                code = str(k).strip()
                if not (2 <= len(code) <= 4):
                    continue
                if not all(ch.isdigit() or ("A" <= ch <= "Z") for ch in code):
                    continue

                sval = str(v).strip()
                if not sval:
                    continue

                if code not in out:
                    out[code] = sval
                else:
                    existing = (out.get(code) or "").strip()

                    # Replace if existing is empty/{ND} and new is meaningful
                    if (not existing or is_nd(existing)) and (sval and not is_nd(sval)):
                        out[code] = sval
                    # Replace if existing is 0 and new is non-zero (and not {ND})
                    elif is_zero(existing) and (sval and not is_zero(sval) and not is_nd(sval)):
                        out[code] = sval


    def _structural_counts(self, root: etree._Element) -> Dict[str, int]:
        """Counts based on tag-name heuristics (namespace-safe)."""
        def local(tag: str) -> str:
            return tag.split("}", 1)[1].lower() if "}" in tag else tag.lower()

        trade_like = {"trade", "account", "tradeline", "creditaccount", "loan", "instalment", "installment"}
        enquiry_like = {"enquiry", "inquiry", "enquiries", "inquiries"}
        public_record_like = {"publicrecord", "judgment", "bankruptcy", "ccj"}
        collection_like = {"collection", "collections"}

        counts = {
            "n_elements": 0,
            "n_trade_like": 0,
            "n_enquiry_like": 0,
            "n_public_record_like": 0,
            "n_collection_like": 0,
        }

        for el in root.iter():
            if not isinstance(el.tag, str):
                continue
            counts["n_elements"] += 1
            t = local(el.tag)

            if t in trade_like:
                counts["n_trade_like"] += 1
            if t in enquiry_like:
                counts["n_enquiry_like"] += 1
            if t in public_record_like:
                counts["n_public_record_like"] += 1
            if t in collection_like:
                counts["n_collection_like"] += 1

        return counts

    @staticmethod
    def _norm_code(code: str) -> str:
        c = code.strip().replace(" ", "_")
        return c


# ---------------------------------------------------------------------------
# Standalone account extraction (for UW tradeline export)
# ---------------------------------------------------------------------------

SUPPLIER_TYPE_LABELS = {
    "AF": "Agricultural Finance", "BF": "Bank Finance", "BK": "Bank",
    "BS": "Building Society", "CA": "Callcredit", "CB": "Credit Broker",
    "CC": "Credit Card Company", "CD": "CC Consumer Data Capture",
    "CU": "Credit Union", "DC": "Debt Collection Agency", "ED": "EuroDirect",
    "FC": "Factoring", "FH": "Finance House", "FN": "Finance House",
    "GO": "Government", "HC": "Home Credit", "HS": "Home Shopping",
    "IN": "Insurance Company", "IT": "Internet/Online Services",
    "LS": "Leasing Company", "MC": "Mortgage Company",
    "MF": "Motor Vehicle Finance", "MO": "Mail Order",
    "RC": "Rental Company", "RT": "Retailer", "SC": "Securities",
    "SL": "Student Loan", "ST": "Stockbroker",
    "TC": "Telecommunications", "UT": "Utility",
    "VR": "Vehicle Rental", "XX": "Unknown",
}

ACCOUNT_TYPE_LABELS = {
    # Loan and Instalment Credit
    "LN": "Loan (unspecified)", "UL": "Unsecured Loan", "SE": "Secured Loan",
    "BA": "Balloon Repayment Loan", "FD": "Fixed Term Deferred Payment",
    "SL": "Student Loan", "CP": "Personal Contract Purchase",
    "RG": "Repayment Grant", "TL": "Term Loan",
    "ML": "Motor Finance Loan", "CS": "Unsecured Car Loan",
    "SC": "Secured Car Loan", "BL": "Unsecured Business Loan",
    "SB": "Secured Business Loan", "LP": "Life Policy Loan",
    "ZL": "0% Finance Loan", "BR": "Brewery Loan", "PL": "Petroleum Loan",
    "BN": "Buy Now Pay Later", "EL": "Employer Loan",
    "FS": "Further Secured Loan", "SO": "Set Off Loan",
    "IL": "Advance Against Income", "HP": "Hire Purchase",
    "BH": "Balloon HP", "DH": "Deferred HP", "ZH": "0% HP",
    "CX": "Credit Sale", "CY": "Conditional Sale",
    "RT": "Rental Agreement", "CR": "Consumer Goods Rental",
    "PR": "Property Rental", "LS": "Lease", "FL": "Finance Lease",
    "OL": "Operating Lease", "ED": "Education", "OR": "Other Rental",
    "DA": "Consolidated Debt",
    # Mortgage
    "MG": "Mortgage (unspecified)", "RM": "Residential Mortgage",
    "FM": "Flexible Mortgage", "CM": "Commercial Mortgage",
    "BM": "Residential Buy To Let", "XM": "First Time Buyer",
    "FO": "First Mortgage With Investment Offset",
    "IM": "Second Mortgage With Investment Offset",
    "SM": "Residential Second Mortgage",
    "MM": "Semi-commercial First Mortgage",
    "NM": "Semi-commercial Second Mortgage",
    "DM": "Shared Ownership Mortgage", "MT": "Mortgage + Unsecured Loan",
    "ZM": "100% LTV Mortgage", "OM": "Investment Offset Mortgage",
    # Revolving Credit & Budget
    "CC": "Credit Card", "CO": "Company Credit Card",
    "ST": "Store Card", "RS": "Retail Store Card",
    "FC": "Fuel Card", "CH": "Charge Card", "BD": "Budget Account",
    # Telecommunications
    "TM": "Telecoms", "AM": "Airtime Monthly Contract",
    "AU": "Airtime Upfront", "QA": "Airtime Quarterly",
    "LT": "Landline Even Payment", "QT": "Landline Quarterly",
    "CB": "Cable", "SI": "Satellite", "BU": "Business Line",
    "BO": "Broadband", "MU": "Multi Communications",
    # Utilities
    "UT": "Utility", "UE": "Utility Even Payment",
    "QU": "Utility Quarterly", "GE": "Gas Even Payment",
    "QG": "Gas Quarterly", "EE": "Electricity Even Payment",
    "QE": "Electricity Quarterly", "EW": "Water Even Payment",
    "QW": "Water Quarterly", "OI": "Oil", "DU": "Dual Fuel",
    # Home Shopping
    "HS": "Home Shopping", "MO": "Mail Order", "MA": "Mail Order Agency",
    "TV": "TV Shopping", "HD": "Home Shopping Direct",
    "HX": "Home Shopping Cash", "HA": "Home Shopping Agency",
    "WI": "Weekly Instalment Plan", "BC": "Book/Music/Video Club",
    # Bank
    "CA": "Current Account", "OD": "Overdraft", "DC": "Debit Card",
    "SX": "Set Off Current Account", "RQ": "Returned Cheque",
    "BK": "Bank", "DF": "Bank Default",
    # Miscellaneous
    "MC": "Multifunctional Card", "OA": "Option Account",
    "SA": "Stock Broking Account", "EC": "E-Cash",
    "CT": "Council Tax", "LR": "Local Rates",
    "ZC": "Standby Credit", "CZ": "Combined Credit Accounts",
    "AF": "Agricultural Finance", "AD": "Asset Discounting",
    "BX": "Builders Merchant Credit", "CD": "Crown Debt",
    "FT": "Factoring", "GL": "Guarantee Liability",
    "SS": "Stationery Supplier", "RC": "Revolving Credit",
    "DP": "Deferred Payment", "TR": "Trade Credit",
    "VS": "Variable Subscription", "IC": "Internet Credit Line",
    "IS": "Internet Shopping", "GA": "Gambling",
    # Insurance
    "IN": "Insurance", "GI": "General Insurance",
    "BI": "Buildings Insurance", "CI": "Contents Insurance",
    "HI": "Household Insurance", "MI": "Motor Insurance",
    "PI": "Personal Health Insurance", "PT": "Card Protection",
    "MP": "Mortgage Protection", "PP": "Payment Protection",
    # Home Credit
    "HC": "Home Credit",
}

ACCOUNT_STATUS_LABELS = {
    "OK": "OK", "ST": "Settled", "QY": "Query", "AA": "Arrangement",
    "UC": "Unclassified", "DF": "Default", "DM": "Default (DM)",
    "VD": "Voluntary Default", "DR": "Dormant", "IA": "Inactive",
}

PAYMENT_STATUS_LABELS = {
    "0": "Up to date", "1": "1 behind", "2": "2 behind", "3": "3 behind",
    "4": "4 behind", "5": "5 behind", "6": "6+ behind",
    "U": "Unclassified", "D": "Default",
}

PAY_STATUS_SEVERITY = {"0": 0, "U": 1, "1": 2, "2": 3, "3": 4, "4": 5, "5": 6, "6": 7, "D": 8}
ACC_STATUS_SEVERITY = {"OK": 0, "UC": 1, "ST": 2, "QY": 3, "AA": 4, "DR": 5, "IA": 5, "DF": 6, "DM": 7, "VD": 7}


def extract_accounts_from_xml(xml_bytes: bytes) -> Tuple[list, list]:
    """Parse XML bytes and extract all <acc> tradeline elements.

    Returns (accounts, history) where:
      accounts: list of dicts (one per account) with summary fields + payment pattern
      history:  list of dicts (one per month per account), sorted by acc then month desc
    """
    def _local(tag):
        if not isinstance(tag, str):
            return ""
        return tag.split("}", 1)[1] if "}" in tag else tag

    def _child_text(parent, child_name):
        for c in parent:
            if isinstance(c.tag, str) and _local(c.tag).lower() == child_name.lower():
                return (c.text or "").strip()
        return ""

    def _child_el(parent, child_name):
        for c in parent:
            if isinstance(c.tag, str) and _local(c.tag).lower() == child_name.lower():
                return c
        return None

    parser = etree.XMLParser(recover=True, resolve_entities=False, no_network=True,
                             remove_blank_text=True, huge_tree=True)
    try:
        root = etree.fromstring(xml_bytes, parser)
    except Exception:
        return [], []

    accounts = []
    history = []
    acc_idx = 0

    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        if _local(el.tag).lower() != "acc":
            continue

        acc_idx += 1

        supplier_el = _child_el(el, "supplierdetails")
        supplier_type = _child_text(supplier_el, "suppliertypecode") if supplier_el is not None else ""

        holder_el = _child_el(el, "accholderdetails")
        holder_name = _child_text(holder_el, "name") if holder_el is not None else ""
        holder_dob = _child_text(holder_el, "dob") if holder_el is not None else ""
        holder_status = _child_text(holder_el, "statuscode") if holder_el is not None else ""
        holder_start = _child_text(holder_el, "startdate") if holder_el is not None else ""
        holder_end = _child_text(holder_el, "enddate") if holder_el is not None else ""
        holder_addr = ""
        if holder_el is not None:
            addr_el = _child_el(holder_el, "address")
            if addr_el is not None:
                holder_addr = (addr_el.text or "").strip()

        details_el = _child_el(el, "accdetails")
        det = {}
        if details_el is not None:
            for field in ("joint", "status", "dateupdated", "acctypecode", "accgroupid",
                          "currencycode", "balance", "limit", "openbalance",
                          "paystartdate", "accstartdate", "accenddate",
                          "regpayment", "repayperiod", "repayfreqcode", "lumppayment"):
                det[field] = _child_text(details_el, field)

        acc_type = det.get("acctypecode", "")
        acc_label = f"#{acc_idx} {supplier_type}/{acc_type}"

        # --- Build per-account history and derive summary fields ---
        acc_history_rows = []
        hist_el = _child_el(el, "acchistory")
        if hist_el is not None:
            for ah in hist_el:
                if not isinstance(ah.tag, str):
                    continue
                if _local(ah.tag).lower() != "ah":
                    continue
                row = {
                    "Acc #": acc_idx,
                    "Acc Label": acc_label,
                    "Month": ah.get("m", ""),
                    "Balance": ah.get("bal", ""),
                    "Limit": ah.get("limit", ""),
                    "Acc Status": ah.get("acc", ""),
                    "Acc Status Desc": ACCOUNT_STATUS_LABELS.get(ah.get("acc", ""), ah.get("acc", "")),
                    "Pay Status": ah.get("pay", ""),
                    "Pay Status Desc": PAYMENT_STATUS_LABELS.get(ah.get("pay", ""), ah.get("pay", "")),
                }
                acc_history_rows.append(row)

        # Sort this account's history chronologically for the payment pattern
        sorted_by_month = sorted(acc_history_rows, key=lambda r: r["Month"])
        months_of_data = len(sorted_by_month)

        pay_codes_chrono = [r["Pay Status"] for r in sorted_by_month]
        acc_codes_chrono = [r["Acc Status"] for r in sorted_by_month]

        # Build compact payment pattern string (chronological, newest last)
        pay_pattern = " ".join(pay_codes_chrono[-24:]) if pay_codes_chrono else ""
        acc_pattern = " ".join(acc_codes_chrono[-24:]) if acc_codes_chrono else ""

        worst_pay = ""
        if pay_codes_chrono:
            worst_pay = max(pay_codes_chrono, key=lambda c: PAY_STATUS_SEVERITY.get(c, -1))
        worst_acc = ""
        if acc_codes_chrono:
            worst_acc = max(acc_codes_chrono, key=lambda c: ACC_STATUS_SEVERITY.get(c, -1))

        accounts.append({
            "Acc #": acc_idx,
            "Supplier": supplier_type,
            "Supplier Desc": SUPPLIER_TYPE_LABELS.get(supplier_type, supplier_type),
            "Account Type": acc_type,
            "Account Type Desc": ACCOUNT_TYPE_LABELS.get(acc_type, acc_type),
            "Account Holder": holder_name,
            "DOB": holder_dob,
            "Address": holder_addr,
            "Status": det.get("status", holder_status),
            "Joint": "Yes" if det.get("joint") == "1" else "No",
            "Start Date": det.get("accstartdate", holder_start),
            "End Date": det.get("accenddate", holder_end),
            "Balance": det.get("balance", ""),
            "Open Balance": det.get("openbalance", ""),
            "Limit": det.get("limit", ""),
            "Regular Payment": det.get("regpayment", ""),
            "Repay Period": det.get("repayperiod", ""),
            "Repay Freq": det.get("repayfreqcode", ""),
            "Currency": det.get("currencycode", ""),
            "Last Updated": det.get("dateupdated", ""),
            "Months of Data": months_of_data,
            "Payment Pattern": pay_pattern,
            "Status Pattern": acc_pattern,
            "Worst Pay Status": worst_pay,
            "Worst Pay Desc": PAYMENT_STATUS_LABELS.get(worst_pay, worst_pay),
            "Worst Acc Status": worst_acc,
            "Worst Acc Desc": ACCOUNT_STATUS_LABELS.get(worst_acc, worst_acc),
        })

        # Add to global history (sorted most recent first per account for the sheet)
        for row in sorted(acc_history_rows, key=lambda r: r["Month"], reverse=True):
            history.append(row)

    return accounts, history
