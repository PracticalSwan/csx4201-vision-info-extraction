"""Evidence-bearing multilingual rule baseline for canonical fields."""
from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any

from src.information_extraction.geometry import bbox_to_polygon

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().\-]{6,}\d)(?!\w)")
DATE_RE = re.compile(
    r"\b(?:\d{1,4}[./\-]\d{1,2}[./\-]\d{1,4}|\d{1,2}[\-\s](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\-\s]\d{2,4})\b",
    re.IGNORECASE,
)
MONEY_RE = re.compile(
    r"(?:(?:THB|TRY|USD|EUR|GBP|฿|₺|\$|€|£)\s*)?[-+]?\d[\d.,\s]*\d(?:\s*(?:THB|TRY|USD|EUR|GBP|฿|₺|\$|€|£))?",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"[:#]\s*([A-Z0-9][A-Z0-9/_.\-]{2,})", re.IGNORECASE)

KEYWORDS = {
    "invoice_number": ("invoice no", "invoice number", "invoice #", "fatura no", "เลขที่ใบแจ้งหนี้"),
    "receipt_number": ("receipt no", "receipt number", "transaction no", "transaction number", "fiş no", "เลขที่ใบเสร็จ"),
    "reference_number": ("reference no", "reference number", "account no", "account number", "referans no", "เลขอ้างอิง"),
    "subtotal": ("subtotal", "sub total", "ara toplam", "ยอดรวมก่อนภาษี"),
    "tax": ("tax", "vat", "kdv", "ภาษี", "ภาษีมูลค่าเพิ่ม"),
    "total_amount": ("grand total", "total amount", "amount due", "total", "genel toplam", "toplam", "ยอดรวม", "รวมทั้งสิ้น"),
    "payment_method": ("payment method", "paid by", "cash", "credit card", "nakit", "kredi kartı", "ชำระโดย", "เงินสด", "บัตรเครดิต"),
    "address": ("address", "adres", "ที่อยู่"),
}
CURRENCY_MAP = {
    "฿": "THB", "thb": "THB", "บาท": "THB",
    "₺": "TRY", "try": "TRY", "tl": "TRY",
    "$": "USD", "usd": "USD", "€": "EUR", "eur": "EUR", "£": "GBP", "gbp": "GBP",
}


def extract_rule_fields(
    ocr_result: Mapping[str, Any], *, page_number: int = 1
) -> tuple[dict[str, Any], list[str]]:
    """Extract only supported fields and return explicit conflict warnings."""
    lines = _line_records(ocr_result)
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in lines:
        text = line["text"].strip()
        folded = text.casefold()
        confidence = line["confidence"]
        email = EMAIL_RE.search(text)
        if email:
            candidates["email"].append(_evidence(email.group(0), email.group(0), line, confidence, "rule:email", page_number))
        phone = PHONE_RE.search(text)
        if phone:
            normalized_phone = re.sub(r"[^+\d]", "", phone.group(0))
            candidates["phone_number"].append(_evidence(normalized_phone, phone.group(0), line, confidence, "rule:phone", page_number))
        date = DATE_RE.search(text)
        if date:
            candidates["date"].append(_evidence(date.group(0), date.group(0), line, confidence, "rule:date", page_number))
        for field in ("invoice_number", "receipt_number", "reference_number"):
            if _has_keyword(folded, KEYWORDS[field]):
                number = NUMBER_RE.search(text)
                if number:
                    candidates[field].append(_evidence(number.group(1), number.group(0), line, confidence, f"rule:{field}", page_number))
        for field in ("subtotal", "tax", "total_amount"):
            if _has_keyword(folded, KEYWORDS[field]):
                if field == "total_amount" and _has_keyword(folded, KEYWORDS["subtotal"]):
                    continue
                money_matches = list(MONEY_RE.finditer(text))
                if money_matches:
                    raw = money_matches[-1].group(0).strip()
                    normalized = _normalize_money(raw)
                    if normalized is not None:
                        keyword_bonus = 0.06 if field != "total_amount" or "grand" in folded or "due" in folded else 0.0
                        candidates[field].append(_evidence(normalized, raw, line, min(1.0, confidence + keyword_bonus), f"rule:{field}", page_number))
        if _has_keyword(folded, KEYWORDS["payment_method"]):
            value = _payment_method(text)
            if value:
                candidates["payment_method"].append(_evidence(value, text, line, confidence, "rule:payment_method", page_number))
        if _has_keyword(folded, KEYWORDS["address"]):
            value = _after_label(text)
            if value:
                candidates["address"].append(_evidence(value, text, line, confidence, "rule:address", page_number))
        currency = _currency(text)
        if currency:
            candidates["currency"].append(_evidence(currency, text, line, confidence, "rule:currency", page_number))
        title = _document_title(text)
        if title:
            candidates["document_title"].append(_evidence(title, text, line, confidence, "rule:document_title", page_number))

    organization = _organization_candidate(lines, page_number)
    if organization:
        candidates["organization_name"].append(organization)
    fields: dict[str, Any] = {}
    warnings: list[str] = []
    for field, values in candidates.items():
        values.sort(key=lambda item: (float(item["confidence"] or 0.0), item["page_number"]), reverse=True)
        fields[field] = values[0]
        distinct = {str(item["value"]) for item in values}
        if len(distinct) > 1:
            warnings.append(f"multiple {field} candidates; selected highest-confidence evidence")
    _validate_total_consistency(fields, warnings)
    return fields, warnings


def _line_records(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = list(result.get("lines") or []) or list(result.get("words") or [])
    records = []
    for item in source:
        text = str(item.get("text", "")).strip()
        bbox = item.get("bbox")
        if not text or not isinstance(bbox, Sequence) or len(bbox) != 4:
            continue
        confidence = item.get("confidence")
        records.append({
            "text": text,
            "bbox": [float(value) for value in bbox],
            "polygon": item.get("polygon") or bbox_to_polygon(bbox),
            "confidence": max(0.0, min(1.0, float(confidence))) if confidence is not None else 0.5,
        })
    return records


def _evidence(
    value: Any, raw_text: str, line: Mapping[str, Any], confidence: float,
    method: str, page_number: int,
) -> dict[str, Any]:
    return {
        "value": value,
        "raw_text": raw_text,
        "polygon": [[float(x), float(y)] for x, y in line["polygon"]],
        "bbox": [float(value) for value in line["bbox"]],
        "confidence": max(0.0, min(1.0, float(confidence))),
        "method": method,
        "page_number": int(page_number),
    }


def _has_keyword(text: str, keywords: Sequence[str]) -> bool:
    return any(keyword.casefold() in text for keyword in keywords)


def _after_label(text: str) -> str:
    parts = re.split(r"\s*[:：]\s*", text, maxsplit=1)
    return parts[1].strip() if len(parts) == 2 else ""


def _normalize_money(value: str) -> str | None:
    cleaned = re.sub(r"(?:THB|TRY|USD|EUR|GBP|฿|₺|\$|€|£|\s)", "", value, flags=re.IGNORECASE)
    if cleaned.count(",") and cleaned.count("."):
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif cleaned.count(",") == 1 and len(cleaned.rsplit(",", 1)[1]) in {2, 3}:
        cleaned = cleaned.replace(",", ".")
    else:
        cleaned = cleaned.replace(",", "")
    try:
        amount = Decimal(cleaned)
    except InvalidOperation:
        return None
    return format(amount.quantize(Decimal("0.01")), "f")


def _currency(text: str) -> str | None:
    folded = text.casefold()
    for marker, code in CURRENCY_MAP.items():
        if marker in folded if marker.isalpha() else marker in text:
            return code
    return None


def _payment_method(text: str) -> str | None:
    folded = text.casefold()
    for words, value in (
        (("cash", "nakit", "เงินสด"), "cash"),
        (("credit card", "kredi kartı", "บัตรเครดิต"), "credit_card"),
        (("bank transfer", "havale", "โอนเงิน"), "bank_transfer"),
    ):
        if any(word in folded for word in words):
            return value
    return None


def _document_title(text: str) -> str | None:
    folded = text.casefold()
    if any(word in folded for word in ("invoice", "fatura", "ใบแจ้งหนี้")):
        return "invoice"
    if any(word in folded for word in ("receipt", "fiş", "fis", "ใบเสร็จ")):
        return "receipt"
    return None


def _organization_candidate(
    lines: list[Mapping[str, Any]], page_number: int
) -> dict[str, Any] | None:
    if not lines:
        return None
    maximum_y = max(float(line["bbox"][3]) for line in lines)
    for line in sorted(lines, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        text = str(line["text"])
        if line["bbox"][1] > maximum_y * 0.25:
            break
        if len(text) >= 3 and any(character.isalpha() for character in text) and not re.search(r"invoice|receipt|date|total|tax|fatura|ใบเสร็จ", text, re.I):
            uppercase_ratio = sum(character.isupper() for character in text) / max(1, sum(character.isalpha() for character in text))
            if uppercase_ratio >= 0.5:
                return _evidence(
                    text.strip(), text, line, float(line["confidence"]) * 0.75,
                    "rule:top_header_organization", page_number,
                )
    return None


def _validate_total_consistency(fields: Mapping[str, Any], warnings: list[str]) -> None:
    try:
        subtotal = Decimal(str(fields["subtotal"]["value"])) if fields.get("subtotal") else None
        tax = Decimal(str(fields["tax"]["value"])) if fields.get("tax") else None
        total = Decimal(str(fields["total_amount"]["value"])) if fields.get("total_amount") else None
    except (InvalidOperation, KeyError):
        return
    if subtotal is not None and total is not None and total < subtotal:
        warnings.append("selected total is lower than subtotal")
    if subtotal is not None and tax is not None and total is not None:
        if abs((subtotal + tax) - total) > max(Decimal("0.05"), abs(total) * Decimal("0.02")):
            warnings.append("subtotal plus tax does not match total within tolerance")
