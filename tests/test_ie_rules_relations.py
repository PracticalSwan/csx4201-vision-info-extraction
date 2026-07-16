from __future__ import annotations

from src.information_extraction.relations import (
    generate_relation_candidates,
    infer_relations,
    relation_features,
)
from src.information_extraction.rules import extract_rule_fields


def _line(index: int, text: str, y: float, confidence: float = 0.95) -> dict:
    bbox = [10.0, y, 300.0, y + 20.0]
    return {
        "id": f"l{index}", "text": text, "confidence": confidence,
        "bbox": bbox,
        "polygon": [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]],
    }


def test_multilingual_rule_fields_include_evidence_and_conflicts() -> None:
    result = {"lines": [
        _line(0, "ACME COMPANY", 5),
        _line(1, "INVOICE # INV-2026-7", 35),
        _line(2, "Date: 15/07/2026", 65),
        _line(3, "Ara Toplam: 100,00 TRY", 95),
        _line(4, "KDV: 7,00 TRY", 125),
        _line(5, "GENEL TOPLAM: 107,00 TRY", 155),
        _line(6, "Email: billing@example.com", 185),
        _line(7, "ชำระโดย: เงินสด", 215),
    ]}
    fields, warnings = extract_rule_fields(result, page_number=2)
    assert fields["invoice_number"]["value"] == "INV-2026-7"
    assert fields["subtotal"]["value"] == "100.00"
    assert fields["tax"]["value"] == "7.00"
    assert fields["total_amount"]["value"] == "107.00"
    assert fields["currency"]["value"] == "TRY"
    assert fields["email"]["value"] == "billing@example.com"
    assert fields["payment_method"]["value"] == "cash"
    assert fields["organization_name"]["value"] == "ACME COMPANY"
    assert all(value["page_number"] == 2 for value in fields.values())
    assert all(value["polygon"] and value["bbox"] for value in fields.values())
    assert all(value["extraction_source"] == "rule" for value in fields.values())
    assert fields["total_amount"]["validation_status"] == "validated"
    assert not any("does not match" in warning for warning in warnings)


def test_extended_commercial_fields_are_label_bound_and_evidence_backed() -> None:
    result = {"lines": [
        _line(0, "Vendor: ACME Bangkok", 5),
        _line(1, "Customer: Ada Lovelace", 35),
        _line(2, "Website: https://example.com", 65),
        _line(3, "Time: 14:35", 95),
        _line(4, "Tax ID: TH-123456789", 125),
        _line(5, "Discount: 5.00 THB", 155),
        _line(6, "Service Charge: 10.00 THB", 185),
        _line(7, "Paid Amount: 105.00 THB", 215),
        _line(8, "Balance: 0.00 THB", 245),
    ]}

    fields, _ = extract_rule_fields(result)

    assert fields["vendor_name"]["value"] == "ACME Bangkok"
    assert fields["customer_name"]["value"] == "Ada Lovelace"
    assert fields["website"]["value"] == "https://example.com"
    assert fields["time"]["value"] == "14:35"
    assert fields["tax_identification_number"]["value"] == "TH-123456789"
    assert fields["discount"]["value"] == "5.00"
    assert fields["service_charge"]["value"] == "10.00"
    assert fields["paid_amount"]["value"] == "105.00"
    assert fields["balance"]["value"] == "0.00"


def _entity(entity_id: str, label: str, bbox: list[float]) -> dict:
    return {"id": entity_id, "label": label, "bbox": bbox, "page_number": 1}


def test_relation_candidates_are_typed_geometric_and_page_local() -> None:
    key = _entity("key", "KEY", [10, 10, 60, 30])
    near = _entity("near", "VALUE", [70, 10, 130, 30])
    far = _entity("far", "VALUE", [900, 900, 980, 930])
    other_page = {**_entity("other-page", "VALUE", [70, 10, 130, 30]), "page_number": 2}
    features = relation_features(key, near)
    assert features["same_line"] == 1.0
    candidates = generate_relation_candidates([key, near, far, other_page], max_normalized_distance=0.5)
    assert [(item["source_id"], item["target_id"], item["relation_type"]) for item in candidates] == [
        ("key", "near", "KEY_VALUE")
    ]
    relations = infer_relations([key, near, far, other_page])
    assert len(relations) == 1
    assert relations[0]["source_id"] == "key"
    assert relations[0]["target_id"] == "near"
    assert relations[0]["confidence"] > 0.8
