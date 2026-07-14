"""Shared command-line helpers for Phase 0 scripts."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "artifacts" / "gate_results.json"


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def update_gate_results(key: str, result: dict[str, object]) -> Path:
    """Atomically merge one gate result into the shared artifact."""

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if ARTIFACT_PATH.exists():
        with ARTIFACT_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = {"phase": "Phase 0"}
    result = dict(result)
    result["status"] = "PASS" if result.get("passed") is True else "FAIL"
    payload[key] = result
    gate_a_passed = payload.get("gate_a", {}).get("passed") is True
    gate_b = payload.get("gate_b", {})
    gate_b_passed = gate_b.get("passed") is True
    reservoir_passed = gate_b.get("reservoir_probe", {}).get("passed") is True
    payload["phase0_status"] = (
        "PASS" if gate_a_passed and gate_b_passed and reservoir_passed else "INCOMPLETE"
    )
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = ARTIFACT_PATH.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    temporary.replace(ARTIFACT_PATH)
    return ARTIFACT_PATH
