from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness import read_json, validate_and_clean
from provider import LlamaCppProvider


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--config", default="harness_config.json")
    args = ap.parse_args()

    cfg = read_json(Path(args.config))
    schema = read_json(Path("vision_response.schema.json"))

    provider = LlamaCppProvider(cfg["provider"], schema)

    print("[smoke] image:", Path(args.image).resolve())
    print("[smoke] calling local Qwen2.5-VL-3B...")

    raw, latency_ms, meta = provider.identify(
        image_path=Path(args.image).resolve(),
        selection_method="smoke_test",
        yolo_hint=None,
        quality_flags=[],
    )

    cleaned, valid, decision, esc, warnings = validate_and_clean(
        raw, schema
    )

    print(json.dumps({
        "raw": raw,
        "latency_ms": round(latency_ms, 2),
        "provider_metadata": meta,
        "schema_valid": valid,
        "decision": decision,
        "escalation_recommended": esc,
        "warnings": warnings,
        "cleaned": cleaned,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
