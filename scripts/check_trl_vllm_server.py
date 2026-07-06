#!/usr/bin/env python3
import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


def request_json(url: str, payload: Optional[Dict[str, Any]] = None, timeout: float = 20.0) -> Dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        if not body:
            return {}
        return json.loads(body)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check that a TRL vLLM server is reachable.")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--expected-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    print(f"Checking TRL vLLM server at {base}")

    try:
        models = request_json(f"{base}/v1/models")
        ids = [item.get("id") for item in models.get("data", [])]
        print("OpenAI-compatible models:", ids)
        if ids and args.expected_model not in ids:
            print(f"WARNING: expected model {args.expected_model!r}, got {ids!r}")
    except Exception as exc:
        print(f"WARNING: /v1/models check failed: {exc}")

    # TRL >= 1.x renamed /get_tensor_parallel_size/ -> /get_world_size/; the
    # weight-sync endpoints are /init_communicator/ /update_named_param/ etc.
    checks = [
        ("health", "GET", "/health/", None),
        ("world size", "GET", "/get_world_size/", None),
    ]

    for name, method, path, payload in checks:
        try:
            result = request_json(f"{base}{path}", payload=payload)
            print(f"{name}: OK {result}")
        except urllib.error.HTTPError as exc:
            print(f"{name}: FAIL HTTP {exc.code} at {path}", file=sys.stderr)
            print(exc.read().decode("utf-8", errors="replace")[:1000], file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"{name}: FAIL at {path}: {exc}", file=sys.stderr)
            return 1

    print("TRL vLLM server preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
