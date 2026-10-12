#!/usr/bin/env python3
"""
Collect (model, prompt, output) records from an Ollama server (local or
on another machine) or from the Gemini API. We used this script for
Llama and Qwen. Standard library only.

Usage:
  python3 collect_ollama_remote.py --list-models
  python3 collect_ollama_remote.py --models llama3.1:8b --limit 5
  python3 collect_ollama_remote.py --models llama3.1:8b qwen2.5:7b

Requests are retried with backoff, and finished prompts are skipped on
re-run. Output is one JSON array per model, e.g.
ollama_outputs_llama3.1_8b.json, with the fields
  model_family, model_name, prompt_id, prompt_category,
  prompt, output, temperature, max_tokens, timestamp_utc, ollama_host
model_family is derived from the model name (llama3.1:8b -> llama).
Temperature and max_tokens are kept identical across all models.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

DEFAULT_HOST = "http://100.82.174.98:11434"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta"


def api(host, path, payload=None, timeout=900):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(host + path, data=data,
                                 method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:300]}")
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"cannot reach {host} ({e.reason}). Is Ollama running there, "
            "and is this machine on the same network?")


def list_models(host):
    d = api(host, "/api/tags", timeout=15)
    return [m["name"] for m in d.get("models", [])]


def generate(host, model, prompt, temperature, max_tokens):
    d = api(host, "/api/generate",
            {"model": model, "prompt": prompt, "stream": False,
             "options": {"temperature": temperature,
                         "num_predict": max_tokens}})
    return d.get("response", "")


def generate_gemini(api_key, model, prompt, temperature, max_tokens, ssl_context=None):
    """One generateContent call against the Gemini API (AI Studio key)."""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            # Disable thinking so 100% of maxOutputTokens is spent on visible output
            "thinkingConfig": {
                "thinkingBudget": 0
            }
        },
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{GEMINI_ENDPOINT}/models/{model}:generateContent",
        data=data,
        headers={"Content-Type": "application/json",
                 "x-goog-api-key": api_key},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=300, context=ssl_context) as r:
            d = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:300]}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"cannot reach Gemini API ({e.reason})")

    cands = d.get("candidates") or []
    if not cands:
        fb = d.get("promptFeedback", {})
        reason = fb.get("blockReason")
        if reason:
            raise NonRetryableError(
                f"prompt refused by API (blockReason={reason}); "
                f"edit or drop this prompt, then re-run to resume")
        raise RuntimeError("empty response (no candidates)")

    cand = cands[0]
    finish_reason = cand.get("finishReason")
    if finish_reason and finish_reason not in ("STOP", None):
        print(f"    NOTICE: candidate finishReason={finish_reason}", flush=True)

    parts = cand.get("content", {}).get("parts", [])
    # Only keep visible text parts, filtering out any thought traces
    text_parts = [p.get("text", "") for p in parts if not p.get("thought")]
    text = "".join(text_parts)
    if not text.strip():
        raise RuntimeError(
            f"empty response text (finishReason={finish_reason})")
    return text


def list_gemini_models(api_key):
    req = urllib.request.Request(
        f"{GEMINI_ENDPOINT}/models?pageSize=100",
        headers={"x-goog-api-key": api_key}, method="GET")
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read().decode())
    return [m["name"].replace("models/", "") for m in d.get("models", [])]


def family_of(model):
    base = model.split(":")[0]
    m = re.match(r"[a-zA-Z]+", base)
    return m.group(0).lower() if m else base


def safe_name(model):
    """llama3.1:8b -> llama3.1_8b (filesystem-friendly)."""
    return re.sub(r"[^A-Za-z0-9.]+", "_", model).strip("_")


def out_path(provider, outdir, model):
    return os.path.join(
        outdir, f"{provider}_outputs_{safe_name(model)}.json")


def load_existing(path):
    """Return (rows list, set of completed prompt_ids)."""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        return rows, {r["prompt_id"] for r in rows}
    return [], set()


def save_rows(path, rows):
    """Rewrite the whole per-model JSON file (atomic via temp file)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


class NonRetryableError(RuntimeError):
    """Errors that will never succeed on retry: bad key, unknown model,
    malformed request, or a prompt the API refuses outright."""


def _http_status(e):
    m = re.search(r"HTTP (\d{3})", str(e))
    return int(m.group(1)) if m else None


def with_retry(fn, initial_wait=2.0, max_wait=120.0):
    """Call fn() until it succeeds, retrying transient failures forever
    with capped exponential backoff (2s, 4s, 8s ... up to 120s).
    Ctrl-C still interrupts cleanly, and re-running the script resumes
    from the last saved prompt. HTTP 4xx (except 429) is not retried."""
    wait = initial_wait
    attempt = 0
    while True:
        try:
            return fn()
        except NonRetryableError:
            raise
        except Exception as e:
            status = _http_status(e)
            if status is not None and 400 <= status < 500 and status != 429:
                raise NonRetryableError(f"not retrying HTTP {status}: {e}")
            attempt += 1
            print(f"    attempt {attempt} failed "
                  f"({type(e).__name__}: {str(e)[:160]}); "
                  f"retrying in {wait:.0f}s ...", flush=True)
            time.sleep(wait)
            wait = min(wait * 2, max_wait)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["ollama", "gemini"],
                    default="ollama",
                    help="which backend to call")
    ap.add_argument("--host", default=DEFAULT_HOST,
                    help="Ollama server base URL (ollama provider only)")
    ap.add_argument("--api-key", default=None,
                    help="Gemini API key (or set GEMINI_API_KEY env var)")
    ap.add_argument("--list-models", action="store_true",
                    help="print available models and exit")
    ap.add_argument("--models", nargs="+", default=None,
                    help="model names, e.g. llama3.1:8b or gemini-3.6-flash")
    ap.add_argument("--prompts", default="llm_classification_prompts_500.json")
    ap.add_argument("--outdir", default=".",
                    help="folder for the per-model JSON files")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="fixed for ALL models — record it, don't vary it")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--delay", type=float, default=None,
                    help="seconds between requests "
                         "(default 0.5 ollama, 1.0 gemini)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only run first N prompts (smoke tests)")
    args = ap.parse_args()

    provider = args.provider
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    delay = args.delay
    if delay is None:
        delay = 1.0 if provider == "gemini" else 0.5

    if args.list_models:
        if provider == "gemini":
            if not api_key:
                sys.exit("ERROR: set GEMINI_API_KEY env var or pass --api-key "
                         "to list Gemini models.")
            models = list_gemini_models(api_key)
            print("Gemini models available to this key:")
        else:
            models = list_models(args.host)
            print(f"Models on {args.host}:")
        for m in models:
            print(f"  {m}  (family label would be: {family_of(m)})")
        if not models:
            print("  (none found)")
        return

    if provider == "gemini" and not api_key:
        sys.exit("ERROR: set GEMINI_API_KEY env var or pass --api-key "
                 "(export GEMINI_API_KEY=\"<your AI Studio key>\").")

    if not args.models:
        sys.exit("ERROR: pass --models, or run with --list-models first to see what's available.")

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.prompts, encoding="utf-8") as f:
        prompts = json.load(f)
    if args.limit:
        prompts = prompts[:args.limit]

    for model in args.models:
        family = family_of(model)
        path = out_path(provider, args.outdir, model)
        rows, done = load_existing(path)
        if done:
            print(f"Resuming {model}: {len(done)} prompts already collected.")
        where = GEMINI_ENDPOINT if provider == "gemini" else args.host
        print("")
        print(f"=== {model} (family: {family}) via {provider} on {where}",
              flush=True)
        print(f"    writing to {path}", flush=True)
        for p in prompts:
            if p["id"] in done:
                continue
            if provider == "gemini":
                text = with_retry(
                    lambda: generate_gemini(api_key, model, p["prompt"],
                                            args.temperature,
                                            args.max_tokens))
            else:
                text = with_retry(
                    lambda: generate(args.host, model, p["prompt"],
                                     args.temperature, args.max_tokens))
            row = {
                "model_family": family,
                "model_name": model,
                "prompt_id": p["id"],
                "prompt_category": p["category"],
                "prompt": p["prompt"],
                "output": text.strip(),
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            if provider == "gemini":
                row["provider"] = "google_gemini_api"
                row["endpoint"] = GEMINI_ENDPOINT
            else:
                row["ollama_host"] = args.host
            rows.append(row)
            save_rows(path, rows)
            done.add(p["id"])
            print(f"  [{p['id']}/{prompts[-1]['id']}] {len(text)} chars",
                  flush=True)
            time.sleep(delay)
        print(f"Done: {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
