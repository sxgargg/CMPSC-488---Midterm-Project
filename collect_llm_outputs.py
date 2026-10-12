#!/usr/bin/env python3
"""
Collect (model, prompt, output) records for every prompt in
llm_classification_prompts_500.json. We used this script for Gemini.

Each response is appended to a .jsonl file with the fields
  model_family, model_name, prompt_id, prompt_category,
  prompt, output, temperature, max_tokens, timestamp_utc
The file is saved after every response, so an interrupted run can be
restarted with the same command and finished prompts are skipped.

Setup:
  pip install google-genai openai requests
  export GEMINI_API_KEY="..."    # read from the environment, never stored

Example:
  python collect_llm_outputs.py --models gemini-3.8-flash --limit 5
  python collect_llm_outputs.py --models gemini-3.8-flash \
      --out gemini_outputs_gemini_3.8_flash.jsonl

Temperature (0.7) and max_tokens (512) are the same for every model and
are recorded on every row.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import requests

# ---------------------------------------------------------------------------
# Model registry. Key = name you pass with --models.
# "family" is the classification label. Add rows here for more models.
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    # --- Google Gemini (needs GEMINI_API_KEY) ---
    "gemini-3.5-flash": {"family": "gemini", "kind": "gemini",
                         "api_name": "gemini-3.5-flash"},
    "gemini-3.6-flash": {"family": "gemini", "kind": "gemini",
                         "api_name": "gemini-3.6-flash"},  # fallback; tested 2026-09-23
    "gemini-3.8-flash": {"family": "gemini", "kind": "gemini",
                         "api_name": "gemini-3.8-flash"},
    # --- OpenAI (needs OPENAI_API_KEY — the API, not a ChatGPT subscription) ---
    "gpt-5.5":          {"family": "gpt",    "kind": "openai",
                         "api_name": "gpt-5.5"},
    # --- Local open-source models via Ollama (no key needed) ---
    "llama3.1:8b":      {"family": "llama",    "kind": "ollama",
                         "api_name": "llama3.1:8b"},
    "qwen2.5:7b":       {"family": "qwen",     "kind": "ollama",
                         "api_name": "qwen2.5:7b"},
    "deepseek-r1:8b":   {"family": "deepseek", "kind": "ollama",
                         "api_name": "deepseek-r1:8b"},
    "mistral:7b":       {"family": "mistral",  "kind": "ollama",
                         "api_name": "mistral:7b"},
}


# ---------------------------------------------------------------------------
# Clients — one tiny wrapper per provider, same .generate() interface
# ---------------------------------------------------------------------------
class GeminiClient:
    def __init__(self, api_name):
        from google import genai
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            sys.exit("ERROR: set GEMINI_API_KEY (get one free at aistudio.google.com)")
        self.client = genai.Client(api_key=key)
        self.api_name = api_name

    def generate(self, prompt, temperature, max_tokens):
        from google.genai import types
        resp = self.client.models.generate_content(
            model=self.api_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return resp.text or ""


class OpenAIClient:
    def __init__(self, api_name):
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            sys.exit("ERROR: set OPENAI_API_KEY (platform.openai.com — the API, "
                     "not your ChatGPT subscription)")
        self.client = OpenAI(api_key=key)
        self.api_name = api_name

    def generate(self, prompt, temperature, max_tokens):
        resp = self.client.chat.completions.create(
            model=self.api_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


class OllamaClient:
    def __init__(self, api_name):
        self.api_name = api_name
        # fail fast if Ollama isn't running / model isn't pulled
        try:
            requests.get("http://localhost:11434/api/tags", timeout=5).raise_for_status()
        except Exception:
            sys.exit("ERROR: Ollama not reachable. Start it and run "
                     f"`ollama pull {api_name}` first.")

    def generate(self, prompt, temperature, max_tokens):
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": self.api_name,
                  "prompt": prompt,
                  "stream": False,
                  "options": {"temperature": temperature,
                              "num_predict": max_tokens}},
            timeout=900,
        )
        r.raise_for_status()
        return r.json().get("response", "")


def make_client(model_key):
    spec = MODEL_REGISTRY[model_key]
    if spec["kind"] == "gemini":
        return GeminiClient(spec["api_name"])
    if spec["kind"] == "openai":
        return OpenAIClient(spec["api_name"])
    if spec["kind"] == "ollama":
        return OllamaClient(spec["api_name"])
    raise ValueError(f"unknown provider kind: {spec['kind']}")


# ---------------------------------------------------------------------------
# Runner with checkpointing + retries
# ---------------------------------------------------------------------------
def load_done(path):
    """(model_name, prompt_id) pairs already saved — for resume."""
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    done.add((row["model_name"], row["prompt_id"]))
                except json.JSONDecodeError:
                    pass
    return done


def with_retry(fn, tries=5):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # rate limits, timeouts, transient 5xx
            wait = 2 ** i
            print(f"    retry {i + 1}/{tries} in {wait}s "
                  f"({type(e).__name__}: {e})", flush=True)
            time.sleep(wait)
    raise RuntimeError("giving up after retries — fix the error and re-run to resume")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    choices=list(MODEL_REGISTRY),
                    help="model keys from MODEL_REGISTRY")
    ap.add_argument("--prompts", default="llm_classification_prompts_500.json")
    ap.add_argument("--out", default="llm_outputs.jsonl")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="fixed for ALL models — record it, don't vary it")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--delay", type=float, default=1.0,
                    help="seconds between API calls (rate-limit courtesy)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only run first N prompts (for smoke tests)")
    args = ap.parse_args()

    with open(args.prompts, encoding="utf-8") as f:
        prompts = json.load(f)
    if args.limit:
        prompts = prompts[:args.limit]

    done = load_done(args.out)
    if done:
        print(f"Resuming: {len(done)} (model, prompt) pairs already collected.")

    out = open(args.out, "a", encoding="utf-8")
    try:
        for model_key in args.models:
            spec = MODEL_REGISTRY[model_key]
            client = make_client(model_key)
            print(f"\n=== {model_key} (family: {spec['family']}) ===", flush=True)
            for p in prompts:
                if (model_key, p["id"]) in done:
                    continue
                text = with_retry(
                    lambda: client.generate(p["prompt"], args.temperature,
                                            args.max_tokens))
                row = {
                    "model_family": spec["family"],
                    "model_name": model_key,
                    "prompt_id": p["id"],
                    "prompt_category": p["category"],
                    "prompt": p["prompt"],
                    "output": text.strip(),
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                done.add((model_key, p["id"]))
                print(f"  [{p['id']}/{prompts[-1]['id']}] {len(text)} chars",
                      flush=True)
                time.sleep(args.delay)
    finally:
        out.close()
    print(f"\nDone. Dataset at {args.out} "
          f"({sum(1 for _ in open(args.out, encoding='utf-8'))} rows)")


if __name__ == "__main__":
    main()
