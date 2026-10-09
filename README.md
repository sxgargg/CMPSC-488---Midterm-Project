# 448 Project 1 — LLM Authorship Attribution

Dataset and collection artifacts for identifying which LLM generated a response. Intended families: Llama, Gemini, and Qwen. CNN and LSTM training is being developed separately; no verified training results or final PDF report are included here.

## Dataset snapshot

| Model | File | Saved responses |
|---|---|---:|
| Llama 3.1 8B | `ollama_outputs_llama3.1_8b.json` | 500/500 |
| Gemini 3.8 Flash | `gemini_outputs_gemini_3.8_flash.jsonl` | 500/500 |
| Qwen 2.5 7B | `ollama_outputs_qwen2.5_7b.json` | 316/500 — partial |

This is a fixed snapshot, not a live sync. Qwen collection continues on the owner's Mac. Snapshot timestamps and SHA-256 checksums are recorded in `DATASET_STATUS.json`.

`llm_classification_prompts_500.json` contains 500 unique prompts across 20 categories (25 each). Each model receives the same prompt text. Saved records contain `model_family`, `model_name`, `prompt_id`, `prompt_category`, `prompt`, `output`, `temperature`, `max_tokens`, and `timestamp_utc`; Ollama records also include `ollama_host`.

The snapshot was checked for unique prompt IDs, exact matching prompt/category text, nonempty outputs, temperature 0.7, and max_tokens 512. Recorded settings describe requested configuration, not independent proof of backend enforcement.

## Collection artifacts

- `collect_llm_outputs.py`: checkpointed API/local collector; used for Gemini.
- `collect_ollama_remote.py`: original owner-edited collector, preserved unchanged; used for Llama and initial Qwen collection.
- `collect_qwen_concurrent.py`: resumable two-worker Qwen collector. Saves atomically in prompt order, using a temporary local Ollama server on port 11435.
- `qwen_collection_events.jsonl`: records the switch to two workers after 94 saved responses.

Qwen uses `qwen2.5:7b`, Q4_K_M, digest `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`. Production has no fixed seed. Pilot/benchmark responses are excluded. The Gemini collector requests `thinking_budget=0`, but thought-token usage was not saved, so actual thinking-off behavior was not verified.

A five-prompt Qwen pilot produced usable explanation, fiction, coding, math, and translation answers. A separate four-prompt benchmark took 38.4 seconds sequentially and 34.4 seconds with two concurrent requests. The production collector switched to concurrency at the owner's request; earlier saved records were verified unchanged.

The Pollinations GPT-OSS dataset was removed at the owner's request following an audit that found provider advertisements, reasoning-only wrappers, and unresolved provenance/token-limit questions. It is not included.

## Training artifact status

`train_authorship_attribution.py` is the original legacy prototype, not the replacement pipeline built by another agent. Known issues: its default `files/` directory does not match this layout; input/output/both selection is absent; tokenization loses non-Latin letters; LSTM padding is not length-aware; CPU best-checkpoint tensors are not cloned. It should not be represented as a validated final pipeline.

The intended evaluation groups all responses to a prompt into the same split: 400/50/50 prompt IDs for train/validation/test, with training-only vocabulary construction. The original implementation groups prompts but does not stratify categories. Required comparisons are CNN and LSTM on input-only, output-only, and input-plus-output.

## Scope and limitations

A 512-token generation cap does not equalize actual output lengths. Some outputs finish early; others are truncated. Each family is represented by one model configuration and collection setup. Final conclusions should reflect that scope. Runtime logs, process IDs, caches, credentials, model weights, and the outdated handoff document are excluded.
