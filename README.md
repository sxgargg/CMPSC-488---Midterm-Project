# 448 Project 1 - LLM Authorship Attribution

CNN and LSTM classification of Llama, Gemini, and Qwen responses. Includes the raw dataset, collection provenance, a tested training pipeline, 14 completed neural experiments, diagnostic baselines, trained output-only checkpoints, and the project report.

**Team:** Adithya Vemulapali, Shreya Garg, Ani Ganesan. **Team leader:** not yet designated, at the team's request.

- [Project report (PDF)](output/pdf/448_project_1_report.pdf)
- [Results table](results/summary.csv)
- [Experiment protocol](EXPERIMENTS.md)
- [Dataset caveats](DATASET_CAVEATS.md)

## Main test results

Fixed 400/50/50 prompt split; all three families stay with each prompt. Test set: 150 responses. Primary seed: 42. Model names are masked in preprocessing; raw text is unchanged.

| Model | Input only | Output only | Input + output |
|---|---:|---:|---:|
| CNN | 33.3% | 86.0% | 90.0% |
| LSTM | 33.3% | 60.7% | 68.0% |

Output-only macro-F1: CNN 0.859; LSTM 0.601. Additional seeds, per-family recall, confusion matrices, prompt-bootstrap intervals, and paired comparisons appear in the report and saved metrics. Held-out writing accuracy is 64.3% for CNN and 50.3% for LSTM. Removing formatting reduces both models' performance. These are results for this dataset and these configurations, not universal authorship guarantees.

## Reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_experiments.py --device auto
python analyze_results.py
python verify_results.py
python build_report.py
```

The suite skips completed matching runs in `results/`. To retrain a single experiment without replacing the published evidence:

```bash
python train_authorship_attribution.py --model cnn --text-source output --out results/my_cnn_run
python train_authorship_attribution.py --model lstm --text-source both --out results/my_lstm_run
```

For a complete fresh suite, use a separate checkout with an empty `results/` directory; do not mix custom experiments into the published report's 14-run grid. `--device auto` prefers CUDA, then MPS, then CPU. Install a CUDA-enabled PyTorch build separately when using an NVIDIA PC. `requirements-lock.txt` records the actual Mac package versions, while `requirements.txt` is the portable dependency list.

Use a published checkpoint on your own response text:

```bash
python predict.py --checkpoint checkpoints/cnn_output_s42.pt --output-file response.txt
```

Predicted probabilities are uncalibrated; the model always chooses among these three families and cannot detect unknown models or human authors. The two versioned checkpoints are the primary output-only models, not retrospectively selected best seeds. Other experiment checkpoints remain under the Git-ignored `models/` directory after training.

`results/<run>/` contains metrics, training history, validation/test predictions, prompt splits, and train-only vocabulary. Domain runs add target predictions. `results/analysis.json` holds diagnostic baselines and paired comparisons. `results/verification.txt` records successful evidence checks. Figures and the PDF/Markdown report are generated from these artifacts. Training code is independent of collection code.

## Dataset snapshot

| Model | File | Saved responses |
|---|---|---:|
| Llama 3.1 8B | `ollama_outputs_llama3.1_8b.json` | 500/500 |
| Gemini 3.8 Flash | `gemini_outputs_gemini_3.8_flash.jsonl` | 500/500 |
| Qwen 2.5 7B | `ollama_outputs_qwen2.5_7b.json` | 500/500 |

Collection is complete: 1,500 responses across three families. This is a fixed snapshot, not a live sync. Snapshot timestamps and SHA-256 checksums are recorded in `DATASET_STATUS.json`.

`llm_classification_prompts_500.json` contains 500 unique prompts across 20 categories (25 each). Each model receives the same prompt text. Saved records contain `model_family`, `model_name`, `prompt_id`, `prompt_category`, `prompt`, `output`, `temperature`, `max_tokens`, and `timestamp_utc`; Ollama records also include `ollama_host`.

The snapshot was checked for unique prompt IDs, exact matching prompt/category text, nonempty outputs, temperature 0.7, and max_tokens 512. Recorded settings describe requested configuration, not independent proof of backend enforcement.

## Collection artifacts

- `collect_llm_outputs.py`: checkpointed API/local collector; used for Gemini.
- `collect_ollama_remote.py`: original owner-edited collector, preserved unchanged; used for Llama and initial Qwen collection.
- `collect_qwen_concurrent.py`: resumable two-worker Qwen collector. Saves atomically in prompt order, using a temporary local Ollama server on port 11435.
- `collect_qwen_pc.py`: sequential remote RTX 4070 collector.
- `collect_qwen_split.py`: one worker each on the Mac and RTX 4070, dynamically assigning pending prompts and saving through a single atomic writer. Supports resuming noncontiguous saved prompt IDs.
- `qwen_collection_events.jsonl`: checkpoint hashes and execution changes after 94, 347, and 363 saved responses.
- `qwen_pc_metrics.jsonl` and `qwen_split_metrics.jsonl`: token counts, finish reasons, and timing for the remote and dual-host stages; earlier stages did not save these metrics.

Qwen uses `qwen2.5:7b`, Q4_K_M, digest `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`. Production has no fixed seed. Pilot/benchmark responses are excluded. The Gemini collector requests `thinking_budget=0`, but thought-token usage was not saved, so actual thinking-off behavior was not verified.

A five-prompt Qwen pilot produced usable explanation, fiction, coding, math, and translation answers. A separate four-prompt benchmark took 38.4 seconds sequentially and 34.4 seconds with two concurrent requests. The production collector switched to concurrency at the owner's request, then to the RTX 4070 after 347 saved responses, and to both machines after 363. Both machines had the same model digest. Remote and dual-host requests explicitly set `num_ctx=4096`, temperature 0.7, and `num_predict=512`, with no seed or other sampling overrides. All previously saved records were preserved. Hardware and runtime differences may still affect generated text.

Final Qwen records by endpoint: `http://127.0.0.1:11434`: 131, `http://127.0.0.1:11435`: 253, `http://100.82.174.98:11434`: 116. Endpoint and timing metadata must not be used as classifier inputs.

The Pollinations GPT-OSS dataset was removed at the owner's request following an audit that found provider advertisements, reasoning-only wrappers, and unresolved provenance/token-limit questions. It is not included.

## Scope and limitations

A 512-token generation cap does not equalize actual output lengths. Some outputs finish early; others are truncated. Each family is represented by one model configuration and collection setup. Final conclusions should reflect that scope. Runtime logs, process IDs, caches, credentials, model weights, and the outdated handoff document are excluded.

Dataset audit: all 1,500 records have matching prompts/categories and nonempty outputs, with no duplicate prompts or exact duplicate outputs after whitespace/case normalization. Qwen response 84 explicitly mentions “Qwen.” Gemini response 51 contains only two haiku lines; its stopping reason was not recorded. Of the 153 Qwen responses with saved finish metadata, 84 reached the generation limit. Raw responses are preserved unchanged.
