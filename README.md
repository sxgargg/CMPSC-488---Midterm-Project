# Who Wrote It? Identifying LLMs from Their Responses

CMPSC 448 Midterm Project. We train a CNN and an LSTM to predict which LLM family (Llama, Gemini, or Qwen) wrote a given response.

**Team leader:** Adithya Vemulapali
**Team members:** Adithya Vemulapali, Shreya Garg, Ani Ganesan

Project report: *(add report PDF link here)*

## Dataset

| Family | Model | File | Responses |
|---|---|---|---:|
| Llama | `llama3.1:8b` (Ollama) | `ollama_outputs_llama3.1_8b.json` | 500 |
| Gemini | `gemini-3.8-flash` (Gemini API) | `gemini_outputs_gemini_3.8_flash.jsonl` | 500 |
| Qwen | `qwen2.5:7b` (Ollama) | `ollama_outputs_qwen2.5_7b.json` | 500 |

`llm_classification_prompts_500.json` has 500 prompts across 20 categories (25 each), such as coding, math, poetry, recipes, and history. Every model received the same 500 prompts with temperature 0.7 and a 512-token limit. Each record stores `model_family`, `model_name`, `prompt_id`, `prompt_category`, `prompt`, `output`, `temperature`, `max_tokens`, and `timestamp_utc`.

We collected the data ourselves with `collect_llm_outputs.py` (Gemini API) and `collect_ollama_remote.py` (Ollama models). To speed up the Qwen run, we split it between a Mac and a PC with an RTX 4070 that served the same model.

## Code

| File | Purpose |
|---|---|
| `train_authorship_attribution.py` | Data loading, prompt-grouped split, tokenizer, CNN and LSTM models, training and evaluation |
| `run_experiments.py` | Runs all 14 experiments (skips ones already in `results/`) |
| `analyze_results.py` | Builds `results/summary.csv`, the style-feature baselines, and the figures |
| `predict.py` | Classifies a new response with a saved checkpoint |
| `tests/test_pipeline.py` | Unit tests for splitting, tokenization, and padding handling |

## Results (test set, 150 responses, seed 42)

| Model | Input only | Output only | Input + output |
|---|---:|---:|---:|
| CNN | 33.3% | 86.0% | 90.0% |
| LSTM | 33.3% | 60.7% | 68.0% |

Every prompt is identical across the three families, so input-only accuracy is exactly chance (1/3). For more results (other seeds, the held-out writing domain, and the formatting ablation), see `results/summary.csv` and `figures/`.

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_experiments.py --device auto
python analyze_results.py
```

Train a single model:

```bash
python train_authorship_attribution.py --model cnn --text-source output --out results/my_cnn_run
```

Classify your own text with a trained checkpoint:

```bash
python predict.py --checkpoint checkpoints/cnn_output_s42.pt --output-file response.txt
```
