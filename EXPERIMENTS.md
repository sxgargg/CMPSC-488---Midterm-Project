# Experiment protocol

The experiment grid was specified before the full training suite ran. There is no hyperparameter search or test-driven model selection. A one-epoch CPU CNN smoke run was used to verify execution; its test prediction is not a reported experiment and did not change the grid.

## Primary comparisons (RQ1 and RQ2)

Train CNN and unidirectional LSTM on input only, output only, and input + output, seed 42. Use the same fixed split for all runs: 20 training prompts in each of 20 categories, with remaining five divided 2/3 or 3/2 between validation and test. All three family responses stay with their prompt. This yields 1,200/150/150 examples (400/50/50 prompts). Initialization robustness repeats output-only at seeds 7 and 123, retaining the same split.

Unicode word/punctuation/newline tokenization, lowercasing, model-name masking applied consistently to both text fields, minimum vocabulary frequency 2, train-only vocabulary. At most 256 response tokens and 64 prompt tokens. For `both`, append a separator and the SAME capped response after the capped prompt (maximum 321 tokens); the prompt does not displace response tokens. Truncation and unknown-token rates are saved. Generation sampling and raw data are unchanged.

CNN: random 128-dimensional embeddings; 64 convolution filters each of widths 3, 4, 5; ReLU; masked global max pooling; dropout 0.4; three-class head. Sequences shorter than a kernel use one padded window. Entirely padded trailing windows cannot win the maximum.

LSTM: random 128-dimensional embeddings, one unidirectional layer with 128 hidden units, dropout 0.4 on the last REAL token state, three-class head. Gathering before right padding is correct for unidirectional recurrence and supports the Mac GPU without packed-sequence dependence. This is intentionally not the old bidirectional prototype.

Both: AdamW, learning rate 0.001, weight decay 0.01, cross entropy, batch size 32, gradient-norm clipping 1.0, maximum 15 epochs, patience 4 by validation loss (minimum decrease 0.0001). Save a cloned best state and restore it before evaluation. No test-based checkpoint selection. MPS/CUDA floating-point behavior may vary across runs/hardware despite seeded RNGs.

Report accuracy, macro-F1, per-family recall (called per-family accuracy here), and confusion matrices with true rows/predicted columns. Accuracy intervals resample prompt triplets rather than individual rows (2,000 bootstrap draws); paired ablation differences use 5,000 draws. Intervals describe prompt sampling within this fixed dataset, not uncertainty over all LLMs or collection runs. Three initialization seeds are descriptive, not three independent test sets.

## Held-out domains (RQ3)

Hold out all 100 prompts in creative_fiction, poetry, marketing_copy, and business_writing. The other 16 categories supply 320/40/40 train/validation/source-test prompts (960/120/120 examples). Target evaluation has 300 examples. Train each architecture on output only, seed 42, and select checkpoints using SOURCE validation only. No target data enters vocabulary or tuning. Compare source-test and held-out target accuracy from the same model; the task mixes differ, so the difference is not a controlled causal effect of domain alone.

## Signals and ablations (RQ4)

Describe length, formatting, and lexical diversity for the complete corpus. Fit diagnostic logistic-regression classifiers (C=1, maximum 2,000 iterations) on (a) word count alone and (b) eleven structural/lexical summary features. StandardScaler fits only on training data. Evaluate on the same primary test split. These baselines access full responses, unlike the neural token cap, and are diagnostic rather than exactly input-matched competitors.

Retrain each neural model from scratch with punctuation and newlines removed (`plain` transform), seed 42. Both primary and plain preprocessing already lowercase text and mask model names. The ablation changes formatting AND the effective number of words within the 256-token budget; it cannot isolate punctuation causally. No punctuation/case result should be presented as proving which individual signal the model uses.

## Reproduction and artifacts

`python run_experiments.py --device auto` executes 14 runs and skips completed matching configurations. `python analyze_results.py` creates summaries, paired intervals, diagnostic baselines, and figures. `python build_report.py` builds the PDF and Markdown report from saved results and `TEAM.json`. Checkpoints are saved locally in `models/` (excluded from Git); the two primary output-only checkpoints are also versioned in `checkpoints/`; metrics, histories, vocabularies, splits, and predictions are versioned under `results/`. `requirements-lock.txt` records the execution environment; macOS pinned wheels may differ from CUDA installations.
