# Dataset caveats and audit handoff

Audit date: October 9, 2026 (America/New_York). This file records the collection audit. A new pipeline and completed experiments were added on October 11, 2026; see EXPERIMENTS.md and the project report.

## What was verified

- 1,500 nonempty responses: 500 each for `llama`, `gemini`, and `qwen`.
- Each family has prompt IDs 1–500 exactly once, with 20 categories and 25 prompts per category.
- Every record's prompt and category match the master prompt file exactly across all families.
- Every record specifies temperature 0.7 and max_tokens 512.
- No duplicate prompts or exact duplicate outputs were found after whitespace normalization and case folding. This was not a semantic near-duplicate audit.
- File hashes match `DATASET_STATUS.json`. Existing Qwen records were preserved during each execution change.
- Pattern checks found no `<think>` blocks, sampled chat-template markers, or outputs starting with the checked HTTP/error/traceback patterns. These checks are not an exhaustive content or provenance guarantee.

## Known content issues

1. **Explicit model identity:** Qwen prompt 84 (parody of a motivational speech) says: `Now, I know what you’re thinking: "Qwen, that’s a bit far-fetched, isn’t it?"` This is a direct identity clue rather than purely stylistic evidence. The raw response is unchanged. A model-name-masking comparison can measure sensitivity to this clue; apply any rule consistently across families.
2. **Incomplete short response:** Gemini prompt 51 asks for a haiku about autumn rain but contains only two lines: `Rain taps amber leaves,` and `Puddles catch the fallen gold,`. Its finish reason and token usage were not saved, so the cause cannot be established from the artifacts. Do not claim it was definitely truncated by the cap or by thinking.
3. **Short responses are not automatically failures:** Riddles, translations, and slogans legitimately produce short answers. Llama prompt 297 asks the user to provide the tongue twister rather than completing the task. These remain genuine recorded outputs; correctness on the underlying task is distinct from authorship attribution.
4. **Truncation is common:** Of the 153 Qwen responses with saved finish metadata, 84 have `done_reason="length"` and 69 have `stop`. The maximum recorded generated-token count in those metrics is 512. Finish metadata is unavailable for the first 347 Qwen responses and for the Llama/Gemini datasets, so this fraction must not be presented as a whole-dataset truncation rate.

## Collection limitations

- A common 512-token cap does not equalize actual output lengths. Tokenizers differ across families, and many responses finish early or end mid-sentence. Median whitespace-word counts are 337 for Llama, 316.5 for Gemini, and 333 for Qwen; these are not model-token counts.
- Recorded settings describe the requested configuration, not independent proof of backend enforcement on every record. The Gemini collector requests thinking_budget=0, but thought-token usage and full response metadata were not saved; actual thinking-off behavior is unverified.
- Each family is represented by one named model/configuration. Results cannot establish generalization to all models in that family. Names in records are collection labels; saved text alone cannot independently prove backend identity.
- Qwen used the same `qwen2.5:7b` Q4_K_M digest on Mac and PC: `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`. Production requests had no fixed seed. Hardware, runtime, and concurrency can still change sampled text despite a matching digest.
- Qwen stages: first 94 responses on the Mac sequentially; through response 347 on the Mac with two workers; responses 348–363 on the RTX 4070 sequentially; remaining responses dynamically distributed between one Mac worker and one PC worker. Completed rows were saved by a single writer and sorted by prompt ID. Events and timing sidecars document these changes.
- Explicit num_ctx=4096 is present in PC and split-collector requests. Earlier collectors did not explicitly send num_ctx in each request; do not describe their request payloads as identical in this respect.
- Collection stage is associated with prompt order/category. Host or stage effects cannot be cleanly separated from task effects in this dataset.
- The Pollinations GPT-OSS dataset is excluded following provider-advertisement, reasoning-wrapper, and provenance concerns. It is not a fourth family in this repository.

## Data interface and evaluation pitfalls

- Llama and Qwen are JSON arrays; Gemini is JSONL (one object per line). An array-only or `*.json`-only loader will miss Gemini.
- The text fields are `prompt` and `output`; the target is `model_family`. Do not feed serialized records or metadata such as model_name, ollama_host, timestamps, prompt_id, or category into the classifier.
- All three responses for a prompt must stay together in a split. The intended split is 400/50/50 prompt IDs, giving 1,200/150/150 examples. Fit vocabulary and other learned preprocessing only on training data.
- Prompts are identical and equally represented for all three labels. For a deterministic input-only classifier evaluated on complete prompt triplets, accuracy is exactly 1/3: it makes the same prediction on three identical inputs with different labels. A clear deviation calls for checking data balance, evaluation stochasticity, and leakage. Adding prompts to outputs can still affect classification through task context; improvement is not ruled out by the input-only result.
- Grouping by prompt does not itself stratify categories. The removed legacy implementation's seed-42 split omitted math from validation and code_debugging from test. The replacement pipeline needs its own split audit.
- Non-Latin text and formatting are present. The removed prototype's ASCII-oriented tokenizer discarded non-Latin letters. Its 128-token limit truncated 1,276/1,500 responses under that tokenizer; this is a preprocessing statistic, not the model generation limit.
- The legacy training prototype was removed and has now been replaced by the tested pipeline in this repository. It loads all three files explicitly, preserves Unicode letters, groups and stratifies prompt splits, and handles padding. New experiment results are documented separately; historical prototype statistics above are not claims about the replacement pipeline.

Raw dataset records were not modified to remove these issues. Any filtering, masking, or ablation belongs in a documented derived preprocessing step, retaining the original artifacts and reporting its effect.
