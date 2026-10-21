# llm-eval-harness
A config-driven Python harness that benchmarks small open-source LLMs on English and Arabic tasks (sentiment, QA), reporting accuracy, F1, and confidence intervals alongside error analysis and reproducible results.

> **Status:** work in progress (Phases 0-1 complete, Phase 2 in progress). Results, quick start and method sections will be added as the project is built.

## Draft notes (Step 0.4)

**Models:** Qwen2.5-1.5B-Instruct, Gemma-3-1B-it, Hala-700M (Arabic-focused)
**Tasks:** SST-2 (validation, n=200), SQuAD v1.1 (validation, n=200), AJGT Jordanian-dialect Arabic sentiment (`komari6/ajgt_twitter_ar`, n=200)
**Seed:** 42
**Environment:** transformers 5.x, versions frozen in `requirements.txt`

**Licenses:** Qwen2.5-1.5B-Instruct: Apache 2.0. Gemma-3-1B-it: Gemma license (gated).
Hala-700M: [CC BY-NC 4.0](https://spdx.org/licenses/CC-BY-NC-4.0) (non-commercial use only). SQuAD v1.1: CC BY-SA 4.0.
SST-2 and AJGT: license listed as "unknown" on their Hugging Face cards.

**Notes from setup:**
- Jais-590M and Arabic Stable LM 2 1.6B were dropped: their custom code broke on current transformers.
- Chat template plus a "one word only" instruction stopped rambling outputs.
- The Arabic prompt needed `max_new_tokens=24` and a parser that looks at the first few words.
- Hala-700M was timed without the optimized conv kernel, so its latency isn't directly comparable.