# Lab 21: upstream contribution

**Status:** Planned (start collecting candidates from Lab 01 onward)  
**Baseten posting items:** "Contribute upstream to open-source inference engines (vLLM, SGLang, TensorRT-LLM)"  
**Needs:** time; a GPU only for some fixes · **Depends on:** any lab

## 1. The question
Can I find, fix and land a real improvement in an inference engine?

## 2. Mental model

### 2.1 How open-source contribution works
1. **Find** something real: a bug, a missing metric, unclear documentation, a missing test, a benchmark script, model support. The best candidates come from your own work in these labs.
2. **Check the project's guide** (`CONTRIBUTING`): coding style, how to run tests, sign-off requirements (some projects require a Developer Certificate of Origin sign-off on commits), PR template, whether to open an issue first.
3. **Reproduce** with a minimal case and record versions.
4. **Fix and test**: the smallest change that solves the problem, plus a test that fails before and passes after.
5. **Open the PR**: clear title, what and why, how you tested, benchmark numbers if it is a performance change. Keep it small.
6. **Respond to review** promptly and politely; expect requests for changes, CI failures, and delays.

### 2.2 Where to look
vLLM (`vllm-project/vllm`), SGLang (`sgl-project/sglang`), TensorRT-LLM (`NVIDIA/TensorRT-LLM`). Look for issues labelled for new contributors, documentation gaps, and failures you hit yourself. Read recent merged PRs to learn the review style.

### 2.3 Good first contributions, ranked by risk
1. Documentation fixes and clarifications (low risk, high learning).
2. A reproducible bug report with a minimal case (useful even without a fix).
3. Tests and benchmark scripts.
4. Metrics, logging and warning improvements.
5. Small bug fixes.
6. Model support and performance work (larger; discuss in an issue first).

### 2.4 Candidates already visible from these labs
- A warning when the server's concurrency cap is below what a benchmark requests (a lesson from Lab 05), or documentation of the interaction between `max_num_seqs` and observed concurrency.
- Documentation of benchmarking pitfalls (prefix-cache contamination from repeated prompts).
- Anything that surprises you in Labs 06-19, such as a flag that behaves differently than documented.

## 3. Predictions
Before starting, write what you expect review to focus on and how long a first PR will take. Compare afterwards.

## 4. Tracking
| Date | Repo | PR / issue | Type | Status | What review taught me |
|---|---|---|---|---|---|

## 9. Done when
- At least one PR opened, ideally merged.
- A short note on what review feedback changed.

## 10. Interview questions
1. Describe a change you made upstream and what the maintainers pushed back on.
2. How do you reproduce an engine bug minimally?
3. How did you decide this change was worth proposing?
4. How do you keep a performance-related PR credible?

## Further reading
- Open Source Guides: "How to Contribute to Open Source" (opensource.guide).
- The CONTRIBUTING files of vLLM, SGLang and TensorRT-LLM.
- Recently merged pull requests in the target repository (to learn the review style).
