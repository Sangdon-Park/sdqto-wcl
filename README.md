# State-Dependent Quality Threshold Offloading for Wireless Edge-Cloud LLM Inference

Reproducibility code for the IEEE WCL letter "State-Dependent Quality Threshold Offloading for Wireless Edge-Cloud LLM Inference" by Sangdon Park and Joohyung Lee.

## What this repository contains

| Path | Purpose |
|------|---------|
| `simulator.py` | Trace-driven simulator that produces every numerical result and figure in the paper. |
| `profile_cloud.py` | Optional script that profiles a streaming cloud LLM endpoint (Gemini-3-Flash by default) to obtain the cloud TTFT and per-token decode times used by `simulator.py`. |
| `cached/experiment_summary.json` | Output of one full simulator run with the paper seed. Lets reviewers verify the headline numbers without re-running. |
| `cached/llm_profile_summary.json` | Measured edge profile (Qwen2.5-0.5B-Instruct) and cached cloud profile used to drive the simulator. |
| `cached/service_prompt_bank.json` | The 15-prompt bank that determines per-request prompt length and difficulty. |
| `requirements.txt` | NumPy and Matplotlib version pins. |

## Reproducing the paper results

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python simulator.py
```

`simulator.py` is deterministic: it reads `cached/llm_profile_summary.json` and `cached/service_prompt_bank.json`, runs the 4000-request trace with seed `20260315`, and writes the full statistics to `outputs/experiment_summary.json` along with all paper figures (`baseline_comparison.pdf`, `state_trace.pdf`, `load_ablation.pdf`, etc.) under `outputs/`.

Expected runtime on a recent laptop: about 4-6 minutes (most of it is the load-sweep and oracle ablations).

## Verifying without running

The cached run that backs the paper numbers is in `cached/experiment_summary.json`. The headline values are:

```
proposed.objective    = 5.42         improvements.vs_nominal      = 20.7%
proposed.delay_p95    = 9.07 s       improvements.vs_delay_aware  =  2.8%
proposed.shortfall    = 0.031        improvements.vs_rollout_h5   = -4.9%
```

These should match the paper's Section IV.B exactly.

## Re-profiling the cloud (optional)

`cached/llm_profile_summary.json` already contains streamed Gemini-3-Flash TTFT and per-token measurements. To re-profile:

```bash
export GEMINI_API_KEY=<your key>
python profile_cloud.py
```

The script merges the new cloud measurements into `cached/llm_profile_summary.json` while preserving the edge profile.

## Override knobs

A few environment variables control non-default behavior:

- `STATE_TRACE_PINNED_START` - integer; pins the burst window used in the state-trace figure to start at a specific arrival index. Used to keep figures aligned with hard-coded numbers in the paper prose.
- `GEMINI_MODEL`, `GEMINI_TEMPERATURE`, `GEMINI_PROFILE_REPEATS`, `GEMINI_THINKING_BUDGET` - cloud-profiling overrides for `profile_cloud.py`.

## Citing

```bibtex
@article{park2026sdqto,
  author  = {Sangdon Park and Joohyung Lee},
  title   = {State-Dependent Quality Threshold Offloading for Wireless Edge-Cloud {LLM} Inference},
  journal = {IEEE Wireless Communications Letters},
  year    = {2026}
}
```

## License

MIT, see `LICENSE`.
