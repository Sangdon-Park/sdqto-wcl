from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from statistics import mean, pstdev
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
CACHE_DIR = ROOT_DIR / "cached"
PROFILE_PATH = CACHE_DIR / "llm_profile_summary.json"
PROMPT_BANK_PATH = CACHE_DIR / "service_prompt_bank.json"

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
DEFAULT_REPEATS = int(os.environ.get("GEMINI_PROFILE_REPEATS", "1"))
DEFAULT_TEMPERATURE = float(os.environ.get("GEMINI_TEMPERATURE", "0.2"))


def load_prompt_bank() -> dict[str, list[dict[str, object]]]:
    return json.loads(PROMPT_BANK_PATH.read_text(encoding="utf-8"))


def maybe_build_generation_config(model_name: str, target_new_tokens: int) -> dict[str, object]:
    config: dict[str, object] = {
        "maxOutputTokens": int(target_new_tokens),
        "temperature": DEFAULT_TEMPERATURE,
    }
    if "pro" in model_name:
        config["thinkingConfig"] = {"thinkingBudget": int(os.environ.get("GEMINI_THINKING_BUDGET", "128"))}
    return config


def iter_sse_payloads(resp) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for raw_line in resp:
        line = raw_line.decode("utf-8", errors="ignore").strip()
        if not line or not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        events.append(json.loads(payload))
    return events


def candidate_text_length(payload: dict[str, object]) -> int:
    total_chars = 0
    for candidate in payload.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                total_chars += len(text)
    return total_chars


def generate_once(model_name: str, prompt_text: str, target_new_tokens: int) -> dict[str, float]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:streamGenerateContent?alt=sse&key={urllib.parse.quote(api_key)}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": maybe_build_generation_config(model_name, target_new_tokens),
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    first_chunk_time: float | None = None
    payloads: list[dict[str, object]] = []
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw_line in resp:
            now = time.perf_counter()
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line or not line.startswith("data:"):
                continue
            event = line[5:].strip()
            if not event or event == "[DONE]":
                continue
            payload_obj = json.loads(event)
            payloads.append(payload_obj)
            if first_chunk_time is None and candidate_text_length(payload_obj) > 0:
                first_chunk_time = now
    elapsed = time.perf_counter() - t0

    final_payload = payloads[-1] if payloads else {}
    usage = final_payload.get("usageMetadata", {})
    output_tokens = int(usage.get("candidatesTokenCount", 0) or 0)
    if output_tokens <= 0:
        output_tokens = max(int(target_new_tokens), 1)

    ttft = (first_chunk_time - t0) if first_chunk_time is not None else elapsed
    decode_window = max(elapsed - ttft, 0.0)
    per_token = decode_window / max(output_tokens, 1)

    return {
        "ttft_sec": float(ttft),
        "elapsed_sec": float(elapsed),
        "output_tokens": float(output_tokens),
        "per_token_sec": float(per_token),
    }


def profile_service(model_name: str, prompts: list[dict[str, object]]) -> dict[str, float]:
    samples: list[dict[str, float]] = []
    target_tokens = int(prompts[0]["target_new_tokens"])
    for _ in range(DEFAULT_REPEATS):
        for entry in prompts:
            prompt_text = str(entry["prompt"])
            target_new_tokens = int(entry["target_new_tokens"])
            samples.append(generate_once(model_name, prompt_text, target_new_tokens))

    elapsed = [sample["elapsed_sec"] for sample in samples]
    ttft = [sample["ttft_sec"] for sample in samples]
    per_token = [sample["per_token_sec"] for sample in samples]

    return {
        "target_new_tokens": target_tokens,
        "ttft_sec_mean": float(mean(ttft)),
        "ttft_sec_std": float(pstdev(ttft)),
        "tpot_sec_mean": float(mean(per_token)),
        "tpot_sec_std": float(pstdev(per_token)),
        "latency_sec_mean": float(mean(elapsed)),
        "latency_sec_std": float(pstdev(elapsed)),
        "peak_vram_gb_mean": 0.0,
        "prompt_count": len(samples),
        "timing_mode": "streaming_ttft_plus_decode_per_token",
    }


def main() -> None:
    prompt_bank = load_prompt_bank()
    if PROFILE_PATH.exists():
        profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    else:
        profile = {"device": "api"}

    cloud_services = {
        service: profile_service(DEFAULT_MODEL, prompt_bank[service])
        for service in ("dialogue", "summary", "code")
    }
    profile["cloud"] = {
        "model_name": DEFAULT_MODEL,
        "role": "cloud",
        "services": cloud_services,
        "api_provider": "google_gemini",
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(json.dumps(profile["cloud"], indent=2))


if __name__ == "__main__":
    main()
