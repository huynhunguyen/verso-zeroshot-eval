"""End-to-end smoke tests verifying dataset integrity, prompt hashing, provider params, and grading."""

from pathlib import Path

from verso.direct import extract, to_result
from verso.llm import FakeProvider, LLMClient, LLMConfig, build_params, config_from_profile, load_profile
from verso.llm.types import ChatRequest
from verso_eval import gold, grader, paths
from verso_eval.baselines import mllm_zero_shot as zs


def test_dataset_complete():
    eids = gold.evaluation_ids()
    items = gold.dataset_items()
    strata = gold.circuit_strata()
    assert len(eids) == 1765
    assert len(items) == 1765
    assert len(strata) == 1765
    imgs = {items[qid]["id_image"] for qid in eids}
    assert len(imgs) == 245
    missing = [img for img in imgs if not (paths.IMAGES_DIR / img).exists()]
    assert not missing, f"Missing diagram images: {missing[:5]}"


def test_default_prompt_and_extraction():
    version, template = zs.load_prompt()
    assert version.startswith("mllm_zero_shot_v1@")
    assert "{question}" in template
    ex = extract('Step 1...\n```json\n{"status": "ok", "value": 2.5, "unit": "mA"}\n```')
    assert ex.method == "json"
    res = to_result(ex.answer)
    assert abs(res.value - 0.0025) < 1e-12


def test_commercial_profiles_valid():
    for name in ("gemini-3-pro", "gemini-3-flash", "openai-gpt-4o", "openai-gpt-5", "openai-o3"):
        profile = load_profile(name, paths.MODEL_PROFILES_DIR)
        cfg = config_from_profile(profile, decoding="recommended", seed=0)
        req = ChatRequest(
            model=cfg.model,
            messages=[{"role": "user", "content": "hi"}],
            temperature=cfg.temperature,
            seed=cfg.seed,
            max_tokens=cfg.max_tokens,
            max_tokens_param=cfg.max_tokens_param,
            send_seed=cfg.send_seed,
            extra=cfg.extra,
        )
        params = build_params(req, cfg.structured_output)
        assert params["model"] == profile.served_name
        if name in ("openai-gpt-5", "openai-o3"):
            assert "max_completion_tokens" in params
            assert "max_tokens" not in params
            assert "temperature" not in params
            assert "seed" not in params
        if name.startswith("gemini"):
            assert "seed" not in params
            assert "max_tokens" in params


def test_fake_provider_end_to_end(tmp_path: Path):
    items, image_of, strata, grade_fn = zs.load_set("eval")
    subset = {qid: items[qid] for qid in sorted(items)[:5]}
    prompt = zs.load_prompt()
    provider = FakeProvider(lambda req: '```json\n{"status": "ok", "value": 1, "unit": "A"}\n```')
    cfg = LLMConfig(model="fake-model", temperature=0.0, max_tokens=256, cache_path=None)
    client = LLMClient(provider, cache=None, config=cfg)
    records = zs.run_questions(subset, image_of, client, cfg, prompt, workers=2)
    assert len(records) == 5
    assert provider.calls == 5
    report = zs.grade_records(records, subset, grade=grade_fn)
    summary = zs.summarize(records, report, strata)
    assert summary["questions"] == 5
    assert summary["call_errors"] == 0
    assert summary["extraction"] == {"json": 5}
