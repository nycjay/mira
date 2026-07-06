"""Tests for the model registry including MiniMax-2.7."""

from __future__ import annotations

import pytest

from mira.llm import registry


class TestMiniMaxInRegistry:
    """MiniMax-M2.7 must be registered and usable for both indexing and review."""

    def test_minimax_in_all_models(self):
        assert "minimax/MiniMax-M2.7" in registry.all_models()

    def test_minimax_entry_structure(self):
        info = registry.get("minimax/MiniMax-M2.7")
        assert info["label"] == "MiniMax M2.7"
        assert info["provider"] == "minimax"
        assert info["max_input_tokens"] == 1000000
        assert info["max_output_tokens"] == 131072
        assert info["supports_json_mode"] is True

    def test_minimax_supports_indexing(self):
        assert registry.is_supported("minimax/MiniMax-M2.7", purpose="indexing")

    def test_minimax_supports_review(self):
        assert registry.is_supported("minimax/MiniMax-M2.7", purpose="review")

    def test_minimax_in_indexing_models_list(self):
        indexing = registry.models_for_purpose("indexing")
        values = [m["value"] for m in indexing]
        assert "minimax/MiniMax-M2.7" in values

    def test_minimax_in_review_models_list(self):
        review = registry.models_for_purpose("review")
        values = [m["value"] for m in review]
        assert "minimax/MiniMax-M2.7" in values

    def test_minimax_pricing(self):
        inp, out = registry.pricing("minimax/MiniMax-M2.7")
        assert inp == 0.30
        assert out == 2.50

    def test_minimax_max_output_tokens(self):
        assert registry.max_output_tokens("minimax/MiniMax-M2.7") == 131072


class TestRegistryRegression:
    """Existing models must not be affected by the MiniMax addition."""

    def test_existing_models_still_present(self):
        assert "anthropic/claude-sonnet-4-6" in registry.all_models()
        assert "google/gemini-2.5-flash" in registry.all_models()

    def test_existing_indexing_models_still_work(self):
        indexing = registry.models_for_purpose("indexing")
        values = [m["value"] for m in indexing]
        assert "google/gemini-2.5-flash" in values

    def test_existing_review_models_still_work(self):
        review = registry.models_for_purpose("review")
        values = [m["value"] for m in review]
        assert "anthropic/claude-sonnet-4-6" in values


class TestCurrentGenerationModels:
    """GPT-5.x / Gemini 3.x additions (issue #125) — ids and pricing verified
    against the live OpenRouter catalog."""

    @pytest.mark.parametrize(
        ("model_id", "pricing", "purposes"),
        [
            ("openai/gpt-4.1-mini", (0.40, 1.60), ["indexing"]),
            ("openai/gpt-5-nano", (0.05, 0.40), ["indexing"]),
            ("openai/gpt-5-mini", (0.25, 2.00), ["indexing", "review"]),
            ("openai/gpt-5.1-codex-mini", (0.25, 2.00), ["indexing", "review"]),
            ("openai/gpt-5.1-codex", (1.25, 10.00), ["review"]),
            ("google/gemini-3-flash-preview", (0.50, 3.00), ["indexing", "review"]),
            ("google/gemini-3.1-flash-lite", (0.25, 1.50), ["indexing"]),
        ],
    )
    def test_registered_with_pricing_and_purposes(self, model_id, pricing, purposes):
        assert registry.pricing(model_id) == pricing
        for purpose in purposes:
            assert registry.is_supported(model_id, purpose=purpose)
            assert model_id in [m["value"] for m in registry.models_for_purpose(purpose)]

    def test_recommended_defaults_unchanged(self):
        indexing = registry.models_for_purpose("indexing")
        review = registry.models_for_purpose("review")
        assert [m["value"] for m in indexing if m["recommended"]] == [
            "anthropic/claude-haiku-4-5",
            "us.anthropic.claude-haiku-4-5-v1:0",
        ]
        assert [m["value"] for m in review if m["recommended"]] == [
            "anthropic/claude-sonnet-4-6",
            "us.anthropic.claude-sonnet-4-6-v1:0",
        ]
