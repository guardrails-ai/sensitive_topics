import json
import pytest
from unittest.mock import patch

from guardrails.validator_base import FailResult, PassResult
from validator import SensitiveTopic

DEVICE = -1
MODEL = "facebook/bart-large-mnli"
LLM_CALLABLE = "gpt-3.5-turbo"


class TestSensitiveTopic:
    def test_init_with_valid_args(self):
        validator = SensitiveTopic(
            device=DEVICE,
            model=MODEL,
            llm_callable=LLM_CALLABLE,
            on_fail=None,
        )
        assert validator._device == DEVICE
        assert validator._model == MODEL
        assert validator._llm_callable.__name__ == "openai_callable"

    def test_init_with_invalid_llm_callable(self):
        with pytest.raises(ValueError):
            SensitiveTopic(
                llm_callable="invalid_model",
            )

    def test_get_topics_ensemble(self):
        text = "This is an article about sports."
        candidate_topics = ["sports", "politics", "technology"]
        validator = SensitiveTopic()

        with patch.object(validator, "get_topic_zero_shot") as mock_zero_shot:
            mock_zero_shot.return_value = ("sports", 0.6)

            topics = validator.get_topics_ensemble(text, candidate_topics)
            assert topics == ["sports", "sports", "sports"]

    def test_get_topics_llm(self):
        text = "This is an article about politics."
        candidate_topics = ["sports", "politics", "technology"]
        validator = SensitiveTopic()

        with patch.object(validator, "call_llm") as mock_llm:
            mock_llm.return_value = '{"topic": "politics"}'

            validation_result = validator.get_topics_llm(text, candidate_topics)
            assert validation_result == ["politics", "politics", "politics"]

    def test_set_callable_string(self):
        validator = SensitiveTopic()
        validator.set_callable("gpt-3.5-turbo")
        assert validator._llm_callable.__name__ == "openai_callable"

    def test_set_callable_callable(self):
        def custom_callable(text, topics):
            return json.dumps({"topic": topics[0]})

        validator = SensitiveTopic()
        validator.set_callable(custom_callable)
        assert validator._llm_callable.__name__ == "custom_callable"

    def test_get_topics_zero_shot(self):
        text = "This is an article about technology."
        candidate_topics = ["sports", "politics", "technology"]
        validator = SensitiveTopic(sensitive_topics=candidate_topics)

        topics = validator.get_topics_zero_shot(text, candidate_topics)
        assert ["other", "other", "technology"] == topics

        with patch.object(validator, "get_topic_zero_shot") as mock_zero_shot:
            mock_zero_shot.return_value = ("technology", 0.6)
            topics = validator.get_topics_zero_shot(text, candidate_topics)
            assert ["technology", "technology", "technology"] == topics

    def test_validate_message_without_sensitive_topic(self):
        text = "This is an article about sports."
        validator = SensitiveTopic(
            sensitive_topics=["violence"],
        )
        validation_result = validator.validate(text, metadata={})
        assert validation_result == PassResult()

    def test_validate_message_with_sensitive_topic(self):
        text = "This is an article about sports."
        validator = SensitiveTopic(
            sensitive_topics=["sports"],
        )
        validation_result = validator.validate(text, metadata={})
        assert validation_result == FailResult(
            error_message="Sensitive topics detected: sports",
            fix_value="Trigger warning:\n- sports\n\n"
            "This is an article about sports.",
        )

    def test_validate_invalid_topic(self):
        validator = SensitiveTopic()
        with patch.object(validator, "call_llm") as mock_llm:
            mock_llm.return_value = '{"topic": "other"}'

            text = "This is an article about music."
            validation_result = validator.validate(text, metadata={})

            assert validation_result == PassResult()
