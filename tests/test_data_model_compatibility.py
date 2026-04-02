#!/usr/bin/env python3
"""
Test data model compatibility for validator prediction payloads.

This test ensures that:
1. Validator request/response models serialize cleanly
2. Provider payloads can be parsed into the validator response shape
3. Required fields are present and typed correctly
"""
import pytest
from pydantic import ValidationError

from babelbit.schemas.prediction import (
    BBPredictedUtterance,
    BBPredictOutput,
    BBUtteranceEvaluation,
)
from babelbit.utils.predict_engine import _extract_prediction_from_provider_payload


class TestRequestModelCompatibility:
    """Test validator request model shape and round-trip behavior."""

    def test_required_fields_present(self):
        required_fields = {"index", "step", "prefix"}
        model_fields = set(BBPredictedUtterance.model_fields.keys())

        missing = required_fields - model_fields
        assert not missing, f"BBPredictedUtterance missing required fields: {missing}"

    def test_optional_fields_present(self):
        optional_fields = {"context", "done", "ground_truth", "prediction", "evaluation"}
        model_fields = set(BBPredictedUtterance.model_fields.keys())

        missing = optional_fields - model_fields
        assert not missing, f"BBPredictedUtterance missing optional fields: {missing}"

    def test_field_types_roundtrip(self):
        payload = {
            "index": "test-123",
            "step": 1,
            "prefix": "Hello",
            "context": "Test context",
            "done": False,
        }

        model = BBPredictedUtterance(**payload)
        assert model.index == "test-123"

        dumped = model.model_dump()
        for key in ["index", "step", "prefix", "context", "done"]:
            assert dumped[key] == payload[key]

    def test_request_payload_roundtrip(self):
        payload = BBPredictedUtterance(
            index="session-456",
            step=5,
            prefix="The quick brown",
            context="Previous dialogue: Hello world",
            done=False,
        )

        payload_json = payload.model_dump(mode="json")
        roundtrip = BBPredictedUtterance(**payload_json)

        assert roundtrip.index == "session-456"
        assert roundtrip.step == 5
        assert roundtrip.prefix == "The quick brown"
        assert roundtrip.context == "Previous dialogue: Hello world"
        assert roundtrip.done is False

    def test_index_must_be_string(self):
        with pytest.raises(ValidationError) as exc_info:
            BBPredictedUtterance(index=123, step=1, prefix="test")

        assert "string_type" in str(exc_info.value)

    def test_evaluation_model_compatible(self):
        eval_data = {
            "lexical_similarity": 0.85,
            "semantic_similarity": 0.92,
            "earliness": 0.75,
            "u_step": 0.80,
        }

        validator_eval = BBUtteranceEvaluation(**eval_data)
        assert validator_eval.lexical_similarity == eval_data["lexical_similarity"]
        assert validator_eval.semantic_similarity == eval_data["semantic_similarity"]


class TestResponseModelCompatibility:
    """Test validator output model and provider payload parsing."""

    def test_simple_prediction_format(self):
        response = BBPredictOutput(
            success=True,
            model="axon",
            utterance=BBPredictedUtterance(
                index="idx-1",
                step=0,
                prefix="Hello",
                prediction="Hello, my name is John",
            ),
            error=None,
            context_used="",
            complete=True,
        )

        response_dict = response.model_dump()
        assert response_dict["utterance"]["prediction"] == "Hello, my name is John"

    def test_response_serialization(self):
        response = BBPredictOutput(
            success=True,
            model="axon",
            utterance=BBPredictedUtterance(
                index="idx-1",
                step=0,
                prefix="Hello",
                prediction="Test output",
            ),
            error=None,
            context_used="",
            complete=False,
        )

        json_data = response.model_dump(mode="json")
        assert json_data["utterance"]["prediction"] == "Test output"

    def test_empty_prediction_allowed(self):
        response = BBPredictedUtterance(index="idx-1", step=0, prefix="hello")

        assert response.prediction == ""
        assert response.model_dump() == {
            "index": "idx-1",
            "step": 0,
            "prefix": "hello",
            "prediction": "",
            "context": "",
            "done": False,
            "ground_truth": None,
            "evaluation": None,
        }

    def test_provider_payload_extraction(self):
        assert _extract_prediction_from_provider_payload({"prediction": "The quick brown fox"}) == "The quick brown fox"
        assert _extract_prediction_from_provider_payload({"output": "The quick brown fox"}) == "The quick brown fox"
        assert _extract_prediction_from_provider_payload({"output": {"prediction": "The quick brown fox"}}) == "The quick brown fox"


class TestEndToEndCompatibility:
    """Test complete validator prediction model cycle."""

    def test_full_cycle_serialization(self):
        validator_payload = BBPredictedUtterance(
            index="test-session",
            step=3,
            prefix="Hello world",
            context="This is context",
            done=False,
        )

        request_json = validator_payload.model_dump(mode="json")
        roundtrip_payload = BBPredictedUtterance(**request_json)

        prediction = "Hello world and goodbye"
        output = BBPredictOutput(
            success=True,
            model="axon",
            utterance=BBPredictedUtterance(
                index=roundtrip_payload.index,
                step=roundtrip_payload.step,
                prefix=roundtrip_payload.prefix,
                prediction=prediction,
                context=roundtrip_payload.context,
            ),
            error=None,
            context_used=roundtrip_payload.context,
            complete=True,
        )

        assert output.success is True
        assert output.utterance.prediction == "Hello world and goodbye"

    def test_all_optional_fields_preserved(self):
        full_payload = BBPredictedUtterance(
            index="full-test",
            step=10,
            prefix="Complete test",
            prediction="Previous prediction",
            context="Full context here",
            done=False,
            ground_truth="Expected output",
            evaluation=BBUtteranceEvaluation(
                lexical_similarity=0.9,
                semantic_similarity=0.85,
                earliness=0.7,
                u_step=0.8,
            ),
        )

        request_data = full_payload.model_dump()
        roundtrip = BBPredictedUtterance(**request_data)

        assert roundtrip.index == "full-test"
        assert roundtrip.step == 10
        assert roundtrip.prefix == "Complete test"
        assert roundtrip.context == "Full context here"
        assert roundtrip.done is False
        assert roundtrip.ground_truth == "Expected output"
        assert roundtrip.evaluation is not None
        assert roundtrip.evaluation.lexical_similarity == 0.9


class TestErrorCases:
    """Test error handling and edge cases."""

    def test_missing_required_field_index(self):
        with pytest.raises(ValidationError):
            BBPredictedUtterance(step=1, prefix="test")

    def test_missing_required_field_step(self):
        with pytest.raises(ValidationError):
            BBPredictedUtterance(index="test", prefix="test")

    def test_missing_required_field_prefix(self):
        with pytest.raises(ValidationError):
            BBPredictedUtterance(index="test", step=1)

    def test_wrong_type_for_step(self):
        with pytest.raises(ValidationError):
            BBPredictedUtterance(index="test", step="not-a-number", prefix="test")

    def test_context_defaults_to_empty_string(self):
        request = BBPredictedUtterance(index="test", step=1, prefix="hello")

        assert request.context == ""

    def test_done_defaults_to_false(self):
        request = BBPredictedUtterance(index="test", step=1, prefix="hello")

        assert request.done is False

    def test_prediction_empty_string_default(self):
        request = BBPredictedUtterance(index="test", step=1, prefix="hello")

        assert request.prediction == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
