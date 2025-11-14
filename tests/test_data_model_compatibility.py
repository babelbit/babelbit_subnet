#!/usr/bin/env python3
"""
Test data model compatibility between miner and validator.

This test ensures that:
1. Miner's request/response models match validator's expectations
2. Serialization/deserialization works correctly
3. All required fields are present and typed correctly
"""
import pytest
from pydantic import ValidationError

# Import miner models
from babelbit.miner.serve_miner import (
    PredictRequest,
    PredictResponse,
    BBUtteranceEvaluation,
)

# Import validator models
from babelbit.chute_template.schemas import (
    BBPredictedUtterance,
    BBPredictOutput,
    BBUtteranceEvaluation as ValidatorEvaluation,
)


class TestRequestModelCompatibility:
    """Test that miner's PredictRequest is compatible with validator's BBPredictedUtterance."""
    
    def test_required_fields_present(self):
        """Test that all required fields from validator are in miner model."""
        # Required fields in validator's BBPredictedUtterance
        validator_required = {"index", "step", "prefix"}
        
        # Check miner model has these fields
        miner_fields = set(PredictRequest.model_fields.keys())
        
        missing = validator_required - miner_fields
        assert not missing, f"Miner PredictRequest missing required fields: {missing}"
    
    def test_optional_fields_compatible(self):
        """Test that optional fields are compatible between models."""
        # Optional fields that should be compatible
        optional_fields = {"context", "done", "ground_truth", "prediction", "evaluation"}
        
        miner_fields = set(PredictRequest.model_fields.keys())
        validator_fields = set(BBPredictedUtterance.model_fields.keys())
        
        # Check that miner has these fields
        for field in optional_fields:
            if field in validator_fields:
                assert field in miner_fields, f"Miner missing optional field: {field}"
    
    def test_field_types_match(self):
        """Test that field types are compatible."""
        # Create a validator model instance
        validator_data = {
            "index": "test-123",
            "step": 1,
            "prefix": "Hello",
            "context": "Test context",
            "done": False,
        }
        
        # Should work with validator model
        validator_obj = BBPredictedUtterance(**validator_data)
        assert validator_obj.index == "test-123"
        
        # Should work with miner model
        miner_obj = PredictRequest(**validator_data)
        assert miner_obj.index == "test-123"
        
        # Test serialization
        validator_dict = validator_obj.model_dump()
        miner_dict = miner_obj.model_dump()
        
        # Check key fields match
        for key in ["index", "step", "prefix", "context", "done"]:
            assert validator_dict[key] == miner_dict[key]
    
    def test_validator_payload_to_miner_request(self):
        """Test that validator's payload can be parsed by miner."""
        # This simulates what happens when validator sends request to miner
        validator_payload = BBPredictedUtterance(
            index="session-456",
            step=5,
            prefix="The quick brown",
            context="Previous dialogue: Hello world",
            done=False,
        )
        
        # Serialize as validator would send it
        payload_json = validator_payload.model_dump(mode="json")
        
        # Miner should be able to parse it
        miner_request = PredictRequest(**payload_json)
        
        assert miner_request.index == "session-456"
        assert miner_request.step == 5
        assert miner_request.prefix == "The quick brown"
        assert miner_request.context == "Previous dialogue: Hello world"
        assert miner_request.done is False
    
    def test_index_must_be_string(self):
        """Test that index field requires string type (not int)."""
        # This should fail - index must be string
        with pytest.raises(ValidationError) as exc_info:
            PredictRequest(index=123, step=1, prefix="test")
        
        assert "string_type" in str(exc_info.value)
    
    def test_evaluation_model_compatible(self):
        """Test that BBUtteranceEvaluation models are compatible."""
        eval_data = {
            "lexical_similarity": 0.85,
            "semantic_similarity": 0.92,
            "earliness": 0.75,
            "u_step": 0.80,
        }
        
        # Both should accept the same data
        validator_eval = ValidatorEvaluation(**eval_data)
        miner_eval = BBUtteranceEvaluation(**eval_data)
        
        assert validator_eval.lexical_similarity == miner_eval.lexical_similarity
        assert validator_eval.semantic_similarity == miner_eval.semantic_similarity


class TestResponseModelCompatibility:
    """Test that miner's PredictResponse matches what validator expects."""
    
    def test_simple_prediction_format(self):
        """Test that miner returns simple {"prediction": "..."} format."""
        # Miner response
        miner_response = PredictResponse(prediction="Hello, my name is John")
        
        # Serialize it
        response_dict = miner_response.model_dump()
        
        # Should have exactly the structure validator expects
        assert "prediction" in response_dict
        assert isinstance(response_dict["prediction"], str)
        assert response_dict["prediction"] == "Hello, my name is John"
    
    def test_response_serialization(self):
        """Test that response can be serialized to JSON correctly."""
        response = PredictResponse(prediction="Test output")
        
        # Should serialize without errors
        json_data = response.model_dump(mode="json")
        
        assert json_data == {"prediction": "Test output"}
    
    def test_empty_prediction_allowed(self):
        """Test that empty predictions are valid."""
        response = PredictResponse(prediction="")
        
        assert response.prediction == ""
        assert response.model_dump() == {"prediction": ""}
    
    def test_validator_can_parse_response(self):
        """Test that validator's parsing logic can handle miner's response."""
        # Miner sends this
        miner_response = PredictResponse(prediction="The quick brown fox")
        response_dict = miner_response.model_dump()
        
        # Validator expects to extract prediction like this (from predict_engine.py line 151)
        prediction = response_dict.get("prediction", "")
        
        assert prediction == "The quick brown fox"


class TestEndToEndCompatibility:
    """Test complete request-response cycle between validator and miner."""
    
    def test_full_cycle_serialization(self):
        """Test that a full request-response cycle works."""
        # 1. Validator creates request payload
        validator_payload = BBPredictedUtterance(
            index="test-session",
            step=3,
            prefix="Hello world",
            context="This is context",
            done=False,
        )
        
        # 2. Validator serializes and sends to miner
        request_json = validator_payload.model_dump(mode="json")
        
        # 3. Miner receives and parses request
        miner_request = PredictRequest(**request_json)
        assert miner_request.prefix == "Hello world"
        
        # 4. Miner generates response
        miner_response = PredictResponse(
            prediction="Hello world and goodbye"
        )
        
        # 5. Miner serializes response
        response_json = miner_response.model_dump(mode="json")
        
        # 6. Validator receives and parses response
        prediction = response_json.get("prediction", "")
        
        # 7. Validator creates BBPredictOutput
        output = BBPredictOutput(
            success=True,
            model="axon",
            utterance=BBPredictedUtterance(
                index=validator_payload.index,
                step=validator_payload.step,
                prefix=validator_payload.prefix,
                prediction=prediction,
                context=validator_payload.context,
            ),
            error=None,
            context_used=validator_payload.context,
            complete=True,
        )
        
        assert output.success is True
        assert output.utterance.prediction == "Hello world and goodbye"
    
    def test_all_optional_fields_preserved(self):
        """Test that all optional fields are preserved through the cycle."""
        # Validator sends request with all optional fields
        full_payload = BBPredictedUtterance(
            index="full-test",
            step=10,
            prefix="Complete test",
            prediction="Previous prediction",
            context="Full context here",
            done=False,
            ground_truth="Expected output",
            evaluation=ValidatorEvaluation(
                lexical_similarity=0.9,
                semantic_similarity=0.85,
                earliness=0.7,
                u_step=0.8,
            ),
        )
        
        # Convert to miner format
        request_data = full_payload.model_dump()
        miner_request = PredictRequest(**request_data)
        
        # All fields should be preserved
        assert miner_request.index == "full-test"
        assert miner_request.step == 10
        assert miner_request.prefix == "Complete test"
        assert miner_request.context == "Full context here"
        assert miner_request.done is False
        assert miner_request.ground_truth == "Expected output"
        assert miner_request.evaluation is not None
        assert miner_request.evaluation.lexical_similarity == 0.9


class TestErrorCases:
    """Test error handling and edge cases."""
    
    def test_missing_required_field_index(self):
        """Test that missing 'index' field raises error."""
        with pytest.raises(ValidationError):
            PredictRequest(step=1, prefix="test")
    
    def test_missing_required_field_step(self):
        """Test that missing 'step' field raises error."""
        with pytest.raises(ValidationError):
            PredictRequest(index="test", prefix="test")
    
    def test_missing_required_field_prefix(self):
        """Test that missing 'prefix' field raises error."""
        with pytest.raises(ValidationError):
            PredictRequest(index="test", step=1)
    
    def test_wrong_type_for_step(self):
        """Test that wrong type for 'step' raises error."""
        with pytest.raises(ValidationError):
            PredictRequest(index="test", step="not-a-number", prefix="test")
    
    def test_context_defaults_to_empty_string(self):
        """Test that context has proper default value."""
        request = PredictRequest(index="test", step=1, prefix="hello")
        
        assert request.context == ""
    
    def test_done_defaults_to_false(self):
        """Test that done has proper default value."""
        request = PredictRequest(index="test", step=1, prefix="hello")
        
        assert request.done is False
    
    def test_prediction_empty_string_default(self):
        """Test that prediction defaults to empty string."""
        request = PredictRequest(index="test", step=1, prefix="hello")
        
        assert request.prediction == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
