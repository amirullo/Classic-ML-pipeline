import pytest
import time
from unittest.mock import patch, MagicMock
from orchestrator import PipelineOrchestrator
from schemas import PredictionRequest
from fastapi.testclient import TestClient
from main import app
import threading
# import queue

@pytest.fixture
def client():
    """Create a test client for FastAPI"""
    return TestClient(app)


@pytest.fixture
def test_orchestrator():
    """Create and start a test orchestrator"""
    orchestrator = PipelineOrchestrator()
    orchestrator.start()
    yield orchestrator
    orchestrator.stop()


def test_score_reset_during_enqueue():
    """Test if the score is reset when enqueuing a new request"""
    # Create orchestrator and manually set a score
    orchestrator = PipelineOrchestrator()
    orchestrator.predict_stage.score = 0.95  # Set initial score

    # Log the score before enqueueing
    before_score = orchestrator.predict_stage.score

    # Enqueue a new request
    request = PredictionRequest(output_path='test_output.csv')
    orchestrator.enqueue(request)

    # Check if score was reset immediately after enqueueing
    after_score = orchestrator.predict_stage.score

    # Score should not change just from enqueuing
    assert before_score == after_score
    assert 0.95 == after_score

    # Clean up
    orchestrator.stop()


def test_predict_stage_processing_resets_score():
    """Test if PredictStage resets the score when processing starts"""
    # Create a mock PredictStage with instrumentation
    with patch('stages.predict_stage.PredictStage') as MockPredictStage:
        # Setup the mock to track when process_item is called
        mock_instance = MagicMock()
        MockPredictStage.return_value = mock_instance
        mock_instance.score = 0.95

        # Create orchestrator with our mock
        orchestrator = PipelineOrchestrator()

        # Start the pipeline
        orchestrator.start()

        # Enqueue a request
        request = PredictionRequest(output_path='test_output.csv')
        orchestrator.enqueue(request)

        # Wait a moment for processing to begin
        time.sleep(0.1)

        # Check if process_item was called
        mock_instance.process_item.assert_called()

        # Check if score was reset during processing
        assert 0.95 == mock_instance.score

        # Clean up
        orchestrator.stop()


def test_pipeline_reinitializes_between_api_calls(client):
    """Test if the orchestrator is re-initialized between API calls"""
    # Create a spy to track orchestrator creation
    with patch('main.PipelineOrchestrator') as MockOrchestrator:
        # Setup our mock orchestrator
        mock_instance = MagicMock()
        MockOrchestrator.return_value = mock_instance
        mock_instance.predict_stage.score = 0.95

        # Make the first API call
        response1 = client.get("/predict")
        assert response1.status_code == 200

        # Make a call to get the score
        response2 = client.get("/score")
        assert response2.status_code == 200
        assert response2.json()["score"] == 0.95

        # Check if orchestrator was initialized more than once
        assert MockOrchestrator.call_count == 1


def test_api_state_persistence(client):
    """Test if state is persisted between API calls"""
    # Make the first prediction
    client.get("/predict")

    # Manually set the score in the global orchestrator
    from main import orchestrator
    orchestrator.predict_stage.score = 0.95

    # Get the score via API
    response = client.get("/score")

    # Check if we get the expected score
    assert response.json()["score"] == 0.95

    # Make another prediction request
    client.get("/predict")

    # Immediately check score again - should still be 0.95
    response = client.get("/score")
    assert response.json()["score"] == 0.95


def test_score_update_timing(test_orchestrator):
    """Test the timing of score updates in the pipeline"""
    # Set initial score for testing
    test_orchestrator.predict_stage.score = 0.95

    # Add logging to track score changes
    original_process = test_orchestrator.predict_stage.process_item
    score_values = []

    def process_with_logging(item):
        # Log score before processing
        score_values.append(("before", test_orchestrator.predict_stage.score))
        result = original_process(item)
        # Log score after processing
        score_values.append(("after", test_orchestrator.predict_stage.score))
        return result

    test_orchestrator.predict_stage.process_item = process_with_logging

    # Enqueue a request
    request = PredictionRequest(output_path='test_output.csv')
    test_orchestrator.enqueue(request)

    # Allow some time for processing
    time.sleep(1)

    # Check logged scores
    print(f"Score values during test: {score_values}")

    # We expect the score to be 0.95 before processing
    assert score_values[0][1] == 0.95


def test_predict_stage_initialization():
    """Test the initialization of the PredictStage to see if score has a default value"""
    with patch('stages.predict_stage.PredictStage.__init__', return_value=None) as mock_init:
        # Create a new orchestrator, which will create a new PredictStage
        orchestrator = PipelineOrchestrator()

        # Check if PredictStage.__init__ was called
        mock_init.assert_called_once()

        # Clean up
        orchestrator.stop()

    # Test the actual initialization of PredictStage
    from stages.predict_stage import PredictStage
    import queue

    predict_stage = PredictStage(queue.Queue(), threading.Event())

    # Check the default value of score
    assert hasattr(predict_stage, 'score'), "PredictStage missing 'score' attribute"
    print(f"Default score value: {predict_stage.score}")