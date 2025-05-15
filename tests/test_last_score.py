import unittest
import threading
import time
import datetime as dt
from unittest.mock import MagicMock, patch
import queue
import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Mock classes based on actual implementation
class PredictionRequest:
    def __init__(self, output_path):
        self.output_path = output_path
        self.data_path = "sample_data.csv"

class MLModel:
    def predict(self, df):
        # Simulate successful prediction
        return pd.DataFrame({"prediction": [1, 0, 1]}), 0.85
        
class MockPredictStage(threading.Thread):
    def __init__(self, input_queue, stop_event=None):
        super().__init__()
        self.input_queue = input_queue
        self.stop_event = stop_event if stop_event else threading.Event()
        self.score = 0.00  # Matches actual implementation
        self.last_dt = None
        self.name = "PredictStage"
    
    def run(self):
        """Simulates the actual run method of PredictStage"""
        while not self.stop_event.is_set():
            try:
                df, request = self.input_queue.get(timeout=1)
                # Simulate prediction
                predictions, self.score = MLModel().predict(df)
                self.last_dt = dt.datetime.now()
                # Simulate saving to CSV
                # predictions.to_csv(request.output_path, index=False)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Smth went wrong: {e}")
                continue

class MockDataStage(threading.Thread):
    def __init__(self, input_queue, output_queue, stop_event):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.name = "DataStage"
    
    def run(self):
        while not self.stop_event.is_set():
            try:
                request = self.input_queue.get(timeout=1)
                # Generate some mock data
                df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
                self.output_queue.put((df, request))
            except queue.Empty:
                # This simulates no data available
                continue
            except Exception as e:
                # Silently continue on any other exception
                continue

class MockFeatureStage(threading.Thread):
    def __init__(self, input_queue, output_queue, stop_event):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.name = "FeatureStage"
        
    def run(self):
        while not self.stop_event.is_set():
            try:
                df, request = self.input_queue.get(timeout=1)
                # Process data
                # For testing, just pass through
                self.output_queue.put((df, request))
            except queue.Empty:
                continue
            except Exception as e:
                continue

class MockOrchestrator:
    def __init__(self):
        self.raw_data_queue = queue.Queue()
        self.features_queue = queue.Queue()
        self.prediction_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        self.data_stage = MockDataStage(self.raw_data_queue, self.features_queue, self.stop_event)
        self.feature_stage = MockFeatureStage(self.features_queue, self.prediction_queue, self.stop_event)
        self.predict_stage = MockPredictStage(self.prediction_queue, self.stop_event)
        
        self.stages = [self.data_stage, self.feature_stage, self.predict_stage]
    
    def start(self):
        """Start all the pipeline stages"""
        for stage in self.stages:
            stage.start()
    
    def enqueue(self, request):
        """Add a request to the pipeline"""
        self.raw_data_queue.put(request)
    
    def stop(self):
        """Stop the pipeline"""
        self.stop_event.set()
        for stage in self.stages:
            stage.join(timeout=1.0)


class TestGetLastScore(unittest.TestCase):
    
    def setUp(self):
        # Create a FastAPI test client
        self.app = FastAPI()
        self.orchestrator = MockOrchestrator()
        
        # Define the FastAPI routes for testing
        @self.app.get("/predict")
        def predict_route():
            request = PredictionRequest(output_path='some_predict.csv')
            self.orchestrator.enqueue(request)
            return {"status": "Prediction enqueued", "datetime": dt.datetime.now()}
        
        @self.app.get("/score")
        def get_last_score():
            score = self.orchestrator.predict_stage.score
            last_dt = self.orchestrator.predict_stage.last_dt
            return {"score": score, "last_dt": last_dt}
        
        self.client = TestClient(self.app)
        self.orchestrator.start()
    
    def tearDown(self):
        self.orchestrator.stop()
    
    def test_initial_score(self):
        """Test that initial score is 0.00 as set in constructor"""
        response = self.client.get("/score")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["score"], 0.0)
    
    def test_pipeline_flow(self):
        """Test if data flows correctly through the pipeline stages"""
        # Mock the MLModel to verify it was called
        original_predict = MLModel.predict
        call_count = [0]  # Use list for mutable counter
        
        def mock_predict(self, df):
            call_count[0] += 1
            return original_predict(self, df)
        
        # Apply the mock
        MLModel.predict = mock_predict
        
        try:
            # Call predict endpoint
            self.client.get("/predict")
            
            # Wait for data to flow through pipeline
            # This might need adjusting based on your actual pipeline speed
            timeout = 10  # seconds
            start_time = time.time()
            while call_count[0] == 0 and time.time() - start_time < timeout:
                time.sleep(0.1)
            
            # Check if prediction was made
            self.assertTrue(call_count[0] > 0, "MLModel.predict() was never called")
            
            # Check the score after prediction
            response = self.client.get("/score")
            self.assertNotEqual(response.json()["score"], 0.0)
            
        finally:
            # Restore original function
            MLModel.predict = original_predict
    
    def test_exception_in_pipeline(self):
        """Test what happens when an exception occurs in the pipeline"""
        # Mock the data stage to raise an exception
        original_run = MockDataStage.run
        
        def mock_run(self):
            while not self.stop_event.is_set():
                try:
                    # Simulate an exception processing data
                    request = self.input_queue.get(timeout=1)
                    raise ValueError("Simulated data processing error")
                except queue.Empty:
                    continue
                except Exception as e:
                    # Log but continue (similar to your actual code)
                    print(f"Error in DataStage: {e}")
                    continue
        
        # Apply the mock
        MockDataStage.run = mock_run
        
        try:
            # Call predict endpoint
            self.client.get("/predict")
            
            # Wait a bit
            time.sleep(0.5)
            
            # Check the score - should still be 0.0 because of the exception
            response = self.client.get("/score")
            self.assertEqual(response.json()["score"], 0.0)
            
        finally:
            # Restore original function
            MockDataStage.run = original_run
    
    def test_queue_issues(self):
        """Test what happens when there are issues with the queue flow"""
        # Modify the feature stage to not forward data to prediction stage
        original_run = MockFeatureStage.run
        
        def mock_run(self):
            while not self.stop_event.is_set():
                try:
                    # Get data but don't forward it (simulates broken pipeline)
                    df, request = self.input_queue.get(timeout=1)
                    # Don't put anything in output queue
                    pass
                except queue.Empty:
                    continue
                except Exception as e:
                    continue
        
        # Apply the mock
        MockFeatureStage.run = mock_run
        
        try:
            # Call predict endpoint
            self.client.get("/predict")
            
            # Wait a bit
            time.sleep(0.5)
            
            # Check the score - should still be 0.0 because data never reached prediction stage
            response = self.client.get("/score")
            self.assertEqual(response.json()["score"], 0.0)
            
        finally:
            # Restore original function
            MockFeatureStage.run = original_run
    
    def test_race_condition(self):
        """Test for race condition where score is checked too quickly after prediction request"""
        # Call predict endpoint
        predict_response = self.client.get("/predict")
        
        # Immediately get score (race condition)
        immediate_response = self.client.get("/score")
        immediate_score = immediate_response.json()["score"]
        
        # Wait for processing to complete
        time.sleep(2.0)
        
        # Get score again
        later_response = self.client.get("/score")
        later_score = later_response.json()["score"]
        
        print(f"Immediate score: {immediate_score}, Later score: {later_score}")
        
        # If immediate_score is 0.0 but later_score is not, we have identified a race condition
        if immediate_score == 0.0 and later_score != 0.0:
            print("✓ Race condition confirmed: score checked before prediction completed")


if __name__ == "__main__":
    unittest.main()
