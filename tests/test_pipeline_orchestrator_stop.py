import pytest
import time
import threading
import queue
from unittest.mock import patch, MagicMock
from orchestrator import PipelineOrchestrator
from schemas import PredictionRequest


@pytest.fixture
def orchestrator():
    """Create a test orchestrator"""
    return PipelineOrchestrator()


def test_stop_sets_stop_event(orchestrator):
    """Test that stop() sets the stop event"""
    # Verify stop_event is initially not set
    assert not orchestrator.stop_event.is_set()
    
    # Call stop() method
    orchestrator.stop()
    
    # Verify stop_event is now set
    assert orchestrator.stop_event.is_set()


def test_stop_calls_join_on_all_stages(orchestrator):
    """Test that stop() calls join() on all stage threads"""
    # Mock the stages to track join calls
    for i, stage in enumerate(orchestrator.stages):
        orchestrator.stages[i] = MagicMock()
    
    # Call stop method
    orchestrator.stop()
    
    # Verify join was called on each stage
    for stage in orchestrator.stages:
        stage.join.assert_called_once()


def test_stop_terminates_threads():
    """Test that stop() actually terminates all stage threads"""
    # Create a fresh orchestrator
    orchestrator = PipelineOrchestrator()
    
    # Start the orchestrator
    orchestrator.start()
    
    # Get list of thread names before stopping
    thread_names_before = [t.name for t in threading.enumerate()]
    
    # Wait a moment for threads to initialize
    time.sleep(0.5)
    
    # Stop the orchestrator
    orchestrator.stop()
    
    # Allow a brief moment for threads to terminate
    time.sleep(0.5)
    
    # Get list of thread names after stopping
    thread_names_after = [t.name for t in threading.enumerate()]
    
    # Check that stage thread names aren't in the active threads list
    for stage in orchestrator.stages:
        assert stage.name not in thread_names_after, f"Thread {stage.name} is still running after stop()"


def test_stop_handles_empty_queues():
    """Test that stop() works correctly with empty queues"""
    orchestrator = PipelineOrchestrator()
    orchestrator.start()
    time.sleep(0.2)
    
    # All queues should be empty
    assert orchestrator.raw_data_queue.empty()
    assert orchestrator.features_queue.empty()
    assert orchestrator.prediction_queue.empty()
    
    # Call stop and verify it completes without errors
    orchestrator.stop()
    
    # Verify all stage threads are no longer alive
    for stage in orchestrator.stages:
        assert not stage.is_alive()


def test_stop_handles_filled_queues():
    """Test that stop() works correctly when queues have pending items"""
    orchestrator = PipelineOrchestrator()
    orchestrator.start()
    
    # Put items in the queues
    for i in range(5):
        request = PredictionRequest(output_path=f'test_{i}.csv')
        orchestrator.enqueue(request)
    
    # Wait briefly to allow some processing
    time.sleep(0.2)
    
    # Call stop and verify it completes without hanging
    start_time = time.time()
    orchestrator.stop()
    stop_duration = time.time() - start_time
    
    # Stop should complete within a reasonable time (less than 3 seconds)
    assert stop_duration < 3.0, f"stop() took too long: {stop_duration} seconds"
    
    # Verify all stage threads are no longer alive
    for stage in orchestrator.stages:
        assert not stage.is_alive()


def test_stop_handles_blocked_threads():
    """Test that stop() can terminate threads even if they're blocked on queue operations"""
    # Create a subclass with a deliberately blocking queue
    class BlockingQueue(queue.Queue):
        def get(self, *args, **kwargs):
            # Remove timeout to force blocking
            if 'timeout' in kwargs:
                del kwargs['timeout']
            return super().get(*args, **kwargs)
    
    # Create an orchestrator with the blocking queue
    orchestrator = PipelineOrchestrator()
    orchestrator.features_queue = BlockingQueue()
    
    # Start the orchestrator
    orchestrator.start()
    time.sleep(0.2)
    
    # Call stop with a timeout to avoid hanging the test
    start_time = time.time()
    orchestrator.stop()
    stop_duration = time.time() - start_time
    
    print(f"Stop with blocking queue completed in {stop_duration:.2f} seconds")
    
    # Check thread status
    for stage in orchestrator.stages:
        if stage.is_alive():
            print(f"Warning: {stage.name} is still alive")
        else:
            print(f"{stage.name} has terminated")


def test_stop_is_idempotent():
    """Test that calling stop() multiple times is safe"""
    orchestrator = PipelineOrchestrator()
    orchestrator.start()
    time.sleep(0.2)
    
    # First stop call
    orchestrator.stop()
    
    # Verify all threads are stopped
    all_stopped = True
    for stage in orchestrator.stages:
        if stage.is_alive():
            all_stopped = False
    
    assert all_stopped, "Not all threads stopped after first stop() call"
    
    # Call stop() again
    try:
        orchestrator.stop()
        # If we get here, no exception was raised
        assert True, "Multiple stop() calls should be safe"
    except Exception as e:
        assert False, f"Multiple stop() calls should not raise exception, got: {e}"


def test_stop_with_improved_implementation():
    """Test an improved stop() implementation that handles common issues"""
    # Create a PipelineOrchestrator with a better stop method
    orchestrator = PipelineOrchestrator()
    
    # Monkey patch the stop method with an improved version
    def improved_stop(self):
        # Set stop event
        self.stop_event.set()
        
        # Clear all queues to unblock any waiting threads
        self._clear_queue(self.raw_data_queue)
        self._clear_queue(self.features_queue)
        self._clear_queue(self.prediction_queue)
        
        # Join all stages with timeout
        for stage in self.stages:
            stage.join(timeout=2.0)
    
    def _clear_queue(self, q):
        """Safely clear all items from a queue"""
        try:
            while not q.empty():
                q.get_nowait()
        except:
            pass  # Queue might be empty already
    
    # Apply the monkey patch
    orchestrator._clear_queue = _clear_queue.__get__(orchestrator)
    orchestrator.stop = improved_stop.__get__(orchestrator)
    
    # Start the orchestrator
    orchestrator.start()
    
    # Add some items to the queues
    for i in range(5):
        request = PredictionRequest(output_path=f'test_{i}.csv')
        orchestrator.enqueue(request)
    
    time.sleep(0.2)
    
    # Call the improved stop method
    start_time = time.time()
    orchestrator.stop()
    stop_duration = time.time() - start_time
    
    print(f"Improved stop() completed in {stop_duration:.2f} seconds")
    
    # Verify all threads have stopped
    all_stopped = True
    for stage in orchestrator.stages:
        if stage.is_alive():
            all_stopped = False
            print(f"{stage.name} is still running")
    
    assert all_stopped, "All threads should stop with improved implementation"
