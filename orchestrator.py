import threading
import queue
from schemas import PredictionRequest
from stages.data_stage import DataStage
from stages.feature_stage import FeatureStage
from stages.predict_stage import PredictStage
from config.logger import logger

class PipelineOrchestrator:
    def __init__(self):
        self.raw_data_queue = queue.Queue()
        self.features_queue = queue.Queue()
        self.prediction_queue = queue.Queue()
        self.stop_event = threading.Event()

        self.data_stage = DataStage(self.raw_data_queue, self.features_queue, self.stop_event)
        self.feature_stage = FeatureStage(self.features_queue, self.prediction_queue, self.stop_event)
        self.predict_stage = PredictStage(self.prediction_queue, stop_event=self.stop_event)

        self.stages = [self.data_stage, self.feature_stage, self.predict_stage]

    def start(self):
        for stage in self.stages:
            logger.debug(f"Starting {stage.__class__} : {stage.name}")
            stage.start()

    def enqueue(self, request: PredictionRequest):
        self.raw_data_queue.put(request)

    def stop(self):
        # Set stop event
        self.stop_event.set()

        # Clear all queues to unblock any waiting threads
        self._clear_queue(self.raw_data_queue)
        self._clear_queue(self.features_queue)
        self._clear_queue(self.prediction_queue)

        # Join all stages with timeout
        for stage in self.stages:
            stage.join(timeout=1.0)
            if stage.is_alive():
                logger.warning(f"Warning: {stage.name} did not terminate properly")


    def _clear_queue(self, q):
        """Safely clear all items from a queue"""
        try:
            while not q.empty():
                q.get_nowait()
                q.task_done()
        except Exception as e:
            logger.debug(f"Error clearing queue: {e}")