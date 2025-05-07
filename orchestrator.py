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

    def stop(self):
        self.stop_event.set()
        for stage in self.stages:
            stage.join()

    def enqueue(self, request: PredictionRequest):
        self.raw_data_queue.put(request)
