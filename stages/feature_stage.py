import threading
from components.feature_engineer import LagFeatureEngineer, FillNAFeatures

from config.logger import logger

class FeatureStage(threading.Thread):
    def __init__(self, input_queue, output_queue, stop_event):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event

    def run(self):
        logger.debug(f"Starting {self.__class__} : {self.name}")
        while not self.stop_event.is_set():
            try:
                df, request = self.input_queue.get(timeout=1)
                # df = LagFeatureEngineer().transform(df)
                df = FillNAFeatures().transform(df)

                self.output_queue.put((df, request))
            except Exception:
                continue
