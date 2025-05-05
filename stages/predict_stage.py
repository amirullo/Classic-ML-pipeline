import threading
from components.model import MLModel

from config.logger import logger

class PredictStage(threading.Thread):
    def __init__(self, input_queue, stop_event):
        super().__init__()
        self.input_queue = input_queue
        self.stop_event = stop_event

    def run(self):
        while not self.stop_event.is_set():
            try:
                df, request = self.input_queue.get(timeout=1)
                predictions = MLModel().predict(df)
                predictions.to_csv(request.output_path, index=False)
            except Exception:
                continue
