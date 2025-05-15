import datetime
import threading
from components.model import MLModel
import datetime as dt
from typing import Optional
from config.logger import logger
import queue

class PredictStage(threading.Thread):
    def __init__(self, input_queue, stop_event):
        super().__init__()
        self.input_queue = input_queue
        self.stop_event = stop_event
        self.score = .00
        self.last_dt: Optional[datetime] = None

    def run(self):
        logger.debug(f"Starting {self.__class__} : {self.name}")
        while not self.stop_event.is_set():
            try:
                df, request = self.input_queue.get(timeout=1)
                predictions, self.score = MLModel().predict(df)
                self.last_dt = dt.datetime.now()
                predictions.to_csv(request.output_path, index=False)
            except queue.Empty:
                continue
            except Exception as e:
                logger.warning(f"Smth went wrong: {e}")
                continue
