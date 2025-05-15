import threading
from components.data_source import CSVDataSource, YahooFinDataSource
import time
import datetime
from config.logger import logger
import queue

class DataStage(threading.Thread):
    def __init__(self, input_queue, output_queue, stop_event):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event

    def run(self):
        logger.debug(f"Starting {self.__class__} : {self.name}")
        while not self.stop_event.is_set():
            try:
                logger.debug("Data found")
                request = self.input_queue.get(timeout=1)
                # df = CSVDataSource().load_data(request.data_path)
                df = YahooFinDataSource().load_data()
                self.output_queue.put((df, request))
                logger.debug(f"Data collected")
            except queue.Empty:
                continue
            except Exception as e:
                logger.warning(f"Smth went wrong: {e}")
                continue
