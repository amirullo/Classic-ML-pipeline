import threading
from components.data_source import CSVDataSource

from config.logger import logger

class DataStage(threading.Thread):
    def __init__(self, input_queue, output_queue, stop_event):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event

    def run(self):
        while not self.stop_event.is_set():
            try:
                request = self.input_queue.get(timeout=1)
                df = CSVDataSource().load(request.data_path)
                self.output_queue.put((df, request))
            except Exception:
                continue
