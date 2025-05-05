import logging

# Configure the global logger
logger = logging.getLogger("ml_service")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(threadName)s %(name)s %(funcName)s() %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
