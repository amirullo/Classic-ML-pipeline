from fastapi import FastAPI
from orchestrator import PipelineOrchestrator
from schemas import PredictionRequest
import time
import datetime as dt

app = FastAPI()
orchestrator = PipelineOrchestrator()
orchestrator.start()

@app.get("/predict")
def predict_route():
    request = PredictionRequest(output_path='some_predict.csv')
    orchestrator.enqueue(request)
    return {"status": "Prediction enqueued", "datetime": dt.datetime.now()}

@app.get("/score")
def get_last_score():
    score = orchestrator.predict_stage.score
    last_dt = orchestrator.predict_stage.last_dt
    return {"score": score, "last_dt": last_dt}


if __name__ == "__main__":
    request = PredictionRequest(output_path='some_predict.csv')
    orchestrator.enqueue(request)
    print("Prediction task submitted. Press ^ + C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        orchestrator.stop()
        print("Stopped.")
