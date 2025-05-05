from fastapi import FastAPI
from orchestrator import PipelineOrchestrator
from schemas import PredictionRequest

app = FastAPI()
orchestrator = PipelineOrchestrator()
orchestrator.start()

@app.post("/predict")
def predict_route(request: PredictionRequest):
    orchestrator.enqueue(request)
    return {"status": "Prediction enqueued"}
