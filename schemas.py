from pydantic import BaseModel

class PredictionRequest(BaseModel):
    # data_path: str
    output_path: str
