from inference import BERT_NER_inference, get_model
from typing import Dict, List
from fastapi import Depends, FastAPI
from pydantic import BaseModel

app = FastAPI()


class NER_Request(BaseModel):

    text: str


class NER_Response(BaseModel):

    predictions: Dict[str, str]


@app.post("/predict", response_model=NER_Response)
def predict(request: NER_Request, model: BERT_NER_inference = Depends(get_model)):

    predictions = model.predict(request.text)

    return NER_Response(predictions=predictions)

