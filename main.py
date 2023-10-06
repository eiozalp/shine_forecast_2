from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
from typing import List
from get_data_processing import *
from model import Model


app = FastAPI()

@app.get('/train')
async def train():
  data_processing = create_data_processing()
  with open('dp.joblib', 'wb') as f:
    joblib.dump(data_processing,f)


  model = Model()
  model.split()
  model.fit()

  joblib.dump(model, "model.joblib")
  return {"success"}


class PredictRequest(BaseModel):
    customer_id: int
    product_ids: List[int]| None = None

@app.post('/predict/')
async def get_predict(predictRequest: PredictRequest):
  with open('model.joblib', 'rb') as f:
    model = joblib.load(f)

  return json.dumps(model.predict(predictRequest.customer_id, predictRequest.product_ids), default=str)





