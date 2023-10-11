from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
from typing import List, Optional
from get_data_processing import *
from model import Model
import os

app = FastAPI()


@app.get('/train')
async def train():
  data_processing = create_data_processing()
  
  with open('dp.joblib', 'wb') as f:
    joblib.dump(data_processing,f)

  # Initialize an S3 client
  s3 = boto3.client('s3', aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"), aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))

  # Upload the file to S3
  s3.upload_file('dp.joblib', 'demotlbucket', 'dp.joblib')

  model = Model()
  model.split()
  model.fit()

  joblib.dump(model, "model.joblib")
  s3.upload_file('model.joblib', 'demotlbucket', 'model.joblib')
  return {"success"}


class PredictRequest(BaseModel):
    customer_id: int
    product_ids: Optional(List[int])

@app.post('/predict')
async def get_predict(predictRequest: PredictRequest):
  s3 = boto3.client('s3', aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"), aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
  

  s3.download_file('demotlbucket', 'model.joblib', 'model.joblib')

  with open('model.joblib', 'rb') as f:
    model = joblib.load(f)

  return json.dumps(model.predict(predictRequest.customer_id, predictRequest.product_ids), default=str)





