import joblib
import pandas as pd
from data_processing import DataProcessing
import boto3
import os
def read_file_from_s3(bucket_name, file_name):
  s3 = boto3.resource(
  service_name='s3',
  region_name=os.getenv("AWS_REGION_NAME"),
  aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
  aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)
    
  try:
    obj = s3.Object(bucket_name, file_name)
    file_content = obj.get()['Body']
    return file_content
  except Exception as e:
    print(f"Error reading the file: {e}")


def create_data_processing():
  file = read_file_from_s3("demotlbucket", "isen_customer_ratings.csv")

  df = pd.read_csv(file)
  # df = pd.read_csv('/Users/13596107/Desktop/customer_product_ratings.csv')
  df['wears_gold'] = df['wears_gold'].astype(bool)
  df['wears_silver'] = df['wears_silver'].astype(bool)
  df['wears_rose_gold'] = df['wears_rose_gold'].astype(bool)
  # scale ratings
  df.loc[df['rating'] == 1, 'rating'] = 0
  # df.loc[df['rating'] == 1, 'rating'] = 5
  df.loc[df['rating'] == 3, 'rating'] = 5
  df.loc[df['rating'] == 4, 'rating'] = 10
  df.loc[df['rating'] == 5, 'rating'] = 15
  # Create an instance of the DataProcessing class
  data_processing = DataProcessing(df.sample(frac=1, random_state=42))

  data_processing.normalized_values("purchase_conversion")
  data_processing.normalized_values("price")
  data_processing.normalized_values("rating")
  data_processing.one_hot_encode_categorical_features("ltv_tier")
  data_processing.one_hot_encode_categorical_features("official_source")
  data_processing.tokenize_string_feature("product_name")
  data_processing.one_hot_encode_categorical_features("metal")
  data_processing.one_hot_encode_categorical_features("kind")
  # data_processing.tokenize_string_feature("designer")
  data_processing.one_hot_encode_categorical_features("sub_kind")
  data_processing.one_hot_encode_categorical_features("loudness_type")
  data_processing.one_hot_encode_categorical_features("color")
  data_processing.get_model_inputs()

  return data_processing

with open('dp.joblib', 'wb') as f:
    joblib.dump(create_data_processing(),f)