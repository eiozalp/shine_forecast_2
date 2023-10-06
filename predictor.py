import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
# import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class Predictor:
    def __init__(self, model, data_processing):
      self.model = model
      self.data_processing = data_processing

    def predict(self, customer_id, dataframe):
        inputs = self.format_inputs_for_prediction(customer_id, dataframe)
        # print("Inputs for Prediction:", inputs)  # Add this line for debugging
        predictions = self.model.predict(inputs)
        # print("Raw Predictions:", predictions)

        return predictions

    def format_inputs_for_prediction(self, customer_id, dataframe):
        customer_data = dataframe[dataframe['customer_id'] == customer_id]

        customer_features = [np.full(len(dataframe), getattr(self.data_processing, f"{feature}2{feature}_encoded").get(customer_data[feature].values[0], 0)) for feature in self.data_processing.CUSTOMER_FEATURES]

        product_features = [dataframe[f"{feature}_encoded"].values for feature in self.data_processing.PRODUCT_FEATURES]

        # Include the tokenized name feature
        # name_encoded = self.data_processing.name_tokenizer.texts_to_sequences([dataframe["name"].values[0]])[0]
        # name_encoded = pad_sequences([name_encoded], maxlen=self.data_processing.max_name_length, padding="post").flatten()
        # product_features.append(name_encoded)

        input_additional_features = [dataframe[feature].values for feature in self.data_processing.INPUT_FEATURES]

        # Print shapes for debugging
        # print("Shapes of arrays before stacking:")
        # print("Customer Features:", [arr.shape for arr in customer_features])
        # print("Product Features:", [arr.shape for arr in product_features])
        # print("Additional Features:", [arr.shape for arr in input_additional_features])

        # Stack arrays
        columns = customer_features + product_features + input_additional_features
        input_data = np.stack(columns, axis=1)

        # Print the shape of the stacked array
        # print("Shape of the stacked array:", input_data.shape)

        return input_data

    def print_predictions(predictions, dataframe):
        recommended_products = []

        # Populate the list with products that meet the threshold criteria
        for product_id, prediction in zip(dataframe["product_id"].unique(), predictions):
            recommended_products.append((product_id, prediction))

        # Sort the list of recommended products by probability in descending order
        recommended_products.sort(key=lambda x: x[1], reverse=True)

        # Print the top 10 recommended products
        top_10_recommendations = recommended_products[:100]
        print("top 10", top_10_recommendations)
        products = []
        for index, (product_id, probability) in enumerate(top_10_recommendations):
            print(f"Product ID: {product_id}, Rating prediction: {probability} vs Real rating: {dataframe[dataframe.product_id == product_id]['rating'].values[0]}")
            products.append({index: product_id})

        return products
