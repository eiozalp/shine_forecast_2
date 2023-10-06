import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
# import matplotlib.pyplot as plt
from data_processing import DataProcessing
from predictor import Predictor
from recommender_net_with_hidden_layers import RecommenderNetWithHiddenLayers


class Model():
  def __init__(self):
    with open('dp.joblib', 'rb') as f:
      self.data_processing = joblib.load(f)

  def split(self):
    self.x_train, self.x_val, self.y_train, self.y_val = self.data_processing.get_model_inputs()

  def fit(self):
    self.model = RecommenderNetWithHiddenLayers()
    self.model.print_summary()
    # Compile the model
    self.model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        metrics=['accuracy', keras.metrics.Precision()]
    )
    self.history = self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=1000, validation_data=(self.x_val, self.y_val))
    # plt.plot(self.history.history["loss"])
    # plt.plot(self.history.history["val_loss"])
    # plt.title("model loss")
    # plt.ylabel("loss")
    # plt.xlabel("epoch")
    # plt.legend(["train", "test"], loc="upper left")
    # plt.show()


  def predict(self, customer_id, product_ids = None):
    dataframe = self.data_processing.dataframe

    if product_ids:
      dataframe = dataframe[dataframe['product_id'].isin(product_ids)]

    predictions = Predictor(self.model, self.data_processing).predict(customer_id, dataframe)
    recommendations = Predictor.print_predictions(predictions, dataframe)

    return recommendations
