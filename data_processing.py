import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class DataProcessing:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.CUSTOMER_FEATURES = ["customer_id"]
        self.PRODUCT_FEATURES = ["product_id", "product_name", "sub_kind", "loudness_type"]
        self.INPUT_FEATURES = []

        self.name_tokenizer = Tokenizer()
        self.max_name_length = 20  #

    def one_hot_encode_categorical_features(self, feature):
        encoded_features = pd.get_dummies(self.dataframe[feature], prefix=feature)
        self.dataframe = pd.concat([self.dataframe, encoded_features], axis=1)
        return encoded_features

    def tokenize_string_feature(self, feature):
        # Tokenize product names
        self.name_tokenizer.fit_on_texts(self.dataframe[feature])
        self.num_names = len(self.name_tokenizer.word_index) + 1

        # Convert product names to sequences
        name_sequences = self.name_tokenizer.texts_to_sequences(self.dataframe[feature])

        # Pad sequences
        self.dataframe[f"{feature}_encoded"] = pad_sequences(name_sequences, maxlen=self.max_name_length, padding="post").tolist()

    def normalized_values(self, feature):
        min_value = min(self.dataframe[feature])
        max_value = max(self.dataframe[feature])
        normalized_values = [(round(float((value - min_value) / (max_value - min_value)), 3)) for value in self.dataframe[feature]]
        self.dataframe[feature] = normalized_values
        return normalized_values

    def create_encoding_mappings(self, feature):
        unique_values = self.dataframe[feature].unique().tolist()
        feature2encoded = {x: i for i, x in enumerate(unique_values)}
        return feature2encoded, f"{feature}_encoded"

    def encode_feature(self, feature):
        feature2encoded, encoded_name = self.create_encoding_mappings(feature)
        setattr(self, f"{feature}2{feature}_encoded", feature2encoded)
        self.dataframe[encoded_name] = self.dataframe[feature].map(feature2encoded)
        setattr(self, f"num_{encoded_name}s", len(feature2encoded))
        return encoded_name

    def encode_features(self):
        self.input_customer_features = [self.encode_feature(feature) for feature in self.CUSTOMER_FEATURES]
        self.input_product_features = [self.encode_feature(feature) for feature in self.PRODUCT_FEATURES]
        self.input_additional_features = [self.encode_feature(feature) for feature in self.INPUT_FEATURES]
        if (len(self.input_additional_features) == 0):
          self.input_features = self.input_customer_features + self.input_product_features
        else:
          self.input_features = self.input_customer_features + self.input_product_features + self.input_additional_features

    def get_model_inputs(self):
        # Encode features
        self.encode_features()

        x = self.dataframe[self.input_features].values
        y = self.dataframe["rating"].values

        # Split data into training and validation sets
        train_indices = int(0.8 * self.dataframe.shape[0])
        x_train, x_val, y_train, y_val = (
            x[:train_indices],
            x[train_indices:],
            y[:train_indices],
             y[train_indices:],
        )
        return x_train, x_val, y_train, y_val