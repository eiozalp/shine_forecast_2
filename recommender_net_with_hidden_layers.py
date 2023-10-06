import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from keras import layers


class RecommenderNetWithHiddenLayers(keras.Model):
    def __init__(self, latent_dim=10, **kwargs):
        super().__init__(**kwargs)
        with open('dp.joblib', 'rb') as f:
          self.data_processing = joblib.load(f)

        self.latent_dim = latent_dim

        self.hidden_units = [128, 64]

        # Customer matrix factorization layers
        self.customer_matrix = layers.Embedding(
            self.data_processing.num_customer_id_encodeds,
            latent_dim,
            embeddings_initializer="he_normal",
            name="customer_matrix",
            embeddings_regularizer=keras.regularizers.l2(1e-3),
        )

        # Product matrix factorization layers
        self.product_matrix = layers.Embedding(
            self.data_processing.num_product_id_encodeds,
            latent_dim,
            embeddings_initializer="he_normal",
            name="product_matrix",
            embeddings_regularizer=keras.regularizers.l2(1e-3),
        )

        self.hidden_layers = [layers.Dense(units, activation='relu', name='hidden_layer') for units in self.hidden_units]

        # regularization
        # self.regularization_layer = layers.Dense(1, activation='relu', name='regularization_layer', kernel_regularizer=keras.regularizers.l2(1e-5))
        self.dropout = layers.Dropout(0.2)

        # Output layer
        self.output_layer = layers.Dense(1, name='output_layer')

    def create_feature_embeddings(self, features, inputs):
        embeddings = []

        for feature in features:
            if feature in self.data_processing.input_customer_features:
                # Customer feature: use the corresponding embedding layer
                feature_embedding = self.customer_matrix(inputs[:, self.data_processing.input_features.index(feature)])
                embeddings.append(feature_embedding)
            elif feature in self.data_processing.input_product_features:
                # Product feature: use the corresponding embedding layer
                feature_embedding = self.product_matrix(inputs[:, self.data_processing.input_features.index(feature)])
                embeddings.append(feature_embedding)
            else:
                # Handle any other cases if needed
                pass

        embeddings_concatenated = tf.concat(embeddings, axis=1)

        print("Shape of embeddings_concatenated:", embeddings_concatenated.shape)

        return embeddings_concatenated



    def call(self, inputs):
        customer_embeddings = self.create_feature_embeddings(self.data_processing.input_customer_features, inputs)
        product_embeddings = self.create_feature_embeddings(self.data_processing.input_product_features, inputs)

        # Expand dimensions for dot product
        customer_embeddings_expanded = tf.expand_dims(customer_embeddings, axis=2)
        product_embeddings_expanded = tf.expand_dims(product_embeddings, axis=1)

        # Dot product
        interaction_terms = tf.keras.layers.Dot(axes=(2, 1))([customer_embeddings_expanded, product_embeddings_expanded])

        # Flatten the result
        interaction_terms = tf.keras.layers.Flatten()(interaction_terms)

        for hidden_layer in self.hidden_layers:
            interaction_terms = hidden_layer(interaction_terms)

        #  Add regularization
        # interaction_terms = self.regularization_layer(interaction_terms)

        # Add dropout
        interaction_terms = self.dropout(interaction_terms)

        x = self.output_layer(interaction_terms)

        return x


    def print_summary(self):
        self.build((None, len(self.data_processing.input_features)))  # Specify input shape
        self.summary()