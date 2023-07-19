import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet, efficientnet


def get_embedder(target_shape, id_=0, freeze=True):
    weights = "imagenet" if target_shape[2] == 3 else None
    print("weights are", weights)
    base = efficientnet.EfficientNetB2  # resnet.ResNet50
    base_cnn = base(
        weights=weights, input_shape=target_shape, include_top=False, pooling="avg",
    )

    flatten = layers.Flatten()(base_cnn.output)
    dense1 = layers.Dense(512, activation="relu")(flatten)
    dense1 = layers.BatchNormalization()(dense1)
    dense2 = layers.Dense(256, activation="relu")(dense1)
    dense2 = layers.BatchNormalization()(dense2)
    output = layers.Dense(64)(dense2)

    embedding = Model(base_cnn.input, output, name=f"embedding_{id_}")

    if freeze:
        trainable = False
        for layer in base_cnn.layers:
            if layer.name == "conv5_block1_out" or layer.name == "block5a_expand_conv":  # resnet of efn
                trainable = True
            layer.trainable = trainable

    return embedding


def get_siamese_model(anchor_shape, mapping_shape):
    anchor_input = layers.Input(name="anchor", shape=anchor_shape)
    positive_input = layers.Input(name="positive", shape=mapping_shape)
    negative_input = layers.Input(name="negative", shape=mapping_shape)
    embedding_anchor = get_embedder(anchor_shape, 0)
    embedding_mapping = get_embedder(mapping_shape, 1)

    distances = DistanceLayer()(
        embedding_anchor(anchor_input),
        embedding_mapping(positive_input),
        embedding_mapping(negative_input),
    )

    siamese_network = Model(
        inputs=[anchor_input, positive_input, negative_input], outputs=distances
    )

    return siamese_network


def get_siamese_model_uniform(output_shape, embedding=None):
    anchor_input = layers.Input(name="anchor", shape=output_shape)
    positive_input = layers.Input(name="positive", shape=output_shape)
    negative_input = layers.Input(name="negative", shape=output_shape)
    if embedding is None:
        embedding = get_embedder(output_shape, 0)

    distances = DistanceLayer()(
        embedding(anchor_input),
        embedding(positive_input),
        embedding(negative_input),
    )

    siamese_network = Model(
        inputs=[anchor_input, positive_input, negative_input], outputs=distances
    )

    return siamese_network


class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return ap_distance, an_distance


class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]


def get_autoencoder(embedding):
    input_shape = embedding.input_shape
    output_shape = embedding.output_shape

    print(input_shape)
    inputs = layers.Input(name="ae_input", shape=input_shape[1:])
    x = embedding(inputs)

    x = layers.Dense(8 * 8 * 32, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Reshape(target_shape=(8, 8, 32))(x)
    x = layers.Conv2DTranspose(  # 16x16
        filters=64, kernel_size=3, strides=2, padding='same',
        activation='relu')(x)
    x = layers.Conv2DTranspose(  # 32x32
        filters=64, kernel_size=3, strides=2, padding='same',
        activation='relu')(x)
    x = layers.Conv2DTranspose(  # 64x64
        filters=32, kernel_size=3, strides=2, padding='same',
        activation='relu')(x)
    x = layers.Conv2DTranspose(  # 128x128
        filters=16, kernel_size=3, strides=2, padding='same',
        activation='relu')(x)
    out = layers.Conv2DTranspose(  # 256x256
        filters=3, kernel_size=3, strides=2, padding='same')(x)

    return Model(inputs, out)
