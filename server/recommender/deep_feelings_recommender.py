from typing import Dict, Text
import pandas as pd
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

# load emotions activities data
emotion_activities = tf.data.experimental.CsvDataset(
  "/Users/gkar/git/zenith/server/recommender/datasets/emotion_activities.csv",
  [
    tf.string, # Required field, use dtype or empty tensor
    tf.string,  # Required field, use dtype or empty tensor
    tf.string,  # Required field, use dtype or empty tensor
     tf.string,  # Required field, use dtype or empty tensor
     tf.string,  # Required field, use dtype or empty tensor
  ],
  header=True,
  select_cols=[0,1,2,3,4]  # Only parse first two columns
)
# Features of all the activities
activities =tf.data.experimental.CsvDataset(
  "/Users/gkar/git/zenith/server/recommender/datasets/activity.csv",
  [tf.string,  # Required field, use dtype or empty tensor
   tf.int32,  # Required field, use dtype or empty tensor
  ],
  header=True,
  select_cols=[0,1]  # Only parse first two columns
)


emotion_activities = emotion_activities.map(lambda a,b,c,d,e: {
    "userId": a,
    "emotion": b,
    "activity": c,
    "emotionId": d,
    "activityId": e

})

activities = activities.map(lambda x,y: x)

#Get unique activities
unique_activities = np.unique(np.concatenate(list(activities.batch(1000))))

# get unique user ids
unique_user_ids = np.unique(np.concatenate(list(emotion_activities.batch(1000).map(
    lambda x: x["userId"]))))

unique_feeling_ids = np.unique(np.concatenate(list(emotion_activities.batch(1000).map(
    lambda x: x["emotionId"]))))

class UserModel(tf.keras.Model):

  def __init__(self):
    super().__init__()

    self.user_embedding = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_user_ids, mask_token=None),
        tf.keras.layers.Embedding(len(unique_user_ids) + 1, 32),
    ])
    self.feelings_embedding = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_feeling_ids, mask_token=None),
        tf.keras.layers.Embedding(len(unique_feeling_ids) + 1, 32),
    ])

  def call(self, inputs):
    # Take the input dictionary, pass it through each input layer,
    # and concatenate the result.
    return tf.concat([
        self.user_embedding(inputs["userId"]),
        self.feelings_embedding(inputs["emotionId"]),
    ], axis=1)

class QueryModel(tf.keras.Model):
  """Model for encoding user queries."""

  def __init__(self, layer_sizes):
    """Model for encoding user queries.

    Args:
      layer_sizes:
        A list of integers where the i-th entry represents the number of units
        the i-th layer contains.
    """
    super().__init__()

    # We first use the user model for generating embeddings.
    self.embedding_model = UserModel()

    # Then construct the layers.
    self.dense_layers = tf.keras.Sequential()

    # Use the ReLU activation for all but the last layer.
    for layer_size in layer_sizes[:-1]:
      self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

    # No activation for the last layer.
    for layer_size in layer_sizes[-1:]:
      self.dense_layers.add(tf.keras.layers.Dense(layer_size))

  def call(self, inputs):
    feature_embedding = self.embedding_model(inputs)
    return self.dense_layers(feature_embedding)


class ActivityModel(tf.keras.Model):

  def __init__(self):
    super().__init__()

    max_tokens = 10000

    self.activity_embedding = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
          vocabulary=unique_activities,mask_token=None),
      tf.keras.layers.Embedding(len(unique_activities) + 1, 32)
    ])


  def call(self, titles):
    return tf.concat([
        self.activity_embedding(titles),
    ], axis=1)


class CandidateModel(tf.keras.Model):
  """Model for encoding movies."""

  def __init__(self, layer_sizes):
    """Model for encoding movies.

    Args:
      layer_sizes:
        A list of integers where the i-th entry represents the number of units
        the i-th layer contains.
    """
    super().__init__()

    self.embedding_model = MovieModel()

    # Then construct the layers.
    self.dense_layers = tf.keras.Sequential()

    # Use the ReLU activation for all but the last layer.
    for layer_size in layer_sizes[:-1]:
      self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

    # No activation for the last layer.
    for layer_size in layer_sizes[-1:]:
      self.dense_layers.add(tf.keras.layers.Dense(layer_size))

  def call(self, inputs):
    feature_embedding = self.embedding_model(inputs)
    return self.dense_layers(feature_embedding)

class FeelinglensModel(tfrs.models.Model):

  def __init__(self, layer_sizes):
    super().__init__()
    self.query_model = QueryModel(layer_sizes)
    self.candidate_model = CandidateModel(layer_sizes)
    self.task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=movies.batch(128).map(self.candidate_model),
        ),
    )

  def compute_loss(self, features, training=False):
    # We only pass the user id and timestamp features into the query model. This
    # is to ensure that the training inputs would have the same keys as the
    # query inputs. Otherwise the discrepancy in input structure would cause an
    # error when loading the query model after saving it.
    query_embeddings = self.query_model({
        "userId": features["userId"],
        "emotionId": features["emotionId"],
    })
    activity_embeddings = self.candidate_model(features["activity"])

    return self.task(
        query_embeddings, activity_embeddings, compute_metrics=not training)

tf.random.set_seed(42)
shuffled = emotion_activities.shuffle(10, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(8)
test = shuffled.skip(8).take(2)

cached_train = train.shuffle(10).batch(2)
cached_test = test.batch(4).cache()

# shallow model
num_epochs = 300

model = FeelinglensModel([32])
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

one_layer_history = model.fit(
    cached_train,
    validation_data=cached_test,
    validation_freq=5,
    epochs=num_epochs,
    verbose=0)

accuracy = one_layer_history.history["val_factorized_top_k/top_100_categorical_accuracy"][-1]
print(f"Top-100 accuracy: {accuracy:.2f}.")