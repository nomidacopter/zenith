from typing import Dict, Text
import pandas as pd
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

# load emotions activities data
emotion_activities = tf.data.experimental.CsvDataset(
  "/Users/gkar/git/zenith/server/recommender/datasets/emotion_activity.csv",
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

# get unique user ids
unique_user_ids = np.unique(np.concatenate(list(emotion_activities.batch(1_000).map(
    lambda x: x["user_id"]))))

emotion_ids_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
emotion_ids_vocabulary.adapt(emotion_activities.map(lambda x: x['emotionId']))

activity_titles_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
activity_titles_vocabulary.adapt(activities)


class MovieLensModel(tfrs.Model):
  # We derive from a custom base class to help reduce boilerplate. Under the hood,
  # these are still plain Keras Models.

  def __init__(
      self,
      user_model: tf.keras.Model,
      movie_model: tf.keras.Model,
      task: tfrs.tasks.Retrieval):
    super().__init__()

    # Set up user and movie representations.
    self.user_model = user_model
    self.movie_model = movie_model

    # Set up a retrieval task.
    self.task = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # Define how the loss is computed.

    user_embeddings = self.user_model(features["emotionId"])
    movie_embeddings = self.movie_model(features["activity"])

    return self.task(user_embeddings, movie_embeddings)


# Define user and movie models.
user_model = tf.keras.Sequential([
    emotion_ids_vocabulary,
    tf.keras.layers.Embedding(emotion_ids_vocabulary.vocab_size(), 64)
])
movie_model = tf.keras.Sequential([
    activity_titles_vocabulary,
    tf.keras.layers.Embedding(activity_titles_vocabulary.vocab_size(), 64)
])

# Define your objectives.
task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
    activities.batch(128).map(movie_model)
  )
)


# Create a retrieval model.
model = MovieLensModel(user_model, movie_model, task)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))

# Train for 3 epochs.
model.fit(emotion_activities.batch(4096), epochs=3)

# Use brute-force search to set up retrieval using the trained representations.
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
index.index(activities.batch(1024).map(model.movie_model), identifiers=activities)
# Get some recommendations.
_, titles = index(np.array(["2"]))
print(f"Top 3 recommendations for emotion 2: {titles[0, :1]}")

tf.saved_model.save(index, "/Users/gkar/git/zenith/server/recommender/models")