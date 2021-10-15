import tensorflow as tf

path = "/Users/gkar/git/zenith/server/recommender/models"

# Load it back; can also be done in TensorFlow Serving.
loaded = tf.saved_model.load(path)

# Pass a user id in, get top predicted movie titles back.
scores, titles = loaded(["53"])

print(f"Recommendations: {titles[0][:3]}")
