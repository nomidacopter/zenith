import tensorflow as tf
import sys
from flask import Flask
from flask import jsonify
app = Flask(__name__)
from util import readFileToDict

path = "/Users/gkar/git/zenith/server/recommender/models"

# Load it back; can also be done in TensorFlow Serving.
loaded = tf.saved_model.load(path)

# Load the lookup dicts
activities = readFileToDict("datasets/activity.csv")
feelings = readFileToDict("datasets/feelings.csv")

def addEntry(user,feeling,activity):
  with open('datasets/emotion_activities.csv', 'a') as fd:
    activityID = activities[activity]
    feelingID = feelings[feeling]
    row = "\n" + ",".join([user,feeling,feelingID,activity,activityID])
    fd.write(row)

# Define activity list recommendation route
@app.route("/users/<user>/feelings/<feeling>")
def getactivity(user,feeling):
  _, activities = loaded({"userId": tf.constant([user]), "emotion": tf.constant([feeling])})
  activityList =[x.decode('utf-8') for x in activities[0].numpy().tolist()]
  return jsonify({"recomnendations": activityList[:3]})

# Define feelings selection route
@app.route("/users/<user>/feelings/<feeling>/activities/<activity>")
def acceptactivity(user,feeling,activity):
  addEntry(user,feeling,activity)
  return jsonify({"status":"success"})

# Define activity ratings rate
@app.route("/users/<user>/feelings/<feeling>/activities/<activity>/rating/<rating>")
def rateactivity(user,feeling,activity,rating):
  print("TODO")


if __name__ == "__main__":
    app.run()

