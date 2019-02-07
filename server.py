import pickle
import flask
from flask import request

#code which helps initialize our server
app = flask.Flask(__name__)

#loading my model
model = pickle.load(open("model.pkl","rb"))

#defining a route for only post requests
@app.route('/', methods=['POST'])
def index():
    #getting an array of features from the post request's body
    feature_array = request.get_json()['feature_array']

    #creating a response object
    #storing a model's prediction in the object
    response = {}
    response['predictions'] = model.predict([feature_array]).tolist()

    return flask.jsonify(response)

if __name__=="__main__":
    app.run(debug=True,port=8080)