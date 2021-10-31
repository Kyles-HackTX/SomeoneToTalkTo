from flask import Flask, jsonify, request
from flask_restful import reqparse, abort, Api, Resource
import json
import torch
from predict import Predict


app = Flask(__name__)
api = Api(app)

predict = Predict()

class AudioUploadResource(Resource):
    def get(self):
        return {'hello': 'GET'}
    def post(self):
        data = request.get_json()
        print(data)
        print(request.files["audio_data"])
        f = request.files['audio_data']
        with open('audio.wav', 'wb') as audio:
            f.save(audio)
        emotion = Predict("audio.wav")
        print(emotion)
        return {'emotion': emotion}
        # Validate the request
        # Capture the audio data from the request


        # invoke the necessary functions on the model

        # Return response and status code
api.add_resource(AudioUploadResource, '/upload')

if __name__ == '__main__':
    app.run(host='0.0.0.0')