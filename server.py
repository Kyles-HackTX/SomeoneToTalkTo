from flask import Flask, jsonify, request
from flask_restful import reqparse, abort, Api, Resource
import json
import torch
from predict import Predict
import subprocess

app = Flask(__name__)
api = Api(app)

predict = Predict()

def convert_and_split(filename):
    command = ['ffmpeg', '-i', filename, '-f', 'segment', '-segment_time', '15', 'out%09d.wav']
    subprocess.run(command,stdout=subprocess.PIPE,stdin=subprocess.PIPE)

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
        
        convert_and_split('audio.wav')
        
        emotion = predict("output.wav")
        print(emotion)
        return {'emotion': emotion}
        # Validate the request
        # Capture the audio data from the request


        # invoke the necessary functions on the model

        # Return response and status code
api.add_resource(AudioUploadResource, '/upload')

if __name__ == '__main__':
    app.run(host='0.0.0.0')