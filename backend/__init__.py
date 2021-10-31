from flask import Flask, jsonify, request
from flask_restful import reqparse, abort, Api, Resource
import json

app = Flask(__name__)
api = Api(app)


class AudioUploadResource(Resource):
    def get(self):
        return {'hello': 'GET'}
    def post(self):
        data = request.get_json()
        print(data)
        print(request.files)
        return {'hello': 'POST'}
        # Validate the request
        # Capture the audio data from the request


        # invoke the necessary functions on the model

        # Return response and status code
api.add_resource(AudioUploadResource, '/upload')

if __name__ == '__main__':
    app.run(host='0.0.0.0')