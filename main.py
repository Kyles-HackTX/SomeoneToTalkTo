from flask import Flask
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class AudioUploadResource(Resource):
    def get(self):
        return {'hello': 'GET'}
    def post(self):
        # Validate the request
        # Capture the audio data from the request
        # invoke the necessary functions on the model

        # Return response and status code
        return {'hello': 'POST'}
api.add_resource(AudioUploadResource, '/upload')

if __name__ == '__main__':
    app.run(debug=True)