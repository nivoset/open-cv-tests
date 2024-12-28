from flask import Flask, jsonify, request;
from Detector import Detector
from settings import settings


detector = Detector()


app = Flask(__name__)

# Define a simple route
@app.route('/')
def home():
    return jsonify(settings)

# Define an API route
@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify(settings)

# Start the server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)