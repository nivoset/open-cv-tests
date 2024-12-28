from flask import Flask, jsonify, Response;
from Card import Card
from Detector import Detector
from settings import settings
import json
import cv2


detector = Detector(
    sort_method=settings["sort_method"],
    brightness_threshold=settings["brightness_threshold"],
    epsilon_factor=settings["epsilon_factor"],
    min_area=settings["min_area"],
    debug=settings["debug"],
)
cardDetector = Card(settings)

camera = cv2.VideoCapture(settings['input_device'])

if not camera.isOpened():
  print("error accessing camera")
  exit()

app = Flask(__name__)

# Define a simple route
@app.route('/')
def home():
    return jsonify(settings)

# Define an API route
@app.route('/api/data', methods=['GET'])
def get_data():
    image_path = 'image001.png'
    image = cv2.imread(image_path)
  
    
    hands = detector.cards(image)
    
    cards = cardDetector.from_hand(hands)
    
    return Response(json.dumps(cards), content_type="application/json")

# Start the server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)