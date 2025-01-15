from flask import Flask, jsonify, Response
from Hand import Hand
from Detector import Detector
from settings import settings
import json
import cv2
import threading
import time

# Initialize detector and card detector
detector = Detector(
    sort_method=settings["sort_method"],
    brightness_threshold=settings["brightness_threshold"],
    epsilon_factor=settings["epsilon_factor"],
    min_area=settings["min_area"],
    debug=settings["debug"],
)
cardDetector = Hand(settings)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("error accessing camera")
    exit()

app = Flask(__name__)

# Cache variable and lock for thread safety
cached_data = None
cache_lock = threading.Lock()

# Time interval to update the cache (in seconds)
CACHE_UPDATE_INTERVAL = 1

# Background thread function to update cache
def update_cache():
    global cached_data
    while True:
        ret, image = camera.read()
        if not ret:
          print("no frame")
          continue        
        
        hands = detector.cards(image)
        cards = cardDetector.from_hand(hands)

        # cv2.imshow("Captured Image", image)
        print(len(hands), len(cards))
        
        # Acquire lock to safely update the cache
        with cache_lock:
            cached_data = cards
        # Wait for the next update
        time.sleep(CACHE_UPDATE_INTERVAL)

# Start the cache updating thread
cache_thread = threading.Thread(target=update_cache, daemon=True)
cache_thread.start()

# Define a simple route
@app.route('/')
def home():
    return jsonify(settings)

# Define an API route
@app.route('/api/data', methods=['GET'])
def get_data():
    # Return the cached data
    with cache_lock:  # Ensure thread safety when accessing cached_data
        if cached_data is None:
            return Response(json.dumps({"error": "Cache not initialized"}), content_type="application/json")
        return Response(json.dumps(cached_data), content_type="application/json", mimetype="application/json")

# Start the server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
    
camera.release()