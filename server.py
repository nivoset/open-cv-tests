from flask import Flask, jsonify, Response
from Hand import Hand
from Detector import Detector
from settings import settings
import json
import cv2
import threading
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize detector and card detector
detector = Detector(
    sort_method=settings["sort_method"],
    brightness_threshold=settings["brightness_threshold"],
    epsilon_factor=settings["epsilon_factor"],
    min_area=settings["min_area"],
    debug=settings["debug"],
)
cardDetector = Hand(settings)

# Initialize camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    logging.critical("[ERROR] Could not access the camera. Exiting.")
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
    failure_count = 0  # Track consecutive failures

    while True:
        ret, image = camera.read()
        if not ret:
            failure_count += 1
            logging.warning(f"[WARNING] No frame captured ({failure_count} attempts)")
            if failure_count > 10:  # Stop after 10 consecutive failures
                logging.critical("[CRITICAL] Camera is not responding. Stopping cache updates.")
                return
            time.sleep(1)
            continue  # Skip processing

        failure_count = 0  # Reset failure count on success

        hands = detector.cards(image)
        cards = cardDetector.from_hand(hands)

        logging.info(f"Hands detected: {len(hands)}, Cards detected: {len(cards)}")

        with cache_lock:  # Safely update the cache
            cached_data = cards

        time.sleep(CACHE_UPDATE_INTERVAL)

# Start the cache updating thread
cache_thread = threading.Thread(target=update_cache, daemon=True)
cache_thread.start()

# Monitor background thread to restart if needed
def monitor_thread():
    while True:
        if not cache_thread.is_alive():
            logging.error("[ERROR] Cache update thread has crashed. Restarting...")
            restart_thread()
        time.sleep(5)

def restart_thread():
    global cache_thread
    cache_thread = threading.Thread(target=update_cache, daemon=True)
    cache_thread.start()

monitoring_thread = threading.Thread(target=monitor_thread, daemon=True)
monitoring_thread.start()

# Flask teardown function to release camera
@app.teardown_appcontext
def cleanup_camera(exception=None):
    logging.info("[INFO] Releasing camera resources.")
    camera.release()

# Define a simple home route
@app.route('/')
def home():
    return jsonify(settings)

# Define API route
@app.route('/api/data', methods=['GET'])
def get_data():
    with cache_lock:  # Ensure thread safety
        if cached_data is None:
            return jsonify({"error": "Cache not initialized"}), 503  # HTTP 503 Service Unavailable
        return jsonify(cached_data)

# Start the server
if __name__ == '__main__':
    try:
        logging.info("[INFO] Starting Flask server on port 8000...")
        app.run(host='0.0.0.0', port=8000)
    except KeyboardInterrupt:
        logging.info("[INFO] Server shutting down.")
    finally:
        cleanup_camera()
