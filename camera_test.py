import yaml
from Hand import Hand
from Detector import Detector
from settings import settings
import cv2
import time

with open('playing-cards/data.yaml') as f:
    config = yaml.safe_load(f)
# Initialize detector and card detector
detector = Detector(
    sort_method=settings["sort_method"],
    brightness_threshold=settings["brightness_threshold"],
    epsilon_factor=settings["epsilon_factor"],
    min_area=settings["min_area"],
    debug=settings["debug"],
)
cardDetector = Hand(settings)
# Initialize the camera (0 is the default camera)
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Create a window to display the live feed
cv2.namedWindow("Live Camera Feed")

while True:
    # Capture a frame from the camera
    ret, image = camera.read()
    if not ret:
      print("no frame")
      continue        
    
    # hands = detector.cards(image)
    cards = cardDetector.from_hand([image])
    print(cards)
    # cv2.imshow("Captured Image", image)
    # print(len(hands), len(cards))
    
    # Acquire lock to safely update the cach
    # Display the frame in a window
    cv2.imshow("Live Camera Feed", image)

    # Exit the live feed on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting live feed.")
        break

# Release the camera and close the window
camera.release()
cv2.destroyAllWindows()