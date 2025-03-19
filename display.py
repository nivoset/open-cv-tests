import cv2
from ultralytics import YOLO

# Load the best training model
model = YOLO("training-data/playing-card-model18/weights/best.pt")  # replace with the actual path to your model

# Open the default camera (camera 0)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Get the frame dimensions
    height, width, _ = frame.shape

    # Run the frame through the model
    results = model(frame)

    # Draw bounding boxes and class labels on the frame
    for result in results:
        frame = result.plot()

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()