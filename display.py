import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model_path = "training-data/playing-card-model18/weights/best.pt"
model = YOLO(model_path)

# Open a video capture (use 0 for webcam or provide a video path)
cap = cv2.VideoCapture(0)  # Change 0 to a file path if using a video

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Perform object detection
    results = model(frame)

    # Draw bounding boxes and labels
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            confidence = box.conf[0].item()  # Confidence score
            class_id = int(box.cls[0].item())  # Class ID

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label with confidence score
            label = f"{model.names[class_id]} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the processed frame
    cv2.imshow("Poker Cards", frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
