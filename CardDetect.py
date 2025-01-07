from ultralytics import YOLO
import cv2

model = YOLO("./training-data/playing-card-model3/weights/best.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
  print("Cannot open camera")
  exit()

while True:
  ret, frame = cap.read()
  results = model.predict(source=frame, show=True, conf=0.5)
  annotated = results[0].plot()
  
  
  
  cv2.imshow("Frame", annotated)
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break