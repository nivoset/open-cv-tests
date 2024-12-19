from ultralytics import YOLO
import cv2
import os

imgTest = "path to new image?"
imgAnnot = "path to annotation file?"

img = cv2.imread(imgTest)
imgpredict = img.copy()

model_path = os.path.join("./playing-card-model", "weight", "best.pt")

model = YOLO(model_path)
threshold = 0.5

results = model(img)

print(results)

for result in results.boxes.data.toList():
  x1, y1, x2, y2, score, class_id = result
  if score > threshold:
    cv2.rectangle(imgpredict, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(imgpredict, result.name[int(class_id)].upper(), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    
img_truth = img.copy()

with open(imgAnnot, "r") as file:
  lines = file.readlines()

annotations = []
for line in lines:
  values = line.split()
  label = values[0]
  
  x, y, w, h = map(float, values[1:])
  annotations.append((label, x, y, w, h))
  
for annotation in annotations:
  label, x, t, w, h = annotation
  label_name = results.names[int(label)].upper()
  
  
  x1 = int((x - w / 2) * W)
  y1 = int((y - h / 2) * H)
  x2 = int((x + w / 2) * W)
  y2 = int((y + h / 2) * H)