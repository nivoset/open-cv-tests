from ultralytics import YOLO
import cv2
import yaml
import time

image_path = "C:/Users/nivos/code/python/card-opencv/playing-cards/train/images/019951084_jpg.rf.0f55beb39575c23186198ea63166504b.jpg"
imgAnot = "C:/Users/nivos/code/python/card-opencv/playing-cards/train/labels/019951084_jpg.rf.0f55beb39575c23186198ea63166504b.txt"

data_yaml_file = "./playing-cards/data.yaml"

with open(data_yaml_file, 'r') as file:
  data = yaml.safe_load(file)
  
label_names = data["names"]
print(label_names)

def read_image():
  print('reading')
  img = cv2.imread(image_path)
  H, W, _ = img.shape

  with open(imgAnot, "r") as file:
    lines = file.readlines()
    
  annotations = []
  for line in lines:
    values = line.split()
    label = values[0]
    
    x, y, w, h = map(float, values[1:])
    annotations.append((label, x, y, w, h))
    
  print(annotations)

  for annotation in annotations:
    label, x, t, w, h = annotation
    label_name = label_names[int(label)]
    # convert yolo coordinates to pixel coordinates
    print(x, w, H, W)
    x1 = int((x - w / 2) * W)
    y1 = int((y - h / 2) * H)
    x2 = int((x + w / 2) * W)
    y2 = int((y + h / 2) * H)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
  cv2.imshow("Image", img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
def main():
  # print('waiting...')
  # time.wait(600000)
  # print('go time')
  #load the model
  model = YOLO("yolov8n.pt")
  
  project = "./playing-cards"
  experiment = "playing-card-model"
  
  batch_size = 8
  
  model.train(
    dnn=True,
    data=data_yaml_file,
    epochs=50, # 50?
    project=project,
    name=experiment,
    batch=batch_size,
    device=0, # todo can this be changed?
    patience=5,
    imgsz=640,
    verbose=True,
    # weights_only=True,
    val=True)
  
if __name__ == '__main__':
  # freeze_support()
  main()