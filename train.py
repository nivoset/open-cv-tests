from ultralytics import YOLO
import cv2
import yaml

image_path = "/Users/benjaminkoop/Desktop/code/python/OpenCV-Playing-Card-Detector/training-data/train/images/000056694_jpg.rf.132f49ccfd4cc9e72ad7ceb43d845090.jpg"
imgAnot = "/Users/benjaminkoop/Desktop/code/python/OpenCV-Playing-Card-Detector/training-data/train/labels/000056694_jpg.rf.132f49ccfd4cc9e72ad7ceb43d845090.txt"

data_yaml_file = "/Users/benjaminkoop/Desktop/code/python/OpenCV-Playing-Card-Detector/training-data/data.yaml"

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
  model = YOLO("training-data/playing-card-model10/weights/best.pt")
  
  project = "./training-data"
  experiment = "playing-card-model"
  
  batch_size = 32
  
  model.train(
    data=data_yaml_file,
    epochs=50, # 50?
    project=project,
    name=experiment,
    batch=batch_size,
    device='mps', # todo can this be changed?
    patience=5,
    imgsz=640,
    verbose=True,
    val=True)
  
  
main()