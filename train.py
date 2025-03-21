from ultralytics import YOLO
import yaml

data_yaml_file = "/Users/benjaminkoop/Desktop/code/python/OpenCV-Playing-Card-Detector/training-data/data.yaml"

with open(data_yaml_file, 'r') as file:
  data = yaml.safe_load(file)
 
def main():
  #load the model
  model = YOLO("training-data/playing-card-model18/weights/best.pt")
  
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