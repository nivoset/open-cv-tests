from ultralytics import YOLO
import yaml

import pathlib

# Find the best weights file
best_weights_file = None
best_epoch = 0

for folder in pathlib.Path("training-data").glob("playing-card-model??"):
    folder_name_parts = folder.name.split("model")
    epoch = int(folder_name_parts[-1])
    weights_files = list(folder.rglob("*.pt"))
    if weights_files:
        if epoch > best_epoch:
            best_epoch = epoch
            best_weights_file = weights_files[0]

# Load the model with the best weights
if best_weights_file is not None:
    print("using ", best_weights_file)
    model = YOLO(best_weights_file)
else:
    print("No weights file found")


# Load the model with the best weights

data_yaml_file = "/Users/benjaminkoop/Desktop/code/python/OpenCV-Playing-Card-Detector/training-data/data.yaml"

with open(data_yaml_file, 'r') as file:
  data = yaml.safe_load(file)
 
def main():
  #load the model
  # model = YOLO("training-data/playing-card-model22/weights/best.pt")
  
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