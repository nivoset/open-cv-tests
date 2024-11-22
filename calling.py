from inference import get_model



model = get_model(model_id="playing-cards-ow27d/4")

results = model.infer("https://media.roboflow.com/inference/people-walking.jpg")