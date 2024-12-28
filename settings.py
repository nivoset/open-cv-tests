
import yaml

# Load the image
image_path = 'image001.png'

# settings = yaml.load(open('./settings.yaml'), Loader=yaml.FullLoader)
with open('./settings.yaml') as f:
    raw_settings = yaml.safe_load(f)

settings = {
    "sort_method": raw_settings.get("sort_method", "left-to-right"),
    "brightness_threshold": raw_settings.get("brightness_threshold", 150),
    "epsilon_factor": raw_settings.get("epsilon_factor", 0.02),
    "min_area": raw_settings.get("min_area", 500),
    "debug": raw_settings.get("debug",  False),
    "input_device": raw_settings.get("input_device", 0),
    "confidence_threshold": raw_settings.get("confidence_threshold", 50),
    "device": raw_settings.get("device", "cpu")
}
