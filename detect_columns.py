import cv2
from Detector import Detector
from Card import Card
import yaml

# Load the image
image_path = 'image001.png'

settings = yaml.load(open('./settings.yaml'), Loader=yaml.FullLoader)


detector = Detector(
    sort_method=settings["sort_method"],
    area_threshold=settings["area_threshold"],
    brightness_threshold=settings["brightness_threshold"],
    epsilon_factor=settings["epsilon_factor"],
    min_area=settings["min_area"],
    debug=settings["debug"],      
)
cardDetector = Card(settings)

print(settings) 
def process_image_fn(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    hands = detector.cards(image)
    
    cards = cardDetector.from_hand(hands)
    print(cards)





process_image_fn(image_path)