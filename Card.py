from ultralytics import YOLO
import yaml

# Path to the config file
config_path = 'playing-cards/data.yaml'  # Adjust the path and model name as needed

# Load the configuration
with open(config_path) as f:
    config = yaml.safe_load(f)

print(config)

class Card:
    def __init__(self, settings):
        """
        Initialize a Card instance with detected data.
        card_data should contain:
        - 'image': The cropped image of the card
        - 'frame': The original frame where the card was detected
        - 'corners': The detected corner points of the card in the original frame
        """
        print("card settings", settings)
        self.settings = settings
        self.device = settings["device"]
        self.confidence_threshold = settings["confidence_threshold"]
        self.debug = settings["debug"]
        self.get_hand = YOLO("playing-cards/playing-card-model/weights/best.pt")
    def format_hand(self, hand):
        """
        Format the hand data to match the Card class.
        takes in the result set and pulls out the card name/rank and confidence
        """
        cards = []
        for box in hand:
            # x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0])
            conf = box.conf[0].item()
            cards.append((config["names"][int(class_id)].upper(), conf))
            print(config["names"][int(class_id)].upper(), conf)
        return cards
    def from_hand(self, card_images):
        hands = []
        for hand in card_images:
            results =(self.get_hand(source=hand, device=self.device, show=self.debug))
        # Visualize the results
            for result in results:
                boxes = result.boxes
                hands.append(self.format_hand(boxes))
                for box in boxes:
                    # x1, y1, x2, y2 = box.xyxy[0].tolist()
                    class_id = int(box.cls[0])
                    conf = box.conf[0].item()
                    print(config["names"][int(class_id)].upper(), conf)
        return hands

