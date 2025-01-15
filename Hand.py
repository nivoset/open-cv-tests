from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import yaml

# Load the configuration
with open('playing-cards/data.yaml') as f:
    config = yaml.safe_load(f)


class Hand:
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
        self.debug = False
        self.get_hand = YOLO("playing-cards/playing-card-model/best.pt", verbose=False)
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
        return cards
    def filter_highest_confidence_of_each_card(self, hand):
        """
        Filter the hand data to keep only the highest confidence for each card.
        """
        found = {}
        filtered_hand = []
        for card in hand:
            name = card.cls[0]
            if (name) not in found:
                found[name] = card.conf
                filtered_hand.append(card)
        return filtered_hand
    

    def from_hand(self, card_images):
        hands = []
        extracted_images = []
        for hand in card_images:
            rotated = cv2.rotate(hand, cv2.ROTATE_180)
            # Get YOLOv8 results
            results =(self.get_hand(
                source=rotated,
                device=self.device,
                show=self.debug,
                conf=self.confidence_threshold,
                verbose=False
            ))
            
            
            # Process and overlay results
            for result in results:
                cards = result.boxes  # Assuming `boxes` is in the YOLOv8 result format
                
                # Sort boxes based on their Y position
                # boxes.sort(key=lambda box: int(box.xyxy[0][1]))
                cards_sorted = sorted(cards, key=lambda box: int(box.xyxy[0][1]), reverse=True)
                
                # Deduplicate based on class_id
                unique_cards = {}
                for box in cards_sorted:
                    class_id = int(box.cls[0])  # Extract class ID as an integer
                    # Check if class_id is not in unique_boxes or the new box has a higher confidence
                    if class_id not in unique_cards or box.conf > unique_cards[class_id].conf:
                        unique_cards[class_id] = box

                # Convert back to a list
                deduped_cards = list(unique_cards.values())
                
                if self.debug:
                    for box in deduped_cards:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract bounding box coordinates
                        class_id = int(box.cls[0])  # Extract class label
                        # confidence = box.conf  # Extract confidence score
                        
                        # Draw the bounding box and label on the image
                        cv2.rectangle(rotated, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Green box
                        cv2.putText(
                            rotated,
                            f"{(config["names"][int(class_id)].upper())}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2
                        )
                        extracted_images.append(rotated)
                # Format the results for hand processing
                hands.append(self.format_hand(deduped_cards))
            
        if self.debug:
            # This code shows a display of the hands next to each other with plotted data
            # # Show the original image first
            plt.subplot(1, 5, 1)
            
            # Show each extracted image in subsequent subplots
            for i, image in enumerate(extracted_images):
                plt.subplot(1, 5, i + 1)  # Position the image in the plot grid
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.title(f'Hand {i + 1}')
                plt.axis('off')

            plt.tight_layout()
            plt.show()
        
        return hands

