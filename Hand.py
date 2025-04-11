from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import yaml
from scipy.spatial import distance
from collections import deque, Counter
import numpy as np

# Load the configuration
with open('playing-cards/data.yaml') as f:
    config = yaml.safe_load(f)


class Hand:
    def __init__(self, settings):
        """
        Initialize a Hand instance with YOLO model and settings.
        """
        print("Card settings", settings)
        self.settings = settings
        self.device = settings["device"]
        self.confidence_threshold = settings["confidence_threshold"]
        self.debug = settings.get("debug", False)
        self.get_hand = YOLO("playing-cards/playing-card-model/best.pt", verbose=False)
        self.tracked_cards = {}  # Track cards across frames
        self.card_history = {}  # history buffer per card_id

    def format_hand(self, hand):
        """
        Format the hand data to match the required output format.
        Extracts card name, confidence, and position.
        """
        cards = []
        for box in hand:
            class_id = int(box.cls[0])
            conf = box.conf[0].item()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            cards.append({"name": config["names"][class_id].upper(), "confidence": conf, "center": center})
        return cards

    def calculate_order(self, cards):
        """
        Determine the order of cards based on their relative positions.
        """
        if not cards:
            return []

        # Extract card centers
        centers = np.array([card["center"] for card in cards])

        # Start with the card closest to the top-left corner
        start_idx = np.argmin(centers[:, 0] + centers[:, 1])  # Top-left heuristic
        ordered_indices = [start_idx]
        remaining_indices = set(range(len(centers))) - {start_idx}

        # Build the order based on closest distances
        current_idx = start_idx
        while remaining_indices:
            distances = [
                (idx, distance.euclidean(centers[current_idx], centers[idx]))
                for idx in remaining_indices
            ]
            next_idx = min(distances, key=lambda x: x[1])[0]
            ordered_indices.append(next_idx)
            remaining_indices.remove(next_idx)
            current_idx = next_idx

        # Return ordered cards
        return [cards[i] for i in ordered_indices]

    def update_tracked_cards(self, detected_cards):
        max_history = 5
        stability_threshold = 3
        position_threshold = 50

        updated_tracked = {}

        for detected in detected_cards:
            # class_id = int(detected.cls[0])
            # confidence = detected.conf[0].item()
            # x1, y1, x2, y2 = map(int, detected.xyxy[0])
            class_id = detected["name"]
            confidence = detected["confidence"]
            center = detected["center"]

            # center = ((x1 + x2) // 2, (y1 + y2) // 2)

            matched_id = None

            # Check proximity for existing card match
            for card_id, card_data in self.tracked_cards.items():
                dist = distance.euclidean(card_data["center"], center)
                if dist < position_threshold:
                    matched_id = card_id
                    break

            if matched_id:
                # Update existing card history
                if matched_id not in self.card_history:
                    self.card_history[matched_id] = deque(maxlen=max_history)
                self.card_history[matched_id].append(class_id)

                # Debounce based on majority vote
                most_common = Counter(self.card_history[matched_id]).most_common(1)[0]
                if most_common[1] >= stability_threshold:
                    updated_tracked[matched_id] = {
                        "class_id": most_common[0],
                        "confidence": confidence,
                        "center": center,
                        "stability": most_common[1]
                    }
            else:
                # New card detected
                new_id = len(self.tracked_cards) + 1
                self.card_history[new_id] = deque([class_id], maxlen=max_history)
                updated_tracked[new_id] = {
                    "class_id": class_id,
                    "confidence": confidence,
                    "center": center,
                    "stability": 1
                }

        # Keep only stable cards
        self.tracked_cards = {
            cid: data for cid, data in updated_tracked.items()
            if data["stability"] >= stability_threshold
        }
        
        
    def from_hand(self, card_images):
        """
        Process hand images to extract card information.
        """
        hands = []
        extracted_images = []

        for hand in card_images:
            rotated = cv2.rotate(hand, cv2.ROTATE_180)
            results = self.get_hand(
                source=rotated,
                device=self.device,
                show=self.debug,
                conf=self.confidence_threshold,
                verbose=False
            )

            for result in results:
                detected_cards = result.boxes
                cards = self.format_hand(detected_cards)

                # Determine card order
                ordered_cards = self.calculate_order(cards)

                self.update_tracked_cards(ordered_cards)

                if self.debug:
                    for card in ordered_cards:
                        center = card["center"]
                        class_id = card["name"]
                        cv2.circle(rotated, center, 5, (255, 255, 0), -1)
                        cv2.putText(
                            rotated,
                            f"{card['name']}",
                            (center[0] - 10, center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2
                        )
                        extracted_images.append(rotated)

                hands.append(ordered_cards)

        if self.debug:
            plt.figure(figsize=(10, 5))
            for i, image in enumerate(extracted_images):
                plt.subplot(1, len(extracted_images), i + 1)
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.title(f'Hand {i + 1}')
                plt.axis('off')
            plt.show()

        return hands
