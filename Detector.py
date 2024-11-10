from Card import Card
import cv2
import numpy as np

class Detector:
    def __init__(self, area_threshold=500, brightness_threshold=200, detection_frames=5, persistence_frames=10):
        """
        Initializes the detector with area and brightness thresholds, detection frames, and persistence frames.
        - area_threshold: Minimum area to consider a contour as a card.
        - brightness_threshold: Minimum average brightness to consider a contour.
        - detection_frames: Number of consecutive frames required to confirm a card.
        - persistence_frames: Number of frames to keep displaying a card after it was last detected.
        """
        self.area_threshold = area_threshold
        self.brightness_threshold = brightness_threshold
        self.detection_frames = detection_frames
        self.persistence_frames = persistence_frames
        self.debounce_cards = {}  # Stores detected card info with frame counters

    def find_card_corners(self, contour):
        """Find and approximate corner points of the card using contour approximation."""
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:
            corners = approx.reshape(4, 2)
            return corners
        return None

    def extract_card_image(self, frame, corners):
        """Extract the card image from the frame, ensure correct orientation, and resize it."""
        card_w, card_h = 350, 500  # Standard card width and height for perspective transform

        # Define points for the perspective transform
        target_pts = np.float32([[0, 0], [card_w, 0], [card_w, card_h], [0, card_h]])

        # Check if the card is horizontally aligned based on the corner positions
        if abs(corners[0][1] - corners[1][1]) > abs(corners[0][0] - corners[1][0]):
            # Rotate corners by 90 degrees clockwise to make the card vertical
            corners = [corners[3], corners[0], corners[1], corners[2]]

        # Perform perspective transform with adjusted corners if needed
        matrix = cv2.getPerspectiveTransform(np.float32(corners), target_pts)
        card_image = cv2.warpPerspective(frame, matrix, (card_w, card_h))

        # Resize the image down if needed, e.g., to half size (optional)
        # scaled_card_image = cv2.resize(card_image, (card_w // 2, card_h // 2), interpolation=cv2.INTER_AREA)

        return card_image

    def check_brightness(self, frame, contour):
        """Check if the average brightness of the area inside the contour meets the threshold."""
        # Create a mask for the contour
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Calculate the average brightness inside the mask
        mean_val = cv2.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), mask=mask)[0]
        return mean_val >= self.brightness_threshold

    def cards(self, frame):
        """Detect and debounce cards in the frame, returning a dictionary of Card objects."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_cards = {}

        # Process each contour to detect cards
        for idx, contour in enumerate(contours):
            if cv2.contourArea(contour) > self.area_threshold and self.check_brightness(frame, contour):
                corners = self.find_card_corners(contour)
                if corners is not None:
                    # Unique key for each card based on corner positions
                    card_key = tuple(corners.ravel())
                    
                    # Check if the card is already tracked in debounce_cards
                    if card_key in self.debounce_cards:
                        card_info = self.debounce_cards[card_key]
                        # Increment detection frames
                        card_info['detected_frames'] += 1
                        # Reset frames remaining since the card is detected in this frame
                        card_info['frames_remaining'] = self.persistence_frames
                    else:
                        # Register new card if not already in debounce_cards
                        card_image = self.extract_card_image(frame, corners)
                        card_data = {
                            "image": card_image,
                            "frame": frame,
                            "corners": corners
                        }
                        card_obj = Card(card_data)
                        self.debounce_cards[card_key] = {
                            'detected_frames': 1,             # Starts detection count
                            'frames_remaining': self.persistence_frames,
                            'card': card_obj
                        }
                    
                    # Add to detected cards if it meets the required detection frames
                    if self.debounce_cards[card_key]['detected_frames'] >= self.detection_frames:
                        detected_cards[card_key] = self.debounce_cards[card_key]['card']

        # Maintain persistence for cards that may not be detected in every frame
        for card_key, card_info in list(self.debounce_cards.items()):
            # If a card was not detected in the current frame
            if card_key not in detected_cards:
                if card_info['frames_remaining'] > 0:
                    # Decrease frames remaining and keep it in detected_cards
                    card_info['frames_remaining'] -= 1
                    detected_cards[card_key] = card_info['card']
                else:
                    # Remove card if it has no frames remaining
                    del self.debounce_cards[card_key]

        return detected_cards
