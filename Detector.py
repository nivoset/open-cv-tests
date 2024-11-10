import cv2
import numpy as np
from Card import Card

class Detector:
    def __init__(self, area_threshold=500, brightness_threshold=200):
        """Initialize with thresholds for area and brightness."""
        self.area_threshold = area_threshold
        self.brightness_threshold = brightness_threshold

    def find_card_corners(self, contour):
        """Find and approximate corner points of the card using contour approximation."""
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:
            corners = approx.reshape(4, 2)
            return corners
        return None

    def extract_card_image(self, frame, corners):
        """Extract the card image from the frame using a perspective transform."""
        card_w, card_h = 200, 300  # Standard card width and height for perspective transform

        # Define points for the perspective transform
        target_pts = np.float32([[0, 0], [card_w, 0], [card_w, card_h], [0, card_h]])
        matrix = cv2.getPerspectiveTransform(np.float32(corners), target_pts)
        card_image = cv2.warpPerspective(frame, matrix, (card_w, card_h))
        return card_image

    def cards(self, frame):
        """Detect cards in the frame and return a dictionary of Card objects."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_cards = {}

        # Iterate through contours to detect card-like shapes
        for idx, contour in enumerate(contours):
            if cv2.contourArea(contour) > self.area_threshold:
                corners = self.find_card_corners(contour)
                if corners is not None:
                    # Extract the card image using perspective transform
                    card_image = self.extract_card_image(frame, corners)
                    
                    # Prepare card data for Card object initialization
                    card_data = {
                        "image": card_image,
                        "frame": frame,
                        "corners": corners
                    }

                    # Create a Card object and add it to the dictionary with a unique ID
                    detected_cards[f"card_{idx}"] = Card(card_data)

        return detected_cards