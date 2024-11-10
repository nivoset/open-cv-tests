import cv2
import numpy as np

class Card:
    def __init__(self, card_data):
        """
        Initialize a Card instance with detected data.
        card_data should contain:
        - 'image': The cropped image of the card
        - 'frame': The original frame where the card was detected
        - 'corners': The detected corner points of the card in the original frame
        """
        self.card_image = card_data.get("image")
        self.frame = card_data.get("frame")
        self.corners = card_data.get("corners")

        if self.card_image is None or self.frame is None or self.corners is None:
            raise ValueError("Card data must include 'image', 'frame', and 'corners' keys.")

    def extract_corner_with_outline(self):
        """Extract the top-left corner of the card with a blue outline."""
        gray_card = cv2.cvtColor(self.card_image, cv2.COLOR_BGR2GRAY)
        corner_region = gray_card[5:60, 5:40]
        corner_with_outline = cv2.cvtColor(corner_region, cv2.COLOR_GRAY2BGR)
        outline_color = (255, 0, 0)  # Blue outline
        thickness = 2
        cv2.rectangle(corner_with_outline, (0, 0), (corner_with_outline.shape[1] - 1, corner_with_outline.shape[0] - 1), outline_color, thickness)
        return corner_with_outline

    def debug_card_image(self):
        """Overlay the card image with a blue outlined corner back onto the frame, aligned by corners."""
        # Extract and outline the corner for debugging
        outlined_corner = self.extract_corner_with_outline()

        # Place the outlined corner back on the card image for debugging purposes
        card_h, card_w = self.card_image.shape[:2]
        corner_h, corner_w = outlined_corner.shape[:2]
        y_offset = (card_h - corner_h) // 2
        x_offset = (card_w - corner_w) // 2
        debug_card = self.card_image.copy()
        debug_card[y_offset:y_offset + corner_h, x_offset:x_offset + corner_w] = outlined_corner

        # Define the target points based on the detected corners in the frame
        target_pts = np.float32(self.corners)
        card_pts = np.float32([[0, 0], [card_w, 0], [card_w, card_h], [0, card_h]])

        # Compute homography matrix
        matrix, _ = cv2.findHomography(card_pts, target_pts)

        # Warp the card image back onto the frame using the homography matrix
        warped_card = cv2.warpPerspective(debug_card, matrix, (self.frame.shape[1], self.frame.shape[0]))
        overlaid_frame = self.frame.copy()

        # Overlay the warped card onto the frame
        mask = (warped_card > 0).any(axis=2)  # Create a mask to overlay only card area
        overlaid_frame[mask] = warped_card[mask]

        return overlaid_frame
