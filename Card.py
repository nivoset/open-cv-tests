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
        print(self.corners)
        # print(self.corners)

        if self.card_image is None or self.frame is None or self.corners is None:
            raise ValueError("Card data must include 'image', 'frame', and 'corners' keys.")
    def overlay_corner_on_frame(self, frame):
        """
        Draws the outlined corner from `extract_corner_with_outline()` in the center of the card area on the main frame.
        """
        
        cv2.drawContours(frame, [self.corners], 0, (0, 255, 0), 2)
        # # Extract the outlined corner
        # outlined_corner = cv2.resize(self.extract_aligned_card_image(), (30, 72), interpolation=cv2.INTER_AREA)
        
        # # Get dimensions of the outlined corner
        # corner_h, corner_w = outlined_corner.shape[:2]

        # # Calculate center position of the card in the main frame using its corner coordinates
        # card_center_x = int((self.corners[0][0] + self.corners[2][0]) / 2)
        # card_center_y = int((self.corners[0][1] + self.corners[2][1]) / 2)

        # # Calculate the top-left position to place the outlined corner in the center of the card
        # y_offset = card_center_y - corner_h // 2
        # x_offset = card_center_x - corner_w // 2

        # # Ensure the outlined corner fits within the frame boundaries
        # if 0 <= y_offset < frame.shape[0] - corner_h and 0 <= x_offset < frame.shape[1] - corner_w:
        #     # Overlay the outlined corner onto the frame
        #     frame[y_offset:y_offset + corner_h, x_offset:x_offset + corner_w] = outlined_corner
        
        # return frame
    def extract_corner_with_outline(self):
        """Extract the top-left corner of the card with a blue outline after ensuring correct orientation."""
        # Check if the card image is in landscape orientation
        if self.card_image.shape[1] > self.card_image.shape[0]:  # width > height
            # Rotate 90 degrees counterclockwise to make it portrait
            self.card_image = cv2.flip(cv2.rotate(self.card_image, cv2.ROTATE_90_CLOCKWISE), 1)

        # Convert to grayscale and extract the corner
        gray_card = cv2.cvtColor(self.card_image, cv2.COLOR_BGR2GRAY)
        corner_region = gray_card  # Extract the corner region
        
        # Convert corner to color (BGR) to add a colored outline
        corner_with_outline = cv2.cvtColor(2, cv2.COLOR_GRAY2BGR)
        
        # Add a blue outline around the corner
        outline_color = (255, 0, 0)  # Blue color in BGR
        thickness = 2  # Outline thickness
        cv2.rectangle(corner_with_outline, (0, 0), 
                      (corner_with_outline.shape[1] - 1, corner_with_outline.shape[0] - 1), 
                      outline_color, thickness)
        
        return corner_with_outline
    def extract_aligned_card_image(self):
        """
        Ensures the full card image is vertically aligned (portrait mode).
        If the card is in landscape mode, rotates it to portrait.
        """
        # Check if the card image is in landscape orientation
        # print(self.card_image.shape)
        # if self.card_image.shape[1] > self.card_image.shape[0]:  # width > height
        #     # Rotate 90 degrees counterclockwise to make it portrait
        #     aligned_card_image = cv2.rotate(self.card_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # else:
        #     # If already in portrait, no rotation needed
        aligned_card_image = self.card_image

        return aligned_card_image
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
