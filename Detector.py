import cv2
import numpy as np

class Detector:
    def __init__(self, area_threshold=30, brightness_threshold=200, timeout=15, proximity_threshold=50):
        """Initialize with thresholds for area, brightness, timeout, and proximity."""
        self.area_threshold = area_threshold
        self.brightness_threshold = brightness_threshold
        self.timeout = timeout
        self.proximity_threshold = proximity_threshold
        self.corner_threshold = brightness_threshold 
        self.debounce_cards = {}  # Stores card locations and their timeout
        self.next_id = 0  # Unique ID counter for each detected card
        self.corner_approx_accuracy = 0.02

    def _is_card_shape(self, contour):
        """Check if contour is rectangular and has a mostly white/light grey fill."""
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        area = cv2.contourArea(contour)

        if len(approx) == 4 and area > self.area_threshold:
            return True
        return False

    def _is_card_color(self, contour, frame):
        """Check if the contour area is mostly white or light grey."""
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        mean_brightness = cv2.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), mask=mask)[0]
        return mean_brightness > self.brightness_threshold

    def _get_contour_center(self, contour):
        """Calculate the center of the contour."""
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return (cX, cY)
        return None

    def _rotate_to_vertical(self, contour, frame):
        """Rotate the card to a vertical orientation and return the rotated card image."""
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = box.astype(int)  # Convert box points to integer type
        width, height = int(rect[1][0]), int(rect[1][1])
        angle = rect[2]

        # Ensure the card is oriented with the short side on top
        if width > height:
            angle += 90

        # Rotate the card to a vertical orientation
        M = cv2.getRotationMatrix2D(rect[0], angle, 1.0)
        rotated_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        card = cv2.getRectSubPix(rotated_frame, (width, height), rect[0])

        return card


    def _extract_corner(self, card_image):
        """Find the perimeter and approximate corner points of the card."""
        # Convert to grayscale and apply Canny edge detection
        gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter by area to detect large shapes, like a card
            area = cv2.contourArea(contour)
            if area > self.area_threshold:
                # Approximate contour to reduce points for corner approximation
                epsilon = self.corner_approx_accuracy * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # If the approximation has 4 points, it’s likely the card's corners
                if len(approx) == 4:
                    # Extract the corner points
                    corners = approx.reshape(4, 2)  # Shape as (4,2) array for easy access
                    return corners

        return None  # If no card-like contour is found
    def extract_corner_with_outline(self, card_image):
        """Extract the top-left corner of the card and add a blue outline."""
        # Extract the top-left corner as done previously
        gray_card = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
        corner_region = gray_card[5:60, 5:40]
        
        # Convert corner to color (BGR) to allow color outline
        corner_with_outline = cv2.cvtColor(corner_region, cv2.COLOR_GRAY2BGR)
        
        # Add a blue outline around the corner
        outline_color = (255, 0, 0)  # Blue in BGR
        thickness = 2  # Outline thickness
        cv2.rectangle(corner_with_outline, (0, 0), (corner_with_outline.shape[1] - 1, corner_with_outline.shape[0] - 1), outline_color, thickness)
        
        return corner_with_outline

    def display_corner_in_center(self, debounced_card_data):
        """
        Display the outlined top-left corner in the center of the card
        based on the debounced card data.
        """
        # Extract the card image from the debounced data
        card_image = debounced_card_data.get("image")
        if card_image is None:
            raise ValueError("Debounced card data must include an 'image' key with the card image.")

        # Extract the outlined corner
        outlined_corner = self.extract_corner_with_outline(card_image)
        
        # Determine center position to place the corner image
        card_h, card_w = card_image.shape[:2]
        corner_h, corner_w = outlined_corner.shape[:2]
        y_offset = (card_h - corner_h) // 2
        x_offset = (card_w - corner_w) // 2
        
        # Place the outlined corner image onto the card's center
        card_image[y_offset:y_offset + corner_h, x_offset:x_offset + corner_w] = outlined_corner
        return card_image
    def cards(self, frame):
        """Detect, debounce, and process card locations in the provided frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_cards = []

        # Filter contours to identify cards
        for contour in contours:
            if self._is_card_shape(contour) and self._is_card_color(contour, frame):
                center = self._get_contour_center(contour)
                if center:
                    current_cards.append((contour, center))

        # Debounce logic to track persistent card locations
        for contour, center in current_cards:
            found_match = False

            for card_id, card_data in list(self.debounce_cards.items()):
                prev_center = card_data['center']
                dist = np.linalg.norm(np.array(center) - np.array(prev_center))

                if dist < self.proximity_threshold:
                    self.debounce_cards[card_id]['center'] = center
                    self.debounce_cards[card_id]['timeout'] = self.timeout
                    found_match = True
                    break

            # If no match found, add a new card entry
            if not found_match:
                # Rotate card to vertical and extract corner
                card_image = self._rotate_to_vertical(contour, frame)
                corner = self._extract_corner(card_image)
                
                # Store card info with the corner image
                self.debounce_cards[self.next_id] = {
                    'contour': contour,
                    'center': center,
                    'timeout': self.timeout,
                    'corner': corner
                }
                self.next_id += 1

        # Update and remove expired cards
        for card_id, card_data in list(self.debounce_cards.items()):
            if card_data['timeout'] > 0:
                card_data['timeout'] -= 1
            else:
                del self.debounce_cards[card_id]

        # Return debounced card data with corner images
        return self.debounce_cards