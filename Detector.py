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
        """Extract the corner from the top-left of the card image and remove background based on brightness."""
        gray_card = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)

        # Define initial region for corner extraction
        initial_corner_region = gray_card[5:60, 5:40]
        mask = np.zeros_like(initial_corner_region, dtype=np.uint8)

        # Identify the boundaries by checking brightness
        rows, cols = initial_corner_region.shape
        for y in range(rows):
            for x in range(cols):
                if initial_corner_region[y, x] > self.corner_threshold:
                    mask[y, x] = 255  # Mark as part of the corner (white)

        # Find the bounding box of the area above the threshold
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            cropped_corner = card_image[y:y+h+5, x:x+w+5]  # Add padding for a clear view of rank/suit
        else:
            cropped_corner = initial_corner_region  # Default if no contour found

        return cropped_corner

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