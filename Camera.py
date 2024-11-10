import cv2
from Detector import Detector

class Camera:
    def __init__(self, camera_index=0):
        """Initialize the camera with a given index (default is 0 for the primary camera)."""
        self.cap = cv2.VideoCapture(camera_index)
        
        # Check if the camera opened successfully
        if not self.cap.isOpened():
            raise ValueError("Camera could not be opened.")

    def set_gain(self, gain_value):
        """Set the camera gain."""
        self.cap.set(cv2.CAP_PROP_GAIN, gain_value)

    def set_white_balance(self, blue_u, red_v):
        """Set the white balance for blue and red channels.
        
        Note: Some cameras may not support manual white balance control.
        """
        self.cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, blue_u)
        self.cap.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V, red_v)

    def set_brightness(self, brightness_value):
        """Set the camera brightness."""
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness_value)

    def set_contrast(self, contrast_value):
        """Set the camera contrast."""
        self.cap.set(cv2.CAP_PROP_CONTRAST, contrast_value)

    def read_frame(self):
        """Read a single frame from the camera."""
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Failed to capture image from camera.")
        return frame

    def release(self):
        """Release the camera resource."""
        self.cap.release()

# Example usage
if __name__ == "__main__":
    # Initialize the Camera and Detector
    camera = Camera()
    detector = Detector(brightness_threshold=125)

    try:
        while True:
            frame = camera.read_frame()
            debounced_cards = detector.cards(frame)

            # Draw each debounced card and display corner in the middle of each card
            for card_id, card_data in debounced_cards.items():
                # print(card_data.extract_corner_with_outline())
                card_data.overlay_corner_on_frame(frame)
                # contour = card_data['contour']
                # center = card_data['center']
                # corner = card_data['corner']
                
                # # Draw contour on the main frame
                # cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)
                # updated_card_image = detector.display_corner_in_center(card_data)

                # # Display or save the result
                # cv2.imshow("Card with Centered Outlined Corner", updated_card_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # Overlay the corner in the middle of the card
                # if corner is not None:
                #     corner_h, corner_w = corner.shape[:2]
                #     x_offset = center[0] - corner_w // 2
                #     y_offset = center[1] - corner_h // 2
                #     # frame[y_offset:y_offset+corner_h, x_offset:x_offset+corner_w] = corner
                #     if len(corner.shape) == 2:  # Check if `corner` is grayscale
                #       corner = cv2.cvtColor(corner, cv2.COLOR_GRAY2BGR)

                #     # Now `corner` can be assigned to the color `frame` without shape mismatch
                #     frame[y_offset:y_offset+corner_h, x_offset:x_offset+corner_w] = corner

                # Display ID and coordinates
                # cv2.putText(frame, f"ID#{card_id} ({center[0]}, {center[1]})", (center[0], center[1] - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Display the frame with detected cards and corners
            cv2.imshow("Debounced Cards", frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()