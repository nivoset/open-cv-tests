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


            # Display the frame with detected cards and corners
            cv2.imshow("Debounced Cards", frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()