from Hand import Hand
import matplotlib.pyplot as plt
import cv2
import numpy as np

class Detector:
    def __init__(self, brightness_threshold=200, epsilon_factor=0.02, min_area=150, sort_method="left-to-right", debug = False):
        self.brightness_threshold = brightness_threshold
        self.epsilon_factor = epsilon_factor
        self.min_area = min_area
        self.sort_method = sort_method
        self.debug = debug

    def find_card_corners(self, contour, epsilon_factor=0.02):
        """
        This function takes an image and a contour, approximates the contour to find
        roughly four corners, marks them on the image, and returns both the image and the corner coordinates.

        Parameters:
        - image: The original image on which the contour is found.
        - contour: The contour data as found by cv2.findContours().
        - epsilon_factor: A factor to determine the approximation accuracy (default 0.02).

        Returns:
        - modified_image: The image with corners marked.
        - corners: List of coordinates for the approximated corners.
        """
        # Approximating the contour
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Create a list to hold corner coordinates
        corners = []
        
        # Drawing the corners on the image and storing their coordinates
        for vertex in approx:
            x, y = vertex.ravel()
            corners.append((x, y))
            # cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Green dots

        return corners

    def extract_card_image(self, image, contour):
        """
        This function takes an image and a contour, creates a mask from the contour,
        applies the mask to the image, and returns the isolated image region inside the contour.

        Parameters:
        - image: The source image from which the region is to be extracted.
        - contour: The contour defining the region to be extracted.

        Returns:
        - cropped_image: The isolated image region inside the contour.
        """
        # Create a mask with the same dimensions as the image, initialized to zero (black)
        mask = np.zeros_like(image)

        # Fill the contour on the mask with white
        cv2.fillPoly(mask, [contour], (255, 255, 255))

        # Apply the mask to the image using bitwise AND
        masked_image = cv2.bitwise_and(image, mask)

        # Optionally, crop the masked_image to the bounding rectangle of the contour to reduce size
        x, y, w, h = cv2.boundingRect(contour)
        cropped_image = masked_image[y:y+h, x:x+w]

        return cropped_image

    def sort_contours(self, contours, method="left-to-right"):
        """
        Sort contours according to the method provided ('left-to-right', 'right-to-left',
        'top-to-bottom', 'bottom-to-top'). Default is 'left-to-right'.
        """
        # Create a list of bounding boxes from contours
        boundingBoxes = [cv2.boundingRect(c) for c in contours]
        
        # Sort the bounding boxes, depending on the method
        try:
            if method == "left-to-right":
                contours, boundingBoxes = zip(*sorted(zip(contours, boundingBoxes),
                                                    key=lambda b: b[1][0]))
            elif method == "right-to-left":
                contours, boundingBoxes = zip(*sorted(zip(contours, boundingBoxes),
                                                    key=lambda b: b[1][0], reverse=True))
            elif method == "top-to-bottom":
                contours, boundingBoxes = zip(*sorted(zip(contours, boundingBoxes),
                                                    key=lambda b: b[1][1]))
            elif method == "bottom-to-top":
                contours, boundingBoxes = zip(*sorted(zip(contours, boundingBoxes),
                                                    key=lambda b: b[1][1], reverse=True))
        except:
            print("error sorting cards")
        

        return contours
    def cards(self, frame):
        """
        Detect cards in the frame, returning a dictionary of Card objects.
        - frame: Input image.
        - Returns: Dictionary of Card objects detected.
        """
        frame = self.filter_crosshairs(frame, self.brightness_threshold)
        blur = cv2.GaussianBlur(frame.copy(), (15, 15), 3)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Using Hough Line Transform to detect lines

        # Threshold to isolate dark areas
        _, thresholded = cv2.threshold(gray, 105, 255, cv2.THRESH_BINARY) 
        
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = self.sort_contours([cnt for cnt in contours if cv2.contourArea(cnt) > self.min_area], self.sort_method)
        
        gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray)
        cv2.drawContours(thresholded, filtered_contours, -1, 255, thickness=cv2.FILLED)
            

        
        blur = cv2.GaussianBlur(frame.copy(), (5,5), 0)
        blur_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV) 

        # create NumPy arrays from the boundaries
        lower = np.array([0,0,0], dtype = "uint8")
        upper = np.array([180,255,40], dtype = "uint8")

        # find the colors within the specified boundaries and apply
        mask = cv2.inRange(blur_hsv, lower, upper)  
        mask = 255 - mask
        # output = cv2.bitwise_and(frame, frame, mask = mask)

        extracted_images = [self.extract_card_image(frame, contour) for contour in filtered_contours]
        # Number of subplots required: number of extracted images + 1 for the original
        num_plots = len(extracted_images) + 1

        if self.debug:
            print("Number of extracted images:", len(extracted_images))
            # Create a figure to display the results
            plt.figure(figsize=(15, num_plots * 3))  # Adjust the figure size based on the number of images

            # Show the original image first
            plt.subplot(1, num_plots, 1)
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')
            plt.axis('off')
            
            # Show each extracted image in subsequent subplots
            for i, image in enumerate(extracted_images):
                plt.subplot(1, num_plots, i + 2)  # Position the image in the plot grid
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.title(f'Hand {i + 1}')
                plt.axis('off')

            plt.tight_layout()
            plt.show()
        return extracted_images



    def filter_crosshairs(self, image, brightness_threshold):
        # Edge detection using Canny
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for brightness analysis
        fixed = image.copy()
        # Get the number of rows (height) in the image
        num_rows = gray.shape[0]

        # List to hold the average brightness of each row
        

        # Analyze the brightness of each row
        for i in range(1, num_rows):  # Start from 1 to avoid indexing error on the first row
            row = gray[i, :]  # Extract the row
            average_brightness = np.mean(row)  # Calculate the average brightness of the row
            if average_brightness > brightness_threshold:
                fixed[i, :] = fixed[max(0,i-1), :]
        # Get the number of columns in the image

        num_columns = gray.shape[1]

        # Analyze the brightness and replace columns as needed
        for j in range(1, num_columns):  # Start from 1 to avoid indexing error on the first column
            column = gray[:, j]  # Extract the column
            average_brightness = np.mean(column)  # Calculate the average brightness of the column
            if average_brightness > brightness_threshold:
                fixed[:, j] = fixed[:, max(0, j-1)] 

        return fixed