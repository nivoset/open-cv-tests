from Card import Card
import matplotlib.pyplot as plt
import cv2
import numpy as np

class Detector:
    def __init__(self, area_threshold=500, brightness_threshold=150):
        """
        Initializes the detector with area and brightness thresholds.
        - area_threshold: Minimum area to consider a contour as a card.
        - brightness_threshold: Minimum average brightness to consider a contour.
        """
        self.area_threshold = area_threshold
        self.brightness_threshold = brightness_threshold

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

    def check_brightness(self, frame, contour) -> bool:
        """Check if the average brightness of the area inside the contour meets the threshold."""
        # Create a mask for the contour
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        # cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Calculate the average brightness inside the mask
        mean_val = cv2.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), mask=mask)[0]
        # print("check brightness", mean_val)
        return mean_val >= self.brightness_threshold

    def filter_contour(self, contour, min_height=200):
    # Get the indices of the highest and lowest points vertically

        # differences = np.diff(contour.squeeze(), axis=0)
        # absolute_distances = np.abs(differences)
        # print("absolute_distances", absolute_distances)
        y_values = contour[:, 0, 1]

        # Calculate the difference between the maximum and minimum Y value
        max_diff_y = np.max(y_values) - np.min(y_values)

        print("Largest difference in Y coordinate:", max_diff_y)

        # Check if the difference between the maximum and minimum Y value is greater than the minimum height
        return max_diff_y > min_height

        
    def cards(self, frame):
        """
        Detect cards in the frame, returning a dictionary of Card objects.
        - frame: Input image.
        - Returns: Dictionary of Card objects detected.
        """
        frame = self.filter_crosshairs(frame)
        blur = cv2.GaussianBlur(frame, (15, 15), 3)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Using Hough Line Transform to detect lines

        # Threshold to isolate dark areas
        _, thresholded = cv2.threshold(gray, 105, 255, cv2.THRESH_BINARY) 
        
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = ([cnt for cnt in contours if cv2.contourArea(cnt) > 150])
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray)
        cv2.drawContours(thresholded, filtered_contours, -1, 255, thickness=cv2.FILLED)
        # union_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # # union_contours = cv2.bitwise_or(union_contours)
            
        # blank_image = np.zeros_like(frame)
        # white_image = np.ones_like(frame)

        # # Calculate the minimum height required for a contour to be considered large enough
        # # Filter the contours based on their size relative to the image height
        # min_height = 75
        # print  ("min height", min_height)
        # filtered_contours = [contour for contour in union_contours if self.filter_contour(contour, min_height)]
        # Draw each contour on the blank image
            

        
        blur = cv2.GaussianBlur(frame, (5,5), 0)
        blur_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV) 

        # create NumPy arrays from the boundaries
        lower = np.array([0,0,0], dtype = "uint8")
        upper = np.array([180,255,40], dtype = "uint8")

        # find the colors within the specified boundaries and apply
        mask = cv2.inRange(blur_hsv, lower, upper)  
        mask = 255 - mask
        # output = cv2.bitwise_and(frame, frame, mask = mask)
        
        
        plt.figure(figsize=(10, 5))
        plt.subplot(7, 1, 2)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        # plt.subplot(1, 3, 2)
        # plt.title("Binary Thresholded")
        # plt.imshow(binary, cmap="gray")
        # plt.axis("off")

        # Iterate over each contour and display the isolated region
        for i, contour in enumerate(filtered_contours):
            try:
                isolated_region = self.extract_card_image(frame, contour)

                # Convert color space for display in matplotlib (from BGR to RGB)
                isolated_region_rgb = cv2.cvtColor(isolated_region, cv2.COLOR_BGR2RGB)

                # Display using matplotlib
                plt.figure(figsize=(5, 5))  # Set the figure size as needed
                plt.imshow(isolated_region_rgb)
                plt.title(f'Hand {i+1}')
                plt.axis('off')  # Turn off axis numbers and ticks
                plt.show()
            except Exception as e:
                print(f"Error processing contour {i}: {e}")

        
        areas = [cv2.contourArea(contour) for contour in filtered_contours]

        # Sort the contours by their areas in descending order
        sorted_contours = sorted(zip(filtered_contours, areas), key=lambda x: x[1], reverse=True)

        # Print the top 5 sizes of contour
        for contour, area in sorted_contours[:6]:
            print(f"Contour area: {area}")

        # detected_cards = {}

        # # Process each contour to detect cards
        # for idx, contour in enumerate(contours):
        #     if cv2.contourArea(contour) > self.area_threshold:
        #         corners = self.find_card_corners(contour)
        #         if corners is not None:
        #             card_image = self.extract_card_image(frame, corners)
        #             card_data = {
        #                 "image": card_image,
        #                 "frame": frame,
        #                 "corners": corners
        #             }
        #             card_obj = Card(card_data)
        #             detected_cards[idx] = card_obj

        # return detected_cards


    def filter_crosshairs(self, image, brightness_threshold=175):

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
                fixed[i, :] = fixed[i-1, :]
        # Get the number of columns in the image
        num_columns = gray.shape[1]

        # Analyze the brightness and replace columns as needed
        for j in range(1, num_columns):  # Start from 1 to avoid indexing error on the first column
            column = gray[:, j]  # Extract the column
            average_brightness = np.mean(column)  # Calculate the average brightness of the column
            if average_brightness > brightness_threshold:
                fixed[:, j] = fixed[:, j-1] 

        return fixed