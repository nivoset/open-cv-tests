import cv2
import matplotlib.pyplot as plt
from Detector import Detector
# Load the image
image_path = 'image001.png'

detector = Detector()

def process_image_fn(image_path, area_threshold=1000):
    # Load the image
    image = cv2.imread(image_path)
    image_with_blobs = image.copy()
    
    cards = detector.cards(image)
    # for _, card in cards.items():
    #     card.overlay_corner_on_frame(image_with_blobs)
    
    
    # cv2.drawContours(image_with_blobs, contours, -1, (0, 255, 0), 2)
    # detector.draw_groups(image_with_blobs, contours)
    
    
    # # Convert the image to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # # Apply binary thresholding to isolate white regions
    # _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # # Apply morphological operations to smooth and connect blobs
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    # morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # # Find contours in the morphed image
    # contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for contour in contours:
    #     # Approximate the contour to get corners
    #     epsilon = 0.02 * cv2.arcLength(contour, True)  # Adjust the approximation factor
    #     approx = cv2.approxPolyDP(contour, epsilon, True)
        
    #     # Draw the contour and corners on the image
    #     cv2.drawContours(image_with_blobs, [approx], -1, (0, 255, 0), 2)
    #     for point in approx:
    #         x, y = point[0]
    #         cv2.circle(image_with_blobs, (x, y), 5, (0, 0, 255), -1)
    # # Filter contours based on area and aspect ratio
    # # filtered_contours = []
    # # for contour in contours:
    # #     area = cv2.contourArea(contour)
    # #     x, y, w, h = cv2.boundingRect(contour)
        
    # #     # Only keep large, vertically elongated blobs
    # #     # if area > area_threshold and aspect_ratio > 2:
    # #     filtered_contours.append((x, y, w, h))
    
    # #     print('draw', area)
    # # # Draw bounding boxes around the filtered blobs
    # # image_with_blobs = image.copy()
    # # for (x, y, w, h) in filtered_contours:
    # #     cv2.rectangle(image_with_blobs, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # # Visualize the results
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.title("Original Image")
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.axis("off")

    # # plt.subplot(1, 3, 2)
    # # plt.title("Binary Thresholded")
    # # plt.imshow(binary, cmap="gray")
    # # plt.axis("off")

    # plt.subplot(1, 2, 2)
    # plt.title("Filtered Blobs")
    # plt.imshow(cv2.cvtColor(image_with_blobs, cv2.COLOR_BGR2RGB))
    # plt.axis("off")

    # plt.tight_layout()
    # plt.show()




process_image_fn(image_path)