import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Debounce parameters
debounce_contours = {}
TIMEOUT = 15  # Frames to keep showing the contour if it disappears
PROXIMITY_THRESHOLD = 50  # Distance threshold to consider a contour as the same card

def find_card_contours(image, area_threshold=30, brightness_threshold=155):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    card_contours = []

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        area = cv2.contourArea(contour)

        if len(approx) == 4 and area > area_threshold:
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            mean_brightness = cv2.mean(gray, mask=mask)[0]

            if mean_brightness > brightness_threshold:
                card_contours.append(contour)
    
    return card_contours

def get_contour_center(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)
    return None
  
# Unique ID counter for each new card
next_id = 0

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    
    current_contours = find_card_contours(frame, area_threshold=30, brightness_threshold=155)
    
    # Check current contours against debounced contours
    for contour in current_contours:
        contour_center = get_contour_center(contour)
        if not contour_center:
            continue

        # Check if this contour is near any previously debounced contour
        found_match = False
        for contour_id, data in debounce_contours.items():
            prev_center = data['center']
            dist = np.linalg.norm(np.array(contour_center) - np.array(prev_center))

            if dist < PROXIMITY_THRESHOLD:
                # Update the debounced contour's data
                debounce_contours[contour_id] = {
                    'contour': contour,
                    'center': contour_center,
                    'timeout': TIMEOUT
                }
                found_match = True
                break

        # If no match found, add a new contour to debounce_contours with a unique ID
        if not found_match:
            debounce_contours[next_id] = {
                'contour': contour,
                'center': contour_center,
                'timeout': TIMEOUT
            }
            next_id += 1

    # Draw debounced contours that are still valid
    for contour_id, data in list(debounce_contours.items()):
        if data['timeout'] > 0:
            # Get the rotated bounding rectangle for the contour
            rect = cv2.minAreaRect(data['contour'])
            box = cv2.boxPoints(rect)
            box = box.astype(int)  # Convert box points to integer type
            width_rot, height_rot = rect[1]

            # Determine if the card needs to be rotated
            angle = rect[2]
            if width_rot > height_rot:
                angle += 90  # Rotate by 90 degrees if width > height

            # Calculate rotation matrix and rotate the card
            M = cv2.getRotationMatrix2D(rect[0], angle, 1.0)
            rotated = cv2.warpAffine(frame, M, (width, height))
            card_img = cv2.getRectSubPix(rotated, (int(height_rot), int(width_rot)), rect[0])

            # Draw the rotated card bounding box on the main frame
            cv2.drawContours(frame, [box], -1, (0, 0, 255), 2)
            
            # Display ID and coordinates of the top-left corner of the contour
            top_left_x, top_left_y = box[1][0], box[1][1]
            cv2.putText(frame, f"ID#{contour_id} ({top_left_x}, {top_left_y})", (top_left_x, top_left_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Decrement timeout if the contour wasn't seen this frame
            if contour_id not in [cid for cid, d in debounce_contours.items() if d['center'] == data['center']]:
                data['timeout'] -= 1
        else:
            # Remove expired contours
            del debounce_contours[contour_id]
    
    # Display the frame
    cv2.imshow("Card", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("p"):
        image = frame
        break

cap.release()
cv2.destroyAllWindows()
