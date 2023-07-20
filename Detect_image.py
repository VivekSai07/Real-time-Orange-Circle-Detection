import cv2
import numpy as np

def detect_orange_circle(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the lower and upper bounds for the orange color
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([20, 255, 255])
    
    # Threshold the image to get only the orange pixels
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Apply morphological operations to remove noise and fill in gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours of the orange areas
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (assuming it's the orange circle)
    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)
        
        # Find the center and radius of the circle
        ((x, y), radius) = cv2.minEnclosingCircle(max_contour)
        
        # Draw a circle around the detected orange circle
        cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 0), 2)
        
        # Calculate the corrected y-coordinate for the text
        text_y = int(y - radius - 10)
        
        # Add text on top of the detected circle
        cv2.putText(image, "Circle", (int(x - radius), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Display the original image with the detected circle
    cv2.imshow("Orange Circle Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Provide the path to your input image
image_path = "game.png"
detect_orange_circle(image_path)
