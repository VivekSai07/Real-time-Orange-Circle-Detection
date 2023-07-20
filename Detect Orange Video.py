import cv2
import numpy as np

def detect_orange_circle(video_path, output_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    # Define the codec for the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create VideoWriter object to save the output video
    output_video = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    while True:
        # Read a frame from the video
        ret, frame = video.read()
        
        # Break the loop if no frame is captured
        if not ret:
            break
    
        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define the lower and upper bounds for the orange color
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([20, 255, 255])
        
        # Threshold the frame to get only the orange pixels
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
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            
            # Add text on top of the detected circle
            cv2.putText(frame, "Circle", (int(x - radius), int(y - radius - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Write the frame with the detected circle to the output video
        output_video.write(frame)
        
        # Display the frame with the detected circle
        cv2.imshow("Orange Circle Detection", frame)
        
        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture and close all windows
    video.release()
    output_video.release()
    cv2.destroyAllWindows()

# Provide the path to your video file and the desired output video path
video_path = "rumbleverse all posiitives Good triggers.mp4"
output_path = r"C:/Users/vivek/Downloads/Orange_Object_Detection_with_OpenCV-master/Orange_Object_Detection_with_OpenCV-master/output_video.mp4"
detect_orange_circle(video_path, output_path)
