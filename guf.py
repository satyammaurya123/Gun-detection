import numpy as np  # For mathematical calculations
import cv2  # OpenCV for computer vision tasks
import imutils  # For image processing
import datetime  # For timestamps

# Load the pre-trained gun cascade
gun_cascade = cv2.CascadeClassifier('cascade.xml')

# Access the camera
camera = cv2.VideoCapture(0)

# Initialize variables
firstframe = None
gun_exist = False

# Infinite loop for continuous detection
while True:
    ret, frame = camera.read()
    
    # Check if the frame is captured properly
    if not ret:
        print("Failed to capture video frame.")
        break

    # Resize the frame for faster processing
    frame = imutils.resize(frame, width=500)

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect guns in the frame using the cascade
    gun = gun_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))

    # Check if any guns are detected
    if len(gun) > 0:
        gun_exist = True
    else:
        gun_exist = False

    # Draw rectangles around detected guns
    for (x, y, w, h) in gun:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y: y + h, x: x + w]
        roi_colour = frame[y: y + h, x: x + w]

    # Initialize the first frame for reference (motion detection, if needed)
    if firstframe is None:
        firstframe = gray
        continue

    # Show the security feed
    cv2.imshow("Security Feed", frame)

    # Break the loop if 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # Print gun detection status
    if gun_exist:
        print("Gun detected!")
    else:
        print("No guns detected.")

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()