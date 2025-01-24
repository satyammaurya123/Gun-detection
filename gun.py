import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt

# Load the cascade classifier for gun detection
gun_cascade = cv2.CascadeClassifier('cascade.xml')  # Ensure 'cascade.xml' exists in the same directory

# Initialize video capture
camera = cv2.VideoCapture(0)

# Variables for processing
firstFrame = None

# Create a figure for displaying the video feed
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()

def on_key(event):
    if event.key == 'q':
        plt.close(fig)  # Close the figure when 'q' is pressed

# Connect the key press event to the figure
fig.canvas.mpl_connect('key_press_event', on_key)

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break
    
    # Resize the frame for faster processing
    frame = imutils.resize(frame, width=500)
    
    # Convert to grayscale for the classifier
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect guns using the cascade
    guns = gun_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))
    gun_exist = len(guns) > 0

    # Draw rectangles around detected guns
    for (x, y, w, h) in guns:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the video feed using matplotlib
    ax.clear()  # Clear the previous frame
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.axis('off')  # Hide axes
    plt.pause(0.001)  # Pause to allow the plot to update

    # Check gun existence
    if gun_exist:
        print("Guns detected")
    else:
        print("No guns detected")

# Release resources
camera.release()
plt.close(fig)  # Close the matplotlib figure