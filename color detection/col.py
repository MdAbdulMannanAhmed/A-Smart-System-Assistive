import cv2
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pyttsx3
import threading

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)

# Load the dataset
data = pd.read_csv("newcolors.csv")
X = data[['R', 'G', 'B']].values  # Features: RGB values
y = data['Name'].values      # Labels: Color names

# Train a K-Nearest Neighbors (KNN) classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Function to predict the color name
def predict_color(r, g, b):
    color_name = knn.predict([[r, g, b]])[0]
    return color_name

# Function to announce the color in a separate thread
def announce_color(color_name):
    engine.say(f"The color is {color_name}")
    engine.runAndWait()

# Variables to manage announcements
last_color = None
announce_count = 0
max_announcements = 5
last_announce_time = 0
cooldown_time = 2  # Cooldown period in seconds for TTS (time between announcements)

# Open webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to speed up processing
    frame = cv2.resize(frame, (640, 480))  # Resize to 640x480

    # Draw rectangle for focused region
    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2
    size = 50
    cv2.rectangle(frame, (cx - size, cy - size), (cx + size, cy + size), (255, 255, 255), 2)

    # Get average color in the rectangle
    roi = frame[cy - size:cy + size, cx - size:cx + size]
    avg_color = roi.mean(axis=0).mean(axis=0)
    b, g, r = map(int, avg_color)

    # Predict the color using the KNN model
    color_name = predict_color(r, g, b)

    # Display color name on the frame
    cv2.putText(frame, f"Color: {color_name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Check if the color has changed
    if color_name != last_color:
        last_color = color_name
        announce_count = 0  # Reset the counter for a new color

    # Announce color only if it's under the limit and cooldown time has passed
    current_time = cv2.getTickCount() / cv2.getTickFrequency()
    if announce_count < max_announcements and current_time - last_announce_time >= cooldown_time:
        threading.Thread(target=announce_color, args=(color_name,)).start()
        announce_count += 1
        last_announce_time = current_time  # Update last announce time

    # Show the frame
    cv2.imshow("Color Identifier", frame)

    # Break on key press 'q'
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
