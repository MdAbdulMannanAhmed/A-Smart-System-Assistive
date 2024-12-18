import cv2
import torch
import pyttsx3

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # Speed of speech

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Use yolov5s (small version)

# Function to announce detected objects
def announce_objects(objects):
    if objects:
        announcement = "I see " + ", ".join(objects)
        engine.say(announcement)
        engine.runAndWait()

# Open webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (required by YOLO)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection
    results = model(rgb_frame)

    # Parse the results
    detections = results.pandas().xyxy[0]  # Get detections as a pandas DataFrame
    detected_objects = list(detections['name'])

    # Display detections on the frame
    for index, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name']
        confidence = row['confidence']

        # Draw rectangle and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the video feed
    cv2.imshow("Object Detector", frame)

    # Announce objects when 'c' is pressed
    key = cv2.waitKey(1)
    if key == ord("c"):
        announce_objects(detected_objects)
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
