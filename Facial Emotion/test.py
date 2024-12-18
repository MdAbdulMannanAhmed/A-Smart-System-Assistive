import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('model_file_30epochs.h5')

# Initialize the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise IOError("Failed to load haarcascade_frontalface_default.xml. Check the file path.")

# Emotion labels corresponding to the model's output
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Open webcam
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture video frame-by-frame
    ret, frame = video.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to grayscale (required for Haar Cascade)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract the face region
        face_img = gray[y:y + h, x:x + w]
        resized = cv2.resize(face_img, (48, 48))  # Resize to the input size expected by the model
        normalized = resized / 255.0  # Normalize pixel values
        reshaped = np.reshape(normalized, (1, 48, 48, 1))  # Reshape to match model's input shape

        # Predict emotion
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        emotion = labels_dict[label]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Put the emotion label on the frame
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (0, 255, 0), -1)
        cv2.putText(frame, emotion, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display the video frame with emotion labels
    cv2.imshow("Emotion Detection", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
