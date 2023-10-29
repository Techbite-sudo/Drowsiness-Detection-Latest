# Importing OpenCV Library for basic image processing functions
import cv2

# Numpy for array related functions
import numpy as np

# Dlib for deep learning based Modules and face landmark detection
import dlib

# face_utils for basic operations of conversion
from imutils import face_utils

# pygame alert audio meanwhile
from pygame import mixer

# Import the subprocess module at the beginning of my code to execute external commands.
import subprocess


# initializing mixer and loading .wav file
mixer.init()
mixer.music.load("music.wav")

# Initializing the camera and taking the instance
cap = cv2.VideoCapture(0)

# Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize face_frame before the loop
face_frame = None

# Define your desired width and height
desired_width = 640  # Change this to your preferred width
desired_height = 480  # Change this to your preferred height

# status marking for current state
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)


def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist


def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    # Checking if it is blinked
    if ratio > 0.25:
        return 2
    elif ratio > 0.21 and ratio <= 0.25:
        return 1
    else:
        return 0


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Print the original frame's dimensions
    print("Original frame size:", frame.shape)

    # Resize the frame to the desired width and height
    frame = cv2.resize(frame, (desired_width, desired_height))
    gray = cv2.resize(gray, (desired_width, desired_height))

    faces = detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = frame.copy()

        # Print the resized face_frame's dimensions
        print("Resized face_frame size:", face_frame.shape)

        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # The numbers are actually the landmarks which will show eye
        left_blink = blinked(
            landmarks[36],
            landmarks[37],
            landmarks[38],
            landmarks[41],
            landmarks[40],
            landmarks[39],
        )
        right_blink = blinked(
            landmarks[42],
            landmarks[43],
            landmarks[44],
            landmarks[47],
            landmarks[46],
            landmarks[45],
        )

        # Now judge what to do for the eye blinks
        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = ""
                # mixer.music.play()
                subprocess.call(["espeak", "YOU ARE SLEEPING WAKE UP"])

                color = (255, 0, 0)

        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status = "Drowsy !"
                subprocess.call(["espeak", "YOU ARE DROWSY, WAKE UP"])
                # mixer.music.play()
                color = (0, 0, 255)

        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 6:
                status = "Active :)"
                subprocess.call(["espeak", "YOU ARE AWAKE CONGRATULATIONS"])
                color = (0, 255, 0)

        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    cv2.imshow("Frame", frame)
    # Check if face_frame has valid dimensions before displaying
    if face_frame is not None and face_frame.shape[0] > 0 and face_frame.shape[1] > 0:
        cv2.imshow("Result of detector", face_frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
