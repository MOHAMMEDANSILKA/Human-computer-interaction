import cv2
import mediapipe as mp
import time

def process_frame(frame, face_mesh):
    # Convert the frame to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image
    results = face_mesh.process(image_rgb)

    # Convert the image color back to BGR
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Calculate the x-coordinate of the face midpoint
            x_min = min([landmark.x * frame.shape[1] for landmark in face_landmarks.landmark])
            x_max = max([landmark.x * frame.shape[1] for landmark in face_landmarks.landmark])
            face_midpoint_x = int((x_min + x_max) / 2)

            
            cv2.putText(image_bgr, f'Face X: {face_midpoint_x}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw the face mesh annotations on the image.
            mp_drawing.draw_landmarks(
                image=image_bgr,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

    return image_bgr

# Face mesh detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Failed to open camera.")
    exit()

# Initialize variables for frame processing rate calculation
start_time = time.time()
frame_count = 0

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Process the frame
        processed_frame = process_frame(frame, face_mesh)

        cv2.imshow('MediaPipe FaceMesh', processed_frame)

        frame_count += 1

        # Calculate and display frame processing rate every 5 seconds
        if time.time() - start_time >= 5:
            fps = frame_count / (time.time() - start_time)
            print(f"Frame processing rate: {fps:.2f} FPS")
            start_time = time.time()
            frame_count = 0

        if cv2.waitKey(1) & 0xFF == 27:
            break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
