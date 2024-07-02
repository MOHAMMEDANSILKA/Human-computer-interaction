import cv2
import mediapipe as mp
import pyautogui
import serial
import numpy as np
from collections import deque
import tensorflow as tf

# Check if GPU is available and set memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Function to detect thumb and index finger touching and holding
def is_touch_hold(hand_keypoints, touch_thresh=0.04):
    thumb_top_y = hand_keypoints.landmark[4].y
    index_top_y = hand_keypoints.landmark[8].y
    thumb_index_dist = abs(thumb_top_y - index_top_y)
    return thumb_index_dist < touch_thresh

# Kalman filter for smoothing hand position
class KalmanFilter:
    def __init__(self, process_noise=0.1, measurement_noise=0.1):
        self.state = np.array([0, 0, 0, 0])  # [x, y, vx, vy]
        self.transition_matrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.observation_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.process_noise_covariance = np.eye(4) * process_noise
        self.measurement_noise_covariance = np.eye(2) * measurement_noise
        self.error_covariance = np.eye(4)

    def update(self, measurement):
        # Predict
        self.state = np.dot(self.transition_matrix, self.state)
        self.error_covariance = np.dot(np.dot(self.transition_matrix, self.error_covariance), self.transition_matrix.T) + self.process_noise_covariance

        # Update
        kalman_gain = np.dot(np.dot(self.error_covariance, self.observation_matrix.T),
                            np.linalg.inv(np.dot(np.dot(self.observation_matrix, self.error_covariance), self.observation_matrix.T) + self.measurement_noise_covariance))
        self.state = self.state + np.dot(kalman_gain, measurement - np.dot(self.observation_matrix, self.state))
        self.error_covariance = (np.eye(4) - np.dot(kalman_gain, self.observation_matrix)) @ self.error_covariance

        return self.state[:2]

# Initialize VideoCapture and MediaPipe Hands
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)  # Set the frame rate to 60 FPS
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize serial communication with ESP32
try:
    ser = serial.Serial('COM3', 9600)  # Replace 'COM3' with the correct port
except serial.SerialException as e:
    print("Error:", e)
    exit()

# Previous hand position for smoothing
prev_hand_pos = None
hand_positions = deque(maxlen=10)  # Store the last 10 hand positions for smoothing
kalman_filter = KalmanFilter(process_noise=0.1, measurement_noise=0.1)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Calculate the current hand position
        thumb_tip_x = int(hand_landmarks.landmark[4].x * frame.shape[1])
        thumb_tip_y = int(hand_landmarks.landmark[4].y * frame.shape[0])
        index_tip_x = int(hand_landmarks.landmark[8].x * frame.shape[1])
        index_tip_y = int(hand_landmarks.landmark[8].y * frame.shape[0])
        current_hand_pos = (index_tip_x, index_tip_y)

        # Store the current hand position in the deque
        hand_positions.append(current_hand_pos)

        # Smooth cursor movement using Kalman filtering
        smoothed_hand_pos = kalman_filter.update(np.array([current_hand_pos[0], current_hand_pos[1]]))
        smoothed_hand_pos = (int(smoothed_hand_pos[0]), int(smoothed_hand_pos[1]))

        # Adjust cursor speed based on hand movement
        hand_movement_speed = np.linalg.norm(np.array(current_hand_pos) - np.array(prev_hand_pos)) if prev_hand_pos else 0
        cursor_speed_factor = 1 + 0.5 * (hand_movement_speed / 100)  # Adjust the speed factor as needed
        pyautogui.moveTo(smoothed_hand_pos[0], smoothed_hand_pos[1], duration=0.01 / cursor_speed_factor)  # Adjust the duration for smoother movement

        prev_hand_pos = current_hand_pos

        if is_touch_hold(hand_landmarks):
            pyautogui.mouseDown()
        else:
            pyautogui.mouseUp()

        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if ser.in_waiting > 0:
        serial_data = ser.readline().decode().strip()
        print("Received:", serial_data)
        if serial_data == "scroll_up":
            pyautogui.scroll(1)  # Scroll up
        elif serial_data == "scroll_down":
            pyautogui.scroll(-1)  # Scroll down

    cv2.imshow("Hand Gesture Control", frame)

    if cv2.waitKey(1) == 27:
        break

# Release resources
cap.release()
ser.close()
cv2.destroyAllWindows()
