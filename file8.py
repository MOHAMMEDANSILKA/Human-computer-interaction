import cv2
import mediapipe as mp
import pyautogui
import serial
import time

# Function to detect thumb and index finger touching and holding
def is_touch_hold(hand_keypoints, touch_thresh=0.04):
    thumb_top_y = hand_keypoints.landmark[4].y
    index_top_y = hand_keypoints.landmark[8].y
    thumb_index_dist = abs(thumb_top_y - index_top_y)
    return thumb_index_dist < touch_thresh

# Initialize VideoCapture and MediaPipe Hands
cap = cv2.VideoCapture(1)
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

        # Smooth cursor movement using exponential smoothing
        if prev_hand_pos is not None:
            smoothing_factor = 0.2  # Adjust the smoothing factor
            current_hand_pos = (int(prev_hand_pos[0] * (1 - smoothing_factor) + current_hand_pos[0] * smoothing_factor),
                                int(prev_hand_pos[1] * (1 - smoothing_factor) + current_hand_pos[1] * smoothing_factor))

        # Update the mouse cursor position
        pyautogui.moveTo(current_hand_pos[0], current_hand_pos[1], duration=0.01)  # Adjust the duration for smoother movement
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
