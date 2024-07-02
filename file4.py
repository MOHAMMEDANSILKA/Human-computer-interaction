import cv2
import mediapipe as mp
import pyautogui

# Function to detect thumb and index finger touching and holding
def is_touch_hold(hand_keypoints):
    # Threshold value for touch detection
    touch_thresh = 0.04  # Adjust as needed
    
    # Calculate distances between the top points of thumb and index finger for touch detection
    thumb_top_y = hand_keypoints.landmark[4].y
    index_top_y = hand_keypoints.landmark[8].y
    thumb_index_dist = abs(thumb_top_y - index_top_y)
    
    # Return True if thumb and index finger are touching and close enough (touch and hold gesture)
    return thumb_index_dist < touch_thresh

# Function to draw top points of thumb and index finger and display distance
def draw_thumb_index_top_points(frame, hand_keypoints):
    thumb_top_x = int(hand_keypoints.landmark[4].x * frame.shape[1])
    thumb_top_y = int(hand_keypoints.landmark[4].y * frame.shape[0])
    index_top_x = int(hand_keypoints.landmark[8].x * frame.shape[1])
    index_top_y = int(hand_keypoints.landmark[8].y * frame.shape[0])
    
    # Draw circles at the top points of thumb and index finger
    cv2.circle(frame, (thumb_top_x, thumb_top_y), 5, (65, 176, 110), -1)
    cv2.circle(frame, (index_top_x, index_top_y), 5, (242, 123, 189), -1)
    
    return frame

# Initialize VideoCapture and MediaPipe Hands
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Main loop
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame to detect hands
    results = hands.process(rgb_frame)
    
    # If hands are detected
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Draw top points of thumb and index finger
        frame = draw_thumb_index_top_points(frame, hand_landmarks)
        
        # Get coordinates of thumb and index finger tips
        thumb_tip_x = int(hand_landmarks.landmark[4].x * frame.shape[1])
        thumb_tip_y = int(hand_landmarks.landmark[4].y * frame.shape[0])
        index_tip_x = int(hand_landmarks.landmark[8].x * frame.shape[1])
        index_tip_y = int(hand_landmarks.landmark[8].y * frame.shape[0])
        
        # Move the mouse cursor to the position of the thumb tip
        pyautogui.moveTo(index_tip_x, index_tip_y)
        
        # Check for touch and hold gesture
        if is_touch_hold(hand_landmarks):
            # Perform left mouse click when touch and hold is detected
            pyautogui.mouseDown()
        else:
            # Release left mouse button when touch and hold is released
            pyautogui.mouseUp()
        
        # Draw hand landmarks on the frame
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow("Hand Gesture Control", frame)
    
    # Exit loop if 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
