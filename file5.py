import cv2
import mediapipe as mp

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
    
    # Calculate the distance between the top points
    thumb_index_dist = abs(thumb_top_y - index_top_y)
    
    # Display the distance on the frame
    cv2.putText(frame, f"Distance: {thumb_index_dist}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 30, 70), 2)
    
    # Display the x-axis value in pink color
    cv2.putText(frame, f"X Value: {thumb_top_x}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return frame

# Initialize VideoCapture and MediaPipe Hands
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Initialize variables
start_time = 0
gesture_detected = False

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
        
        # Draw top points of thumb and index finger, display distance, and x-axis value
        frame = draw_thumb_index_top_points(frame, hand_landmarks)
        
        # Check for touch and hold gesture
        if is_touch_hold(hand_landmarks):
            if not gesture_detected:
                print("Touch and hold detected")
                gesture_detected = True
        else:
            if gesture_detected:
                print("Released")
                gesture_detected = False
        
        # Draw hand landmarks on the frame
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow("Hand Gesture Control", frame)
    
    # Exit loop if 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
