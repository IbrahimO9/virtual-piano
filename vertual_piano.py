import cv2
import mediapipe as mp
import time
from play_notes import SoundPlayer

# Different note sets
ORIGINAL_NOTES = ['A4', 'B4', 'C5', 'D5', 'E5']
ALTERNATE_NOTES = ['F4', 'G4', 'A4', 'B4', 'C5']

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Mediapipe hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Sound player
player = SoundPlayer(ORIGINAL_NOTES)

# Track which notes are currently being played
finger_pressed = [False] * 5
last_press_time = [0] * 5  # Track last press time for each finger
PRESS_COOLDOWN = 0.1  # Minimum time between presses (seconds)
current_notes = ORIGINAL_NOTES
last_gesture_time = 0
GESTURE_COOLDOWN = 1.0  # Minimum time between gesture recognition

def is_finger_down(finger_tip, finger_base):
    """Check if finger is pressed down by comparing tip and base positions"""
    vertical_distance = finger_tip[1] - finger_base[1]
    horizontal_distance = abs(finger_tip[0] - finger_base[0])
    return vertical_distance > 20 and horizontal_distance < 50

def is_open_palm(hand_landmarks):
    """Check if hand is in open palm position (all fingers up)"""
    finger_tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    finger_mcps = [
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp_hands.HandLandmark.RING_FINGER_MCP,
        mp_hands.HandLandmark.PINKY_MCP
    ]
    
    for tip, mcp in zip(finger_tips, finger_mcps):
        if hand_landmarks.landmark[tip].y >= hand_landmarks.landmark[mcp].y:
            return False
    return True

def is_closed_fist(hand_landmarks):
    """Check if hand is in closed fist position (all fingers down)"""
    finger_tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    finger_mcps = [
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp_hands.HandLandmark.RING_FINGER_MCP,
        mp_hands.HandLandmark.PINKY_MCP
    ]
    
    for tip, mcp in zip(finger_tips, finger_mcps):
        if hand_landmarks.landmark[tip].y <= hand_landmarks.landmark[mcp].y:
            return False
    return True

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Hand tracking
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks and results.multi_handedness:
        current_time = time.time()
        
        # Process each detected hand
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label
            
            if hand_label == 'Left':
                # Check for left hand gestures
                if current_time - last_gesture_time > GESTURE_COOLDOWN:
                    if is_open_palm(hand_landmarks):
                        current_notes = ALTERNATE_NOTES
                        player = SoundPlayer(current_notes)
                        last_gesture_time = current_time
                    elif is_closed_fist(hand_landmarks):
                        current_notes = ORIGINAL_NOTES
                        player = SoundPlayer(current_notes)
                        last_gesture_time = current_time
            
            elif hand_label == 'Right':
                # Finger tip and base indices
                finger_indices = [
                    (mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_MCP),
                    (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_MCP),
                    (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
                    (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_MCP),
                    (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_MCP),
                ]
                
                # Process each finger
                for i, (tip_idx, base_idx) in enumerate(finger_indices):
                    tip = hand_landmarks.landmark[tip_idx]
                    base = hand_landmarks.landmark[base_idx]
                    
                    tip_x, tip_y = int(tip.x * w), int(tip.y * h)
                    base_x, base_y = int(base.x * w), int(base.y * h)
                    
                    is_down = is_finger_down((tip_x, tip_y), (base_x, base_y))
                    
                    if is_down and not finger_pressed[i] and (current_time - last_press_time[i]) > PRESS_COOLDOWN:
                        player.play_note_by_index(i)
                        finger_pressed[i] = True
                        last_press_time[i] = current_time
                    elif not is_down:
                        finger_pressed[i] = False
                    
                    color = (0, 255, 0) if finger_pressed[i] else (0, 0, 255)
                    cv2.circle(frame, (tip_x, tip_y), 10, color, -1)
                    cv2.circle(frame, (base_x, base_y), 5, (255, 255, 255), -1)
                    cv2.putText(frame, current_notes[i], (tip_x + 10, tip_y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.line(frame, (tip_x, tip_y), (base_x, base_y), color, 2)
    else:
        finger_pressed = [False] * 5

    # Display current notes at the top of the screen
    notes_text = " ".join(current_notes)
    text_size = cv2.getTextSize(notes_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = (w - text_size[0]) // 2  # Center the text
    cv2.putText(frame, notes_text, (text_x, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow("Virtual Piano", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 