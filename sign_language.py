import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize MediaPipe Hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize the video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Flip the image horizontally
    image = cv2.flip(image, 1)

    # Process the image with MediaPipe Hands
    results = hands.process(image)

    # Check for hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the landmarks of the thumb, index, and middle fingers
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Get the x and y positions of the fingertips
            thumb_x, thumb_y = int(thumb_tip.x * image.shape[1]), int(thumb_tip.y * image.shape[0])
            index_x, index_y = int(index_finger_tip.x * image.shape[1]), int(index_finger_tip.y * image.shape[0])
            middle_x, middle_y = int(middle_finger_tip.x * image.shape[1]), int(middle_finger_tip.y * image.shape[0])

            # Draw circles around the fingertips
            cv2.circle(image, (thumb_x, thumb_y), 10, (255, 0, 0), -1)
            cv2.circle(image, (index_x, index_y), 10, (255, 0, 0), -1)
            cv2.circle(image, (middle_x, middle_y), 10, (255, 0, 0), -1)

            # Check if fingers are folded or not
            finger_fold_status = []
            if thumb_x < index_x:
                # Thumb folded
                cv2.circle(image, (thumb_x, thumb_y), 10, (0, 255, 0), -1)
                finger_fold_status.append(True)
            else:
                finger_fold_status.append(False)

            if thumb_x < middle_x:
                # Thumb folded
                cv2.circle(image, (thumb_x, thumb_y), 10, (0, 255, 0), -1)
                finger_fold_status.append(True)
            else:
                finger_fold_status.append(False)

            # Check if all fingers are folded
            if all(finger_fold_status):
                # Check if thumb is raised up or down
                if thumb_y < index_y and thumb_y < middle_y:
                    # Thumbs up gesture
                    cv2.putText(image, "LIKE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    print("Your friend likes the dress!")
                else:
                    # Thumbs down gesture
                    cv2.putText(image, "DISLIKE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    print("Your friend dislikes the dress!")

    # Convert the RGB image back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Display the image
    cv2.imshow('Hand Gestures', image)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
