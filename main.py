import cv2
from cvzone.PoseModule import PoseDetector
detector=PoseDetector()




def calculate_direction(old_points, new_points):
    direction = "None"
    if old_points is not None and new_points is not None:
        x1, y1 = old_points
        x2, y2 = new_points
        if x2 - x1 > 0:
            direction = "Right"
        elif x2 - x1 < 0:
            direction = "Left"
        elif y2 - y1 > 0:
            direction = "Down"
        elif y2 - y1 < 0:
            direction = "Up"
    return direction



def detect_and_track(video_path):
    # Load pre-trained Haar cascade classifier for human body detection
    body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

    # Open video capture
    cap = cv2.VideoCapture(video_path)

    # Initialize variables for direction tracking
    prev_points = None
    movement_count = 0
    suspicious_detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break


        frame = detector.findPose(frame)
        lmlist, bboxInfo = detector.findPosition(frame, bboxWithHands=True)
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect human bodies in the frame
        bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw bounding boxes around detected human bodies and track back and forth movement
        for (x, y, w, h) in bodies:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Calculate centroid of bounding box
            centroid_x = int(x + w / 2)
            centroid_y = int(y + h / 2)
            current_point = (centroid_x, centroid_y)

            # Calculate direction of movement
            direction = calculate_direction(prev_points, current_point)

            # Check for back and forth movement
            if direction in ["Left", "Right"]:
                movement_count += 1
                if movement_count > 2:  # Adjust the threshold as needed
                    # Suspicious movement detected
                    print("Suspicious")
                    suspicious_detected = True
            else:
                movement_count = 0

            # Update previous points
            prev_points = current_point

        # Display frame with bounding boxes
        cv2.imshow('Frame', frame)

        # Exit if 'q' is pressed or suspicious movement is detected
        if cv2.waitKey(30) & 0xFF == ord('q') or suspicious_detected:
            break

    # Release video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_path = ('vd1.mp4')
detect_and_track(video_path)
