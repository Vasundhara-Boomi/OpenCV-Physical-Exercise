from flask import Flask, render_template, Response,jsonify
import cv2
import numpy as np
import mediapipe as mp

# Initialize Flask app
app = Flask(__name__)

# Initialize mediapipe pose class
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

show_message = False
counter_left = 0  # Define global variables for counters
counter_right = 0


# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle 

# Function to stream the video with landmarks and angles displayed
def video_stream():
    global counter_left, counter_right
    cap = cv2.VideoCapture(0)
    counter_left = 0
    counter_right = 0
    stage_left = None
    stage_right = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
        
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Calculate angles for left arm (similarly for right arm)
                # Left Arm
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                angle_left = calculate_angle(left_shoulder, left_elbow, left_wrist)

                # Right Arm
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                angle_right = calculate_angle(right_shoulder, right_elbow, right_wrist)

                # Visualize angle for left arm
                cv2.putText(image, f"Left Angle: {angle_left:.2f}", tuple(np.multiply(left_elbow, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Visualize angle for right arm
                cv2.putText(image, f"Right Angle: {angle_right:.2f}", tuple(np.multiply(right_elbow, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Counter logic for left arm
                
                if angle_left > 160 and stage_left != 'down':
                    stage_left = "down"
                if angle_left < 30 and stage_left == 'down':
                    stage_left = "up"
                    # Increase count value only when all landmarks are visible
                    counter_left += 1
                    if counter_left == 25:
                        counter_left = 0

                # Inside the code block for right arm count
                
                if angle_right > 160 and stage_right != 'down':
                    stage_right = "down"
                if angle_right < 30 and stage_right == 'down':
                    stage_right = "up"
                    # Increase count value only when all landmarks are visible

                    counter_right += 1
                    if counter_right == 25:
                        counter_right = 0


            except:
                pass
            
            # Render counters
            cv2.putText(image, f"Left Count: {counter_left}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Right Count: {counter_right}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()
    


# Route for the index page
@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/restart')
def restart():
    return render_template('index1.html')

@app.route('/get_counts')
def get_counts():
    global counter_left, counter_right
    # Return updated counts as JSON response
    return jsonify({'counter_left': counter_left, 'counter_right': counter_right})


@app.route('/phy_health')
def phy_health():
    global counter_left, counter_right  
    print(counter_left)
    print(counter_right)
    return render_template('index.html',counter_left=counter_left, counter_right=counter_right)


# Route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
