import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist
import pandas as pd
import time
import threading
from collections import deque
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import warnings
warnings.filterwarnings('ignore')

class SmartExamProctor:
    def __init__(self):
        # Initialize face detector and landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Eye aspect ratio threshold (blinking detection)
        self.EYE_AR_THRESH = 0.25
        self.EYE_AR_CONSEC_FRAMES = 3
        
        # Head pose thresholds
        self.HEAD_TURN_THRESH = 30  # degrees
        self.HEAD_TILT_THRESH = 20   # degrees
        
        # Suspicious activity counters
        self.blink_counter = 0
        self.look_away_counter = 0
        self.suspicious_counter = 0
        
        # Buffers for smooth detection
        self.ear_buffer = deque(maxlen=self.EYE_AR_CONSEC_FRAMES)
        
        # Load pre-trained model for object detection (phone, books, etc.)
        self.object_model = self.load_object_detection_model()
        
        # Log storage
        self.log_data = []
        self.start_time = time.time()
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
    def load_object_detection_model(self):
        """Load pre-trained object detection model"""
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(5, activation='softmax')(x)  # 5 classes: phone, book, person, laptop, other
        model = Model(inputs=base_model.input, outputs=predictions)
        return model
    
    def eye_aspect_ratio(self, eye):
        """Calculate Eye Aspect Ratio (EAR) for blink detection"""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear
    
    def get_head_pose(self, landmarks):
        """Estimate head pose using facial landmarks"""
        # 3D model points for face
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])
        
        # 2D image points from landmarks
        image_points = np.array([
            (landmarks.part(30).x, landmarks.part(30).y),    # Nose tip
            (landmarks.part(8).x, landmarks.part(8).y),      # Chin
            (landmarks.part(36).x, landmarks.part(36).y),    # Left eye left corner
            (landmarks.part(45).x, landmarks.part(45).y),    # Right eye right corner
            (landmarks.part(48).x, landmarks.part(48).y),    # Left Mouth corner
            (landmarks.part(54).x, landmarks.part(54).y)     # Right mouth corner
        ], dtype="double")
        
        # Camera internals
        size = (640, 480)
        focal_length = size[0]
        center = (size[0]/2, size[1]/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]], dtype="double")
        
        dist_coeffs = np.zeros((4,1))
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs)
        
        # Get Euler angles
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rotation_matrix, translation_vector))
        euler_angles = cv2.RQDecomp3x3(proj_matrix)[0]
        
        return euler_angles
    
    def detect_prohibited_objects(self, frame):
        """Detect phones, books, laptops in frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        blob = cv2.dnn.blobFromImage(rgb_frame, 0.007843, (224, 224), 127.5)
        # Simplified detection - in production use YOLO or SSD
        return False  # Placeholder
    
    def log_violation(self, violation_type, confidence=1.0):
        """Log violations to CSV"""
        timestamp = time.time() - self.start_time
        self.log_data.append({
            'timestamp': timestamp,
            'violation_type': violation_type,
            'confidence': confidence
        })
    
    def process_frame(self, frame):
        """Process single frame for proctoring"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        
        violations = []
        
        for rect in rects:
            shape = self.predictor(gray, rect)
            
            # Convert dlib shape to numpy array
            shape_np = np.array([[p.x, p.y] for p in shape.parts()])
            
            # Eye detection
            leftEye = shape_np[42:48]
            rightEye = shape_np[36:42]
            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            
            self.ear_buffer.append(ear)
            
            # Blink detection
            if len(self.ear_buffer) == self.EYE_AR_CONSEC_FRAMES:
                if np.mean(self.ear_buffer) < self.EYE_AR_THRESH:
                    self.blink_counter += 1
                    if self.blink_counter >= 6:  # Too many blinks
                        violations.append("Excessive Blinking")
                        self.log_violation("excessive_blinking", 0.8)
            
            # Head pose estimation
            head_pose = self.get_head_pose(shape)
            yaw, pitch, roll = head_pose[0], head_pose[1], head_pose[2]
            
            if abs(yaw) > self.HEAD_TURN_THRESH:
                self.look_away_counter += 1
                violations.append("Looking Away")
                self.log_violation("looking_away", abs(yaw)/90.0)
            
            if abs(roll) > self.HEAD_TILT_THRESH:
                violations.append("Head Tilted")
                self.log_violation("head_tilt", abs(roll)/90.0)
            
            # Draw landmarks
            for (i, point) in enumerate(shape_np):
                cv2.circle(frame, (int(point[0]), int(point[1])), 1, (0, 255, 0), -1)
        
        # Object detection
        if self.detect_prohibited_objects(frame):
            violations.append("Prohibited Object")
            self.log_violation("prohibited_object", 0.9)
        
        return frame, violations
    
    def save_logs(self, filename="proctoring_log.csv"):
        """Save logs to CSV"""
        df = pd.DataFrame(self.log_data)
        df.to_csv(filename, index=False)
        print(f"Logs saved to {filename}")
    
    def run(self):
        """Main proctoring loop"""
        print("Smart Exam Proctor Started...")
        print("Press 'q' to quit, 's' to save logs")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            processed_frame, violations = self.process_frame(frame)
            
            # Display violations
            if violations:
                violation_text = " | ".join(violations)
                cv2.putText(processed_frame, f"ALERT: {violation_text}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.suspicious_counter += 1
            
            # Show EAR and head pose
            cv2.putText(processed_frame, f"EAR: {np.mean(self.ear_buffer):.2f}", 
                       (10, processed_frame.shape[0] - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Smart Exam Proctor", processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_logs()
        
        self.save_logs()
        self.cap.release()
        cv2.destroyAllWindows()

# Additional features class for advanced proctoring
class AdvancedProctoring:
    @staticmethod
    def audio_monitoring():
        """Monitor for suspicious audio (whispering, typing)"""
        import pyaudio
        import numpy as np
        
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, 
                       input=True, frames_per_buffer=CHUNK)
        
        while True:
            data = stream.read(CHUNK)
            audio_data = np.frombuffer(data, dtype=np.int16)
            # Analyze for suspicious sounds
            volume = np.sqrt(np.mean(audio_data**2))
            if volume > 1000:  # Threshold for suspicious audio
                print("Audio violation detected!")
        
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    # Download required model: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    proctor = SmartExamProctor()
    proctor.run()