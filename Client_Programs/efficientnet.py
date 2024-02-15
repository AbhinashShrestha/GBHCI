import mediapipe as mp
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class GestureDetector:
    def __init__(self, model_path, class_names, img_height=300, img_width=300, skip_frames=5):
        self.model = load_model(model_path)
        self.class_names = class_names
        self.img_height = img_height
        self.img_width = img_width
        self.skip_frames = skip_frames
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
        self.frame_counter = 0
        self.bbox = None
        self.predicted_class = None

    def detect_gestures(self, frame):
        if self.frame_counter % self.skip_frames == 0:
            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_landmarks_array = np.array([[data.x, data.y, data.z] for data in hand_landmarks.landmark])
                    x_min, y_min, _ = np.min(hand_landmarks_array, axis=0)
                    x_max, y_max, _ = np.max(hand_landmarks_array, axis=0)
                    padding = 0.05
                    x_min -= padding
                    y_min -= padding
                    x_max += padding
                    y_max += padding
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(1, x_max)
                    y_max = min(1, y_max)
                    self.bbox = np.array([x_min * frame.shape[1], y_min * frame.shape[0], x_max * frame.shape[1], y_max * frame.shape[0]]).astype(int)
                    hand_img = frame[self.bbox[1]:self.bbox[3], self.bbox[0]:self.bbox[2]]
                    img = cv2.resize(hand_img, (self.img_height, self.img_width))
                    img = image.img_to_array(img)
                    img = np.expand_dims(img, axis=0)
                    predictions = self.model.predict(img)
                    self.predicted_class = np.argmax(predictions[0])
                    confidence = np.max(predictions[0])
                    print('The predicted class is:', self.class_names[self.predicted_class])
                    print('Confidence:', confidence)
        if self.bbox is not None:
            cv2.rectangle(frame, (self.bbox[0], self.bbox[1]), (self.bbox[2], self.bbox[3]), (0, 0, 0), 3)
        if self.predicted_class is not None:
            cv2.putText(frame, self.class_names[self.predicted_class],(self.bbox[0], self.bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), thickness=2,lineType=cv2.LINE_AA)
        self.frame_counter += 1
        return frame

    def start_detection(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            frame = self.detect_gestures(frame)
            cv2.imshow('MediaPipe Hands', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    class_names = ["Brightness_Decrease", "Brightness_Increase", "Chrome_Open", "Cursor_Movement", "Double_Click", "Initiation", "Left_Click", "Neutral", "Nothing", "Right_Click", "Screenshot", "Scroll", "Shutdown", "Volume_Decrease", "Volume_Increase"]
    model_path = r'E:\MajorProject\Gesture based HCI\GBHCI\Non_Git\Models\EfficientNetV2S_300x300_FEB_14_Dataset_alpha.h5'
    detector = GestureDetector(model_path, class_names)
    detector.start_detection()
