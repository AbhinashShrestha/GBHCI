import cv2
import os

# Get the current directory where the script is located
current_directory = os.path.dirname(os.path.abspath(__file__))

# Create the directory path for saving images inside the "Dataset" folder
dataset_directory = "../Data/Dataset_alpha"

# Manually change the subdirectory name here
subdirectory_name = "Play"

# Create the subdirectory if it doesn't exist
directory_name = os.path.join(dataset_directory, subdirectory_name)
if not os.path.exists(directory_name):
    os.makedirs(directory_name)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Define the aspect ratio
aspect_ratio = 3/4

while True:
    ret, frame = cap.read()

    if ret:
        height, width, _ = frame.shape
        target_width = int(height * aspect_ratio)
        crop_start = (width - target_width) // 2
        crop_end = crop_start + target_width

        cropped_frame = frame[:, crop_start:crop_end]

        cv2.imshow("Capturing Images", cropped_frame)

        key = cv2.waitKey(1)

        if key == ord("c"):
            # Check if the "Dataset" directory exists
            if not os.path.exists(dataset_directory):
                os.makedirs(dataset_directory)

            # Check if the subdirectory exists
            if not os.path.exists(directory_name):
                os.makedirs(directory_name)

            image_count = len(os.listdir(directory_name))
            image_name = os.path.join(directory_name, f"{subdirectory_name.lower()}_{image_count + 1}.jpg")
            cv2.imwrite(image_name, cropped_frame)
            print(f"Captured image: {image_name}")

        elif key == 27:
            break

cap.release()
cv2.destroyAllWindows()


# above code is standard code for capturing images from camera and saving it in a folder
#below code is for capturing images of hand gestures and saving it in a folder

# import cv2
# import os
# import mediapipe as mp
# import time

# # Function to calculate hand bounding box coordinates from landmarks
# def get_hand_bounding_box(frame, hand_landmarks, padding=40):  # Increase padding to 50
#     x_min, x_max, y_min, y_max = frame.shape[1], 0, frame.shape[0], 0
#     for landmark in hand_landmarks.landmark:
#         x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
#         x_min = min(x_min, x)
#         x_max = max(x_max, x)
#         y_min = min(y_min, y)
#         y_max = max(y_max, y)
    
#     # Add padding to the bounding box coordinates
#     x_min = max(0, x_min - padding)
#     x_max = min(frame.shape[1], x_max + padding)
#     y_min = max(0, y_min - padding)
#     y_max = min(frame.shape[0], y_max + padding)
    
#     return x_min, x_max, y_min, y_max

# # Initialize the hand tracking module
# mp_hands = mp.solutions.hands
# hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# # Get the current directory where the script is located
# current_directory = os.path.dirname(os.path.abspath(__file__))

# # Create the directory path for saving images inside the "Dataset" folder
# dataset_directory = os.path.join(current_directory, "Dataset")

# # Manually change the subdirectory name here
# subdirectory_name = "Chrome_Open"

# # Create the subdirectory if it doesn't exist
# directory_name = os.path.join(dataset_directory, subdirectory_name)
# if not os.path.exists(directory_name):
#     os.makedirs(directory_name)

# # Initialize the camera
# cap = cv2.VideoCapture(0)

# # Define the fixed size for the display window
# display_size = (1080, 720)  # 2:3 aspect ratio

# # Initialize the time for hand detection
# last_detection_time = time.time()
# detection_interval = 5  # Detect hands every 5 seconds

# # Define the desired frame rate
# desired_fps = 30
# frame_time = 1 / desired_fps

# while True:
#     start_time = time.time()

#     ret, frame = cap.read()

#     if ret:
#         # Check if it's time to detect hands
#         current_time = time.time()
#         if current_time - last_detection_time >= detection_interval:
#             # Update the last detection time
#             last_detection_time = current_time
            
#             # Convert the frame from BGR to RGB
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
#             # Detect hands in the frame
#             results = hands_detector.process(frame_rgb)
            
#             # Check if any hands are detected
#             if results.multi_hand_landmarks:
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     # Access the hand landmarks (keypoints) to get the bounding box coordinates
#                     x_min, x_max, y_min, y_max = get_hand_bounding_box(frame, hand_landmarks)
                    
#                     # Draw the bounding box on the frame
#                     cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
#                     # Resize the frame to the fixed display size
#                     frame_resized = cv2.resize(frame, display_size)
                    
#                     cv2.imshow("Capturing Images", frame_resized)

#                     key = cv2.waitKey(1)

#                     if key == ord("w"):
#                         # Check if the "Dataset" directory exists
#                         if not os.path.exists(dataset_directory):
#                             os.makedirs(dataset_directory)

#                         # Check if the subdirectory exists
#                         if not os.path.exists(directory_name):
#                             os.makedirs(directory_name)

#                         # Get the total number of images already saved
#                         image_count = len(os.listdir(directory_name))
                        
#                         # Generate the image name based on the subdirectory name and the image count
#                         image_name = os.path.join(directory_name, f"{subdirectory_name}_{image_count + 1}.jpg")
                        
#                         # Save the cropped hand image
#                         hand_image = frame[y_min:y_max, x_min:x_max]
#                         cv2.imwrite(image_name, hand_image)
                        
#                         print(f"Captured image: {image_name}")

#                     elif key == ord("q"):
#                         break

#     # Wait for the remaining time to maintain the desired frame rate
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     remaining_time = frame_time - elapsed_time
#     if remaining_time > 0:
#         time.sleep(remaining_time)

# cap.release()
# cv2.destroyAllWindows()
