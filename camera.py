import cv2
import os

# Get the current directory where the script is located
current_directory = os.path.dirname(os.path.abspath(__file__))

# Create the directory path for saving images inside the "Dataset" folder
dataset_directory = os.path.join(current_directory, "Dataset")

# Manually change the subdirectory name here
subdirectory_name = "13"

# Create the subdirectory if it doesn't exist
directory_name = os.path.join(dataset_directory, subdirectory_name)
if not os.path.exists(directory_name):
    os.makedirs(directory_name)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Define the aspect ratio
aspect_ratio = 2 / 3

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

        if key == ord("w"):
            # Check if the "Dataset" directory exists
            if not os.path.exists(dataset_directory):
                os.makedirs(dataset_directory)

            # Check if the subdirectory exists
            if not os.path.exists(directory_name):
                os.makedirs(directory_name)

            image_count = len(os.listdir(directory_name))
            image_name = os.path.join(directory_name, f"image_{image_count + 1}.jpg")
            cv2.imwrite(image_name, cropped_frame)
            print(f"Captured image: {image_name}")

        elif key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
