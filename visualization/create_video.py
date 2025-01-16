import cv2
import os

# Path to the folder containing the images
image_folder = '/home/vk/03/ThermalSfMLearner/ProcessedData/indoor_robust_global/RGB/'  # Update this path to your folder 
video_name = './vikas_data/output_video.avi'  # Output video file name

# Parameters for the video
frame_rate = 5  # Frames per second

# Get a list of image files in the folder and sort them
images = sorted([img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))])

# Ensure there are images in the folder
if not images:
    print("No images found in the folder.")
    exit()

# Read the first image to get the frame dimensions
first_image_path = os.path.join(image_folder, images[0])
frame = cv2.imread(first_image_path)
height, width, layers = frame.shape

# Initialize the video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for the video
video = cv2.VideoWriter(video_name, fourcc, frame_rate, (width, height))

# Add images to the video
for image_name in images:
    image_path = os.path.join(image_folder, image_name)
    frame = cv2.imread(image_path)
    video.write(frame)  # Write the frame to the video

# Release the video writer
video.release()
cv2.destroyAllWindows()

print(f"Video created successfully: {video_name}")
