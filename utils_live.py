import cv2
import os

def images_to_video(input_folder, output_video, fps=30):
    """
    Converts images in a folder to a video.
    
    Parameters:
    - input_folder (str): Path to the folder containing images.
    - output_video (str): Path to save the output video file.
    - fps (int): Frames per second for the output video.
    
    Returns:
    - None
    """ 
    # Get all image files sorted by name
    images = [img for img in os.listdir(input_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()  # Ensure images are in sequence
    
    if not images:
        print("No images found in the specified folder.")
        return
    
    # Read the first image to get the frame size
    first_image_path = os.path.join(input_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For .mp4 files
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    for image in images:
        image_path = os.path.join(input_folder, image)
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"Warning: Unable to read {image_path}. Skipping...")
            continue
        
        video_writer.write(frame)
    
    video_writer.release()
    print(f"Video saved as {output_video}")
 