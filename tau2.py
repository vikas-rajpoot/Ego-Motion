# from flirpy.camera.lepton import Lepton
# import numpy as np
# import cv2  # OpenCV for displaying the image

# # Initialize the camera
# camera = Lepton()

# while True:
#     # Grab the image and convert it to float32
#     image = camera.grab().astype(np.float32)

#     # Normalize the image to the range [0, 255]
#     normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

#     # Convert the normalized image to uint8
#     image_uint8 = normalized_image.astype(np.uint8)

#     # Display the image using OpenCV
#     cv2.imshow("Thermal Image", image_uint8)

#     # Wait for a key press, and break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Close all OpenCV windows after the loop
# cv2.destroyAllWindows()



from flirpy.camera.lepton import Lepton
import numpy as np
import cv2  # OpenCV for displaying the image
import time  # For generating timestamps in filenames

# Initialize the camera
camera = Lepton() 

# Counter for saving images
frame_counter = 0

while True:
    # Grab the image and convert it to float32
    image = camera.grab().astype(np.float32)

    # Normalize the image to the range [0, 255]
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the normalized image to uint8
    image_uint8 = normalized_image.astype(np.uint8)

    # Display the image using OpenCV
    cv2.imshow("Thermal Image", image_uint8)

    # Save the image with a unique filename
    # timestamp = time.strftime("%Y%m%d-%H%M%S")  # Current time
    filename = f"./images/frame_{frame_counter}.png"
    cv2.imwrite(filename, image_uint8)
    print(f"Saved: {filename}")
    
    # Increment the counter
    frame_counter += 1

    # Wait for a key press, and break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close all OpenCV windows after the loop
cv2.destroyAllWindows()




