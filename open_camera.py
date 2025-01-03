import cv2
import os

def open_camera():
    # Open the default camera (index 0) 
    rtsp = "rtsp://192.168.144.25:8554/video2"
    cap = cv2.VideoCapture(rtsp) 

    if not cap.isOpened():
        print("Error: Could not open the camera.") 
        return

    print("Press 'q' to exit.")

    # Create a directory to save the frames
    save_dir = "captured_frames"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    frame_count = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read() 

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Save the frame
        frame_filename = os.path.join(save_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

        # Display the frame
        cv2.imshow('Camera Feed', frame) 

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    open_camera()
