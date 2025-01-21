import flirpy
# from flirpy.camera import Tau2
from flirpy.camera.tau import Tau2 as tau 

# Initialize the FLIR Tau2 camera
camera = tau()

# Start capturing video stream
camera.start_stream()

# Capture a thermal frame
frame = camera.get_frame()

# Extract temperature data (if radiometric)
temperature_data = frame.get_temperature_data()

# Process or display thermal data
frame.show()
