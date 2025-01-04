import serial
import time

# Configure serial port settings
SERIAL_PORT = '/dev/enp4s0'  # Replace with your actual serial port
BAUD_RATE = 115200            # Replace with the device's baud rate
TIMEOUT = 1                   # Timeout in seconds

try:
    # Initialize the serial connection
    ser = serial.Serial(port=SERIAL_PORT, baudrate=BAUD_RATE, timeout=TIMEOUT)
    print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")

    # Give the device some time to initialize
    time.sleep(2)

    # Send a command (example: replace with an actual command for your device)
    command = b'TEMPERATURE_REQUEST\n'  # Replace with your device's specific command
    ser.write(command)
    print(f"Sent command: {command}")

    # Read the response
    response = ser.readline()  # Read a single line of response
    print(f"Response from device: {response}")

    # Parse and display the response if it's temperature data
    # (Adjust parsing logic based on your device's protocol)
    if response:
        # Example: Assuming the device sends raw temperature as a string
        temperature = response.decode('utf-8').strip()
        print(f"Temperature: {temperature}")

except serial.SerialException as e:
    print(f"Serial connection error: {e}")
except Exception as e:
    print(f"Error: {e}")
finally:
    # Close the serial connection
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("Serial connection closed.")




