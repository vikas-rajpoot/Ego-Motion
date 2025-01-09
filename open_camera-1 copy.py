import socket

CAMERA_IP = '192.168.144.25'
CAMERA_PORT = 8554  # Replace with the correct port

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((CAMERA_IP, CAMERA_PORT))
print("Connected to the camera.")


# INIT_COMMAND = b'START_STREAM'
sock.sendall() 

raw_data = b''
while len(raw_data) < 5:  # Adjust expected_length for your resolution
    print("Receiving data...") 
    packet = sock.recv(4096) 
    print("Packet length:", len(packet)) 
    raw_data += packet


