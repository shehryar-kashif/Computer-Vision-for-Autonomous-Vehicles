#rasp pi receiver
import cv2
import socket
import pickle
import struct

# Connect to Raspberry Pi
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.149.105', 8485))  # Pi IP

data = b""
payload_size = struct.calcsize("Q")

while True:
    # Read message length
    while len(data) < payload_size:
        packet = client_socket.recv(4096)
        if not packet:
            exit()  
        data += packet

    if len(data) < payload_size:
        continue  # wait for complete header

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("Q", packed_msg_size)[0]

    # Read frame data
    while len(data) < msg_size:
        data += client_socket.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]

    frame, img_road = pickle.loads(frame_data)
    cv2.imshow("Video from Pi", frame)
    cv2.imshow("Road", img_road)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

client_socket.close()
cv2.destroyAllWindows()