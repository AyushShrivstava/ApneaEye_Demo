import os
import cv2
import numpy as np
import subprocess
import time
from collections import deque
from ultralytics import YOLO
import matplotlib.pyplot as plt
import datetime


ffmpeg_cmd = [
    "ffmpeg",
    "-hwaccel", "videotoolbox",
    "-f", "avfoundation",
    "-framerate", "25",
    "-video_size", "256x384",
    "-pixel_format", "yuyv422",
    "-i", "0",
    "-fflags", "nobuffer",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-"
]

# Start FFmpeg process
process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
width, height = 256, 384
frame_size = width * height * 3

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

name = "thermal-rec"
os.makedirs("thermal-rec", exist_ok=True)
video = cv2.VideoWriter("thermal-rec/" + name + "_" + timestamp + ".avi", cv2.VideoWriter_fourcc(*'XVID'), 25, (512, 384))


while True:
    buffer = bytearray(frame_size)
    bytes_read = process.stdout.readinto(buffer)

    # Check if we read enough bytes
    if bytes_read < frame_size:
        print("Frame is empty or incomplete, breaking")
        break

    frame = np.ndarray((height, width, 3), dtype=np.uint8, buffer=buffer)
    imdata, _ = np.array_split(frame, 2)
    imdata = cv2.resize(imdata, (256*2, 192*2), interpolation=cv2.INTER_LINEAR)

    cv2.imshow("Frame", imdata)
    video.write(imdata)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        video.release()
        break

cv2.destroyAllWindows()
cv2.waitKey(1)
video.release()