import matplotlib.pyplot as plt
import cv2
import numpy as np
import subprocess
import time

data = 'T'  # 'T' for thermal, 'R' for RGB

if data == 'T':
    res = "256x384"
    i = 0
else:
    res = "640x480"
    i = 1

# # --- Wake up camera using OpenCV ---
# temp_cap = cv2.VideoCapture(i)
# time.sleep(0.5)  # small delay to initialize camera
# temp_cap.release()

# FFmpeg command
ffmpeg_cmd = [
    "ffmpeg",
    "-f", "avfoundation",  # Use AVFoundation input
    "-hwaccel", "videotoolbox",
    "-framerate", "25",    # Set frame rate
    "-video_size", f"{res}",  # Set resolution
    "-pixel_format", "yuyv422",  # Set input pixel format
    "-i", f"{i}",  # Camera index (change if needed)
    "-f", "rawvideo",  # Output as raw video
    "-pix_fmt", "bgr24",  # Convert to OpenCV-friendly format
    "-"  # Output to stdout
]

# Start FFmpeg process
process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

if data == 'T':
    width, height = 256, 384 # Set resolution
else:
    width, height = 640, 480 # Set resolution

frame_number = 0
infer_time = []
start_time = time.time()

while True:
    # Read one frame from FFmpeg output
    frame_size = width * height * 3  # BGR24 has 3 channels

    frame_bytes = process.stdout.read(frame_size)

    if not frame_bytes:
        break  # End of stream
    
    # Convert bytes to numpy array
    frame = np.frombuffer(frame_bytes, np.uint8).reshape((height, width, 3))

    if data == 'T':
        # Split into image and thermal data
        imdata, thdata = np.array_split(frame, 2)
    else:
        imdata = frame

    # imdata = cv2.resize(imdata, (512,384), interpolation = cv2.INTER_CUBIC)

    # Show frame
    cv2.imshow("Camera", imdata)

    frame_number += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        break

end_time = time.time()
fps = frame_number / (end_time - start_time)
print(f"Processed {frame_number} frames")
print(f"Average FPS: {fps:.2f}")
print(f"Average inference time: {np.mean(infer_time):.2f} ms")

# Cleanup
# process.terminate()
# process.wait() 
# time.sleep(0.5)
# cv2.waitKey(1)
# cv2.destroyAllWindows()

process.terminate()
process.wait()  # wait until FFmpeg fully exits
cv2.destroyAllWindows()
cv2.waitKey(1)
time.sleep(0.5)  # allow macOS to release camera
