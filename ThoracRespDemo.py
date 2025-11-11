import cv2
import numpy as np
import subprocess
import time
import datetime
import torch
import os
from collections import deque
from ultralytics import YOLO
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.signal import butter, sosfilt, sosfilt_zi
import pandas as pd


def get_dominant_frequency(signal, fs):
    # Compute the Fourier Transform
    fourier = np.fft.fft(signal)
    n = len(signal)
    freq = np.fft.fftfreq(n, 1/fs)
    freq = freq[:n//2]
    fourier = fourier[:n//2]
    # Find the peak frequency
    peak_freq = freq[np.argmax(np.abs(fourier))]
    return peak_freq


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


# Streaming Bandpass Filter with State Retention
class StreamingBandpassFilter:
    def __init__(self, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        self.sos = butter(order, [low, high], btype='band', output='sos')
        self.zi = np.zeros((self.sos.shape[0], 2))  # Filter initial state
    
    def filter(self, data):
        filtered, self.zi = sosfilt(self.sos, data, zi=self.zi)
        return filtered
    

def update_plot(thorac_respiration):
    # Apply filters

    thorac_respiration_smooth = streaming_filter2.filter(thorac_respiration)
    thorac_freq = get_dominant_frequency(thorac_respiration_smooth, 25)

    line2.set_xdata(np.arange(len(thorac_respiration_smooth)))
    line2.set_ydata(thorac_respiration_smooth)

    # Clear previous fills properly
    for coll in ax2.collections:
        coll.remove()

    ax2.fill_between(np.arange(len(thorac_respiration_smooth)), -15, 15, where=chest_tracking,
                     color='red', alpha=0.3, step='mid')
    ax2.set_title(f'Thoracic Respiration from Thermal Data (Freq: {thorac_freq*60:.2f} BrPM)')
    
    ax2.relim()
    ax2.autoscale_view()

    # Draw canvas
    fig.tight_layout()
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

# Initialize filter
streaming_filter2 = StreamingBandpassFilter(lowcut=0.1, highcut=0.4, fs=25, order=3)

# Set up Matplotlib plot with two subplots (stacked vertically)
fig, ax2 = plt.subplots(1, 1, figsize=(8, 3.84), sharex=True,dpi=100)

# Initialize plot lines
line2, = ax2.plot([], [], 'b')  # Thoracic respiration

ax2.set_title('Thoracic Respiration from Thermal Data')
ax2.set_xlabel('Frames')
ax2.set_ylabel('Thoracic Movement')

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


# Start FFmpeg processs
process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,bufsize=10**8)

# Set parameters
width, height = 256, 384
frame_number = 0

thorac_respiration = deque(maxlen=500)
chest_tracking = deque(maxlen=500)

thorac_respiration.extend([0,0])
chest_tracking.extend([0,0])


infer_time = []
start_time = time.time()
prev_gray = None
prev_pts = None
prev = None
resp_plot = None

frame_process = []

lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

timing = []

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Ensure output directory exists and prepare VideoWriter lazily once we know canvas size
os.makedirs("Data", exist_ok=True)
video_writer = None
video_path = os.path.join("Data", f"demo_thorac_{timestamp}.avi")


while True:
    frame_process_start = time.time()
    temp = []

    # Read one frame from FFmpeg output
    frame_size = width * height * 3
    buffer = bytearray(frame_size)
    process.stdout.readinto(buffer)
    frame = np.ndarray((height, width, 3), dtype=np.uint8, buffer=buffer)

    if np.sum(frame) == 0:
        print("No frame received. Exiting...")
        break

    # Split and resize frame
    imdata, _ = np.array_split(frame, 2)

    # video_writer.write(imdata)

    imdata = cv2.resize(imdata, (512, 384), interpolation=cv2.INTER_LINEAR)

    chest = pd.DataFrame([[100,192,400,384]], columns=['x1', 'y1', 'x2', 'y2'])     # Default chest box if none detected
    prev = imdata.copy()


    img_gray = cv2.cvtColor(imdata, cv2.COLOR_BGR2GRAY)
    if chest is not None:
        edges = cv2.Canny(img_gray, 150, 200)

        #makes edges outside the chest region zero
        edges[:,:int(chest['x1'].iloc[0])] = 0
        edges[:,int(chest['x2'].iloc[0]):] = 0
        edges[:int(chest['y1'].iloc[0])] = 0
        edges[int(chest['y2'].iloc[0]):] = 0

        # cv2.imshow("edges", edges)

        if prev_gray is not None:
            frame_diff = cv2.absdiff(img_gray[int(chest['y1'].iloc[0]):int(chest['y2'].iloc[0]),
                                           int(chest['x1'].iloc[0]):int(chest['x2'].iloc[0])], prev_gray[int(chest['y1'].iloc[0]):int(chest['y2'].iloc[0]),
                                           int(chest['x1'].iloc[0]):int(chest['x2'].iloc[0])])
            if frame_diff is not None:
                mean_diff = np.mean(frame_diff)
            else:
                mean_diff = 0

            if prev_pts is None or len(prev_pts) < 30 or mean_diff > 10.0:
                prev_pts = cv2.goodFeaturesToTrack(edges, maxCorners=50, qualityLevel=0.05, minDistance=5, blockSize=7)

            if prev_pts is not None:
                # Compute Optical Flow
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, img_gray, prev_pts, None, **lk_params)

            if next_pts is not None and status is not None and prev_pts is not None or mean_diff > 10.0:
                # print(next_pts,prev_pts)
                dx = next_pts[:, 0, 0] - prev_pts[:, 0, 0]  # x displacement
                dy = next_pts[:, 0, 1] - prev_pts[:, 0, 1]  # y displacement

                mag = np.sqrt(dx**2 + dy**2)
                angle = np.arctan2(dy, dx)
                sign = np.sign(np.cos(angle))

                if len(next_pts) > 0 and len(prev_pts) > 0:
                    thorac_respiration.append(np.mean(np.sqrt(dx**2 + dy**2)*sign))  # Flow magnitude
                else:
                    thorac_respiration.append(thorac_respiration[-1] if thorac_respiration else 0)

                for i, (new, old) in enumerate(zip(next_pts, prev_pts)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                
                    # imdata = cv2.line(imdata, (int(a), int(b)), (int(c), int(d)), (0,0,255), 2)
                    imdata = cv2.circle(imdata, (int(a), int(b)), 2, (255, 0, 0), -1)
                    prev_pts = next_pts[status == 1].reshape(-1, 1, 2)

                chest_tracking.append(0)

        cv2.rectangle(imdata, (int(chest['x1'].iloc[0]), int(chest['y1'].iloc[0])),
                            (int(chest['x2'].iloc[0]), int(chest['y2'].iloc[0])), (255, 0, 0), 2)

    else:
        thorac_respiration.append(thorac_respiration[-1] if thorac_respiration else 0)
        chest_tracking.append(1)

    prev_gray = img_gray.copy()


    canvas = imdata.copy()

    # Update plot every 5 frames for efficiency
    if len(thorac_respiration) > 50 and frame_number % 25 == 0:
        resp_plot = update_plot(thorac_respiration)
        resp_plot = cv2.resize(resp_plot, (800, 384), interpolation=cv2.INTER_LINEAR)

    if resp_plot is not None:  
        canvas = np.concatenate((canvas, resp_plot), axis=1)
        
    cv2.imshow("Respiration from thermal video", canvas)
    
    # Initialize VideoWriter lazily with the actual canvas size to avoid size mismatch
    if video_writer is None and resp_plot is not None:
        h, w = canvas.shape[:2]
        # OpenCV expects (width, height)
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 25, (w, h), isColor=True)
        if not video_writer.isOpened():
            print(f"Warning: VideoWriter failed to open {video_path} with size {(w, h)}")
            video_writer = None

    if video_writer is not None:
        # Ensure frame shape matches writer expectation; convert if necessary
        try:
            video_writer.write(canvas)
        except Exception as e:
            print(f"Error writing frame to video: {e}")

    frame_number += 1
    
    # Quit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        if video_writer is not None:
            try:
                video_writer.release()
            except Exception as e:
                print(f"Warning releasing VideoWriter during key-exit: {e}")
            video_writer = None
        break

# Cleanup
try:
    process.terminate()
    process.wait(timeout=1)
except Exception:
    pass
cv2.destroyAllWindows()
if video_writer is not None:
    try:
        video_writer.release()
    except Exception as e:
        print(f"Warning releasing VideoWriter during cleanup: {e}")
    video_writer = None
cv2.waitKey(1)
