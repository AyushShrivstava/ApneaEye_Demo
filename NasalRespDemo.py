import cv2
import numpy as np
import subprocess
import time
import datetime
import torch
from collections import deque
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.signal import butter, sosfilt, sosfilt_zi
import pandas as pd

# Load YOLO model
yolo_model = YOLO("models/Yolov8_Localiser.pt")


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
    


def update_plot(respiration_data, nostril_visibility):
    # Apply filters
    respiration_data_smooth = streaming_filter1.filter(respiration_data)
    nasal_freq = get_dominant_frequency(respiration_data_smooth, 25)

    # Update nasal respiration plot
    line1.set_xdata(np.arange(len(respiration_data_smooth)))
    line1.set_ydata(respiration_data_smooth)

    # Clear previous fills properly
    for coll in ax1.collections:
        coll.remove()

    ax1.fill_between(np.arange(len(respiration_data_smooth)), -15, 15, where=nostril_visibility, 
                     color='red', alpha=0.3, step='mid')
    ax1.set_title(f'Nasal Respiration from Thermal Data (Freq: {nasal_freq*60:.2f} BrPM)')
    
    # Rescale axes
    ax1.relim()
    ax1.autoscale_view()

    # Draw canvas
    fig.tight_layout()
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

# Initialize filter
streaming_filter1 = StreamingBandpassFilter(lowcut=0.1, highcut=0.4, fs=25, order=3)

# Set up Matplotlib plot with two subplots (stacked vertically)
fig, ax1 = plt.subplots(1, 1, figsize=(8, 3.84), sharex=True,dpi=100)

# Initialize plot lines
line1, = ax1.plot([], [], 'g')  # Nasal respiration

ax1.set_title('Nasal Respiration from Thermal Data')
ax1.set_ylabel('Nasal Airflow')

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
respiration_signal = deque(maxlen=500)
nostril_visibility = deque(maxlen=500)

respiration_signal.append(0)
nostril_visibility.append(0)

infer_time = []
start_time = time.time()
prev = None
resp_plot = None

frame_process = []
timing = []

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Ensure output directory exists and prepare VideoWriter lazily once we know canvas size
os.makedirs("Data", exist_ok=True)
video_writer = None
video_path = os.path.join("Data", f"demo_nasal_{timestamp}.avi")

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

    imdata = cv2.resize(imdata, (512, 384), interpolation=cv2.INTER_LINEAR)

    # YOLO inference with `torch.no_grad()` for speedup
    if frame_number==0 or frame_number%5==0 or np.mean(cv2.absdiff(imdata, prev)) > 0.75:
        with torch.no_grad():
            result = yolo_model(imdata, verbose=False)
        
        box = result[0].boxes.data.cpu().numpy()
        infer_time.append(result[0].speed['inference'] + result[0].speed['preprocess'] + result[0].speed['postprocess'])

    # Take the box with the highest area
    if box.shape[0] > 0:
        box_df = pd.DataFrame(box, columns=['x1', 'y1', 'x2', 'y2','confidence','class'])
        box_df['area'] = (box_df['x2'] - box_df['x1']) * (box_df['y2'] - box_df['y1'])

        # check if nose is detected. If detected, take the biggest nose. If not, take the previous value
        if (box_df['class'] == 2).any():
            biggest_nose_idx = box_df[box_df['class'] == 2]['area'].idxmax()
            nose = box_df.loc[biggest_nose_idx]

            cv2.rectangle(imdata, (int(nose['x1']), int(nose['y1'])),
                      (int(nose['x2']), int(nose['y2'])), (0, 255, 0), 2)
            
            respiration_signal.append(np.mean(imdata[int(nose['y1']):int(nose['y2']),
                                                  int(nose['x1']):int(nose['x2'])]))
            nostril_visibility.append(0)

        else:
            nose = None
            respiration_signal.append(respiration_signal[-1] if respiration_signal else 0)
            nostril_visibility.append(1)
            
    else:
        respiration_signal.append(respiration_signal[-1] if respiration_signal else 0)
        nostril_visibility.append(1)

    prev = imdata.copy()
    canvas = imdata.copy()

    # Update plot every 5 frames for efficiency
    if len(respiration_signal) > 50 and frame_number % 25 == 0:
        resp_plot = update_plot(respiration_signal, nostril_visibility)
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
