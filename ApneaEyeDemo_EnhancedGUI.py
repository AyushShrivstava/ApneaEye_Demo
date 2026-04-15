# Toggle the bottom "PROCESSING PIPELINE" flowchart panel on/off.
SHOW_FLOWCHART = False

import cv2
import numpy as np
import os
import queue
import subprocess
import threading
import time
import datetime
import torch

try:
    import psutil
    _mem_proc = psutil.Process(os.getpid())
except Exception:
    psutil = None
    _mem_proc = None
from collections import deque
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
from scipy.signal import butter, sosfilt, filtfilt
import pandas as pd
import os

# ---------------------------------------------------------------------------
# Enhanced GUI demo for ApneaEye.
# Core pipeline is identical to ApneaEyeDemo.py — only the visualization
# layer is redesigned: dark theme, HUD overlays, large BrPM readouts,
# status pills, and a composed dashboard canvas.
# ---------------------------------------------------------------------------

# Load YOLO model
yolo_model = YOLO("models/Yolov8_Localiser.pt")

# Per-frame chest activity threshold. Chest ROI absolute-diff mean values
# above this count as "active"; below count as "no activity" and get the red
# fill on the thoracic plot. Tune based on observed magnitudes in practice.
CHEST_ACTIVITY_THRESHOLD = 1.0
# Hold the "active" state for this many frames after each trigger — real
# motion lasts longer than a single frame of absdiff, so we extend each
# detection forward in time to match. 13 frames ≈ 0.52 s at 25 fps.
ACTIVITY_HOLD_FRAMES = 13


# ---------- Signal utilities ------------------------------------------------
def get_dominant_frequency(signal, fs):
    fourier = np.fft.fft(signal)
    n = len(signal)
    freq = np.fft.fftfreq(n, 1 / fs)
    freq = freq[: n // 2]
    fourier = fourier[: n // 2]
    peak_freq = freq[np.argmax(np.abs(fourier))]
    return peak_freq


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


class StreamingBandpassFilter:
    def __init__(self, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        self.sos = butter(order, [low, high], btype='band', output='sos')
        self.zi = np.zeros((self.sos.shape[0], 2))

    def filter(self, data):
        filtered, self.zi = sosfilt(self.sos, data, zi=self.zi)
        return filtered


# ---------- GUI theme -------------------------------------------------------
THEME = {
    "bg":          (0, 0, 0),
    "panel":       (0, 0, 0),
    "panel_edge":  (55, 60, 80),
    "text":        (235, 235, 240),
    "text_dim":    (150, 155, 170),
    "accent":      (120, 220, 255),   # cyan-ish
    "nasal":       (120, 255, 170),   # green  (BGR)
    "thorac":      (255, 180, 90),    # blue   (BGR)  — RGB(90,180,255)
    "ok":          (90, 210, 130),
    "warn":        (90, 140, 250),
    "bad":         (80, 90, 240),
}

mpl.rcParams.update({
    "figure.facecolor": "#000000",
    "axes.facecolor":   "#000000",
    "axes.edgecolor":   "#373c50",
    "axes.labelcolor":  "#ebebf0",
    "xtick.color":      "#9ba0b2",
    "ytick.color":      "#9ba0b2",
    "text.color":       "#ebebf0",
    "grid.color":       "#1a1d2a",
    "font.size":        9,
})


# ---------- Plot setup ------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3.84), sharex=True, dpi=100)
fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.16, hspace=0.55)

line1, = ax1.plot([], [], color=(120 / 255, 1.0, 170 / 255), lw=1.8)
line2, = ax2.plot([], [], color=(90 / 255, 180 / 255, 1.0), lw=1.8)

for ax in (ax1, ax2):
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

ax1.set_title('Nasal Respiration (Thermal)', loc='left', fontsize=10, pad=6)
ax1.set_ylabel('Airflow')
ax2.set_title('Thoracic Respiration (Optical Flow)', loc='left', fontsize=10, pad=6)
ax2.set_xlabel('Frames')
ax2.set_ylabel('Motion')

# Pin ylabels to a fixed axes-fraction position so they don't shift when the
# y-tick labels change width. Also use a fixed-width formatter so the tick
# labels themselves stay (visually) stable as the signal scale changes.
_YTICK_FMT = FuncFormatter(lambda v, _pos: f"{v:6.1f}")
for _ax in (ax1, ax2):
    _ax.yaxis.set_label_coords(-0.065, 0.5)
    _ax.yaxis.set_major_formatter(_YTICK_FMT)

# Legend hints explaining what the red fill means on each subplot. Placed in
# the top-right of each axes in axes-fraction coords so they never move.
_LEGEND_KW = dict(fontsize=7.5, ha='right', va='top',
                  color='#ff5a72', alpha=0.9)
ax1.text(0.995, 0.97, "red = nostril not visible",
         transform=ax1.transAxes, **_LEGEND_KW)
ax2.text(0.995, 0.97, "red = activity",
         transform=ax2.transAxes, **_LEGEND_KW)

streaming_filter1 = StreamingBandpassFilter(lowcut=0.1, highcut=0.4, fs=25, order=3)
streaming_filter2 = StreamingBandpassFilter(lowcut=0.1, highcut=0.4, fs=25, order=3)

# ---------- Background plot rendering --------------------------------------
# Matplotlib draw() is the single biggest per-loop cost (tens to hundreds of
# ms). We run it on a worker thread so the capture/display loop never blocks.
# Only the worker thread touches matplotlib objects, so no figure locking is
# needed beyond the "one worker at a time" guard.
_plot_state = {
    "img": None,
    "nasal": 0.0,
    "thorac": 0.0,
    "version": 0,
}
_plot_state_lock = threading.Lock()
_plot_worker = None


def _render_plot_worker(resp_snap, thorac_snap, nostril_snap, chest_snap, act_snap):
    img, nasal, thorac = update_plot(resp_snap, thorac_snap, nostril_snap,
                                     chest_snap, act_snap)
    with _plot_state_lock:
        _plot_state["img"] = img
        _plot_state["nasal"] = nasal
        _plot_state["thorac"] = thorac
        _plot_state["version"] += 1


def update_plot(respiration_data, thorac_data, nostril_lost, chest_lost,
                chest_activity_arr):
    nasal_smooth = streaming_filter1.filter(respiration_data)
    thorac_smooth = streaming_filter2.filter(thorac_data)

    nasal_freq = get_dominant_frequency(nasal_smooth, 25)
    thorac_freq = get_dominant_frequency(thorac_smooth, 25)

    # Detect chest activity per frame: mean absolute pixel diff in the chest
    # ROI. Above threshold → mark as active (red fill on the thoracic plot).
    act = np.asarray(chest_activity_arr, dtype=float)
    L = len(thorac_smooth)
    if act.size >= L:
        act = act[-L:]
    else:
        act = np.concatenate([np.zeros(L - act.size), act])
    raw_mask = act >= CHEST_ACTIVITY_THRESHOLD
    # Hold each detection forward in time by ACTIVITY_HOLD_FRAMES frames via
    # a backward-looking rolling-OR: thorac_mask[i] = any(raw_mask[i-H..i]).
    thorac_mask = (pd.Series(raw_mask.astype(np.uint8))
                   .rolling(ACTIVITY_HOLD_FRAMES, min_periods=1)
                   .max()
                   .to_numpy() > 0)

    x1 = np.arange(len(nasal_smooth))
    line1.set_xdata(x1)
    line1.set_ydata(nasal_smooth)
    for coll in list(ax1.collections):
        coll.remove()
    ax1.fill_between(x1, 0, 1, where=nostril_lost,
                     color=(1.0, 0.35, 0.45), alpha=0.25, step='mid',
                     transform=ax1.get_xaxis_transform())
    ax1.set_title(f'Nasal Respiration   ·   {nasal_freq*60:5.1f} BrPM',
                  loc='left', fontsize=10, pad=6)

    x2 = np.arange(len(thorac_smooth))
    line2.set_xdata(x2)
    # Display-only gain: thoracic optical-flow amplitude is much smaller than
    # the nasal signal, so scale 10× for visual parity. Frequency analysis is
    # unaffected — BrPM is computed from the unscaled smoothed signal above.
    line2.set_ydata(np.asarray(thorac_smooth) * 10)
    for coll in list(ax2.collections):
        coll.remove()
    ax2.fill_between(x2, 0, 1, where=thorac_mask,
                     color=(1.0, 0.35, 0.45), alpha=0.25, step='mid',
                     transform=ax2.get_xaxis_transform())
    ax2.set_title(f'Thoracic Respiration   ·   {thorac_freq*60:5.1f} BrPM',
                  loc='left', fontsize=10, pad=6)

    ax1.relim(); ax1.autoscale_view()
    ax2.relim(); ax2.autoscale_view()

    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR), nasal_freq * 60, thorac_freq * 60


# ---------- HUD drawing helpers --------------------------------------------
def _rounded_rect(img, p1, p2, color, radius=10, thickness=-1):
    x1, y1 = p1
    x2, y2 = p2
    if thickness < 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        cv2.rectangle(img, p1, p2, color, thickness)


def _text(img, text, org, scale=0.5, color=None, thickness=1):
    color = color or THEME["text"]
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_status_pill(img, org, label, ok):
    x, y = org
    color = THEME["ok"] if ok else THEME["bad"]
    _rounded_rect(img, (x, y), (x + 150, y + 28), THEME["panel"], radius=12)
    cv2.circle(img, (x + 16, y + 14), 6, color, -1)
    _text(img, label, (x + 30, y + 19), 0.5, THEME["text"], 1)


def draw_metric_card(img, org, size, title, value, unit, accent):
    x, y = org
    w, h = size
    _rounded_rect(img, (x, y), (x + w, y + h), THEME["panel"], radius=12)
    _rounded_rect(img, (x, y), (x + w, y + h), THEME["panel_edge"], radius=12, thickness=1)
    _text(img, title, (x + 14, y + 22), 0.5, THEME["text_dim"], 1)
    _text(img, f"{value}", (x + 14, y + 64), 1.3, accent, 2)
    _text(img, unit, (x + w - 52, y + 64), 0.55, THEME["text_dim"], 1)


# ---------- Dashboard layout (precomputed static background) ---------------
LAYOUT = {
    "header_h": 60,
    "footer_h": 110,
    "flow_h": 200,
    "thermal_w": 512,
    "plot_w": 800,
    "gap": 12,
    "body_h": 384,
    "card_w": 220,
}
LAYOUT["W"] = LAYOUT["thermal_w"] + LAYOUT["plot_w"] + LAYOUT["gap"] * 3
# Height: top gap + header + gap + body + gap + footer + [gap + flow] + bottom gap
if SHOW_FLOWCHART:
    LAYOUT["H"] = (LAYOUT["header_h"] + LAYOUT["body_h"] + LAYOUT["footer_h"]
                   + LAYOUT["flow_h"] + LAYOUT["gap"] * 5)
else:
    LAYOUT["H"] = (LAYOUT["header_h"] + LAYOUT["body_h"] + LAYOUT["footer_h"]
                   + LAYOUT["gap"] * 4)
LAYOUT["tx"] = LAYOUT["gap"]
LAYOUT["ty"] = LAYOUT["gap"] + LAYOUT["header_h"] + LAYOUT["gap"]
LAYOUT["px"] = LAYOUT["tx"] + LAYOUT["thermal_w"] + LAYOUT["gap"]
LAYOUT["py"] = LAYOUT["ty"]
LAYOUT["fy"] = LAYOUT["ty"] + LAYOUT["body_h"] + LAYOUT["gap"]
LAYOUT["inset"] = 6
LAYOUT["inner_w"] = LAYOUT["thermal_w"] - LAYOUT["inset"] * 2
LAYOUT["inner_h"] = LAYOUT["body_h"] - LAYOUT["inset"] * 2
LAYOUT["plot_inner_w"] = LAYOUT["plot_w"] - 12
LAYOUT["plot_inner_h"] = LAYOUT["body_h"] - 12


def _build_background_template():
    L = LAYOUT
    W, H = L["W"], L["H"]
    gap = L["gap"]
    canvas = np.full((H, W, 3), THEME["bg"], dtype=np.uint8)

    # Header panel (edge only — pure black fill blends with bg)
    _rounded_rect(canvas, (gap, gap), (W - gap, gap + L["header_h"]),
                  THEME["panel"], radius=14)
    _rounded_rect(canvas, (gap, gap), (W - gap, gap + L["header_h"]),
                  THEME["panel_edge"], radius=14, thickness=1)
    _text(canvas, "ApneaEye  |  Thermal Respiration Monitor",
          (gap + 20, gap + 38), 0.85, THEME["accent"], 2)

    # Thermal panel background + edge
    tx, ty = L["tx"], L["ty"]
    _rounded_rect(canvas, (tx, ty), (tx + L["thermal_w"], ty + L["body_h"]),
                  THEME["panel"], radius=14)
    _rounded_rect(canvas, (tx, ty), (tx + L["thermal_w"], ty + L["body_h"]),
                  THEME["panel_edge"], radius=14, thickness=1)

    # Plot panel background + edge
    px, py = L["px"], L["py"]
    _rounded_rect(canvas, (px, py), (px + L["plot_w"], py + L["body_h"]),
                  THEME["panel"], radius=14)
    _rounded_rect(canvas, (px, py), (px + L["plot_w"], py + L["body_h"]),
                  THEME["panel_edge"], radius=14, thickness=1)

    # Footer panel + static card backgrounds + static titles
    fy = L["fy"]
    _rounded_rect(canvas, (gap, fy), (W - gap, fy + L["footer_h"]),
                  THEME["panel"], radius=14)
    _rounded_rect(canvas, (gap, fy), (W - gap, fy + L["footer_h"]),
                  THEME["panel_edge"], radius=14, thickness=1)

    card_w = L["card_w"]
    card_h = L["footer_h"] - 14
    card_titles = ["NASAL RATE", "THORACIC RATE", "PROCESSING SPEED", "PROCESSED FRAMES"]
    card_units = ["BrPM", "BrPM", "FPS", ""]
    for i, (title, unit) in enumerate(zip(card_titles, card_units)):
        cx = gap + 14 + (card_w + 12) * i
        cy = fy + 7
        _rounded_rect(canvas, (cx, cy), (cx + card_w, cy + card_h),
                      THEME["panel"], radius=12)
        _rounded_rect(canvas, (cx, cy), (cx + card_w, cy + card_h),
                      THEME["panel_edge"], radius=12, thickness=1)
        _text(canvas, title, (cx + 14, cy + 22), 0.5, THEME["text_dim"], 1)
        if i < 2:
            # Rate cards: two static labels — "LIVE" and "1M" — with unit suffix
            _text(canvas, "LIVE", (cx + 10, cy + 48), 0.42, THEME["text_dim"], 1)
            _text(canvas, "1M",   (cx + card_w // 2 - 4, cy + 48), 0.42, THEME["text_dim"], 1)
            _text(canvas, unit,   (cx + card_w - 42, cy + 86), 0.45, THEME["text_dim"], 1)
        else:
            _text(canvas, unit, (cx + card_w - 52, cy + 64), 0.55, THEME["text_dim"], 1)

    # Status pill backgrounds
    sx = gap + 14 + (card_w + 12) * 4 + 8
    for label_y in (fy + 18, fy + 58):
        _rounded_rect(canvas, (sx, label_y), (sx + 150, label_y + 28),
                      THEME["panel"], radius=12)
    _text(canvas, "Nose tracked",  (sx + 30, fy + 18 + 19), 0.5, THEME["text"], 1)
    _text(canvas, "Chest tracked", (sx + 30, fy + 58 + 19), 0.5, THEME["text"], 1)

    # "Activity" indicator label (static). The dot itself is drawn per frame
    # in build_dashboard so it can change color with the current frame state.
    lx = sx + 160
    _text(canvas, "Activity", (lx + 18, fy + 58 + 19), 0.5, THEME["text"], 1)

    # Flowchart panel (static — drawn once into the template). Gated by flag.
    if SHOW_FLOWCHART:
        flow_y = fy + L["footer_h"] + gap
        _rounded_rect(canvas, (gap, flow_y), (W - gap, flow_y + L["flow_h"]),
                      THEME["panel"], radius=14)
        _rounded_rect(canvas, (gap, flow_y), (W - gap, flow_y + L["flow_h"]),
                      THEME["panel_edge"], radius=14, thickness=1)
        _build_flowchart(canvas, gap, flow_y, W - gap * 2, L["flow_h"])

    return canvas


def _build_flowchart(canvas, fx, fy, fw, fh):
    """Draw a static three-row processing pipeline flowchart."""
    _text(canvas, "PROCESSING PIPELINE", (fx + 16, fy + 22), 0.55,
          THEME["text_dim"], 1)

    # Three parallel pipelines
    rows = [
        {
            "label": "NASAL",
            "color": THEME["nasal"],
            "boxes": ["Thermal\nFrame", "YOLOv8\nLocaliser", "Nostril\nROI",
                      "Mean\nIntensity", "Bandpass\n0.1-0.4 Hz",
                      "FFT Peak\nFreq", "Nasal\nBrPM"],
        },
        {
            "label": "THORACIC",
            "color": (255, 180, 90),  # blue-ish in BGR matches thoracic line
            "boxes": ["Thermal\nFrame", "YOLOv8\nLocaliser", "Chest\nROI",
                      "LK Optical\nFlow", "Bandpass\n0.1-0.4 Hz",
                      "FFT Peak\nFreq", "Thoracic\nBrPM"],
        },
        {
            "label": "ACTIVITY",
            "color": THEME["bad"],
            "boxes": ["Thermal\nFrame", "YOLOv8\nLocaliser", "Chest\nROI",
                      "Frame\nDiff |Δ|", "Mean >\nThresh",
                      "Hold\n0.5 s", "Activity\nFlag"],
        },
    ]

    # Layout math
    n_boxes = 7
    label_w = 90
    left = fx + 16 + label_w
    right = fx + fw - 16
    avail_w = right - left
    box_w = 118
    box_h = 40
    gap_x = (avail_w - box_w * n_boxes) // (n_boxes - 1)
    if gap_x < 8:
        gap_x = 8
        box_w = (avail_w - gap_x * (n_boxes - 1)) // n_boxes

    top = fy + 38
    row_gap = (fh - 48) // len(rows)

    for r, row in enumerate(rows):
        ry = top + r * row_gap
        cy = ry + box_h // 2
        # Row label pill
        _rounded_rect(canvas, (fx + 16, ry + 4), (fx + 16 + label_w - 10, ry + box_h - 4),
                      THEME["panel"], radius=8)
        _rounded_rect(canvas, (fx + 16, ry + 4), (fx + 16 + label_w - 10, ry + box_h - 4),
                      row["color"], radius=8, thickness=1)
        _text(canvas, row["label"], (fx + 24, cy + 5), 0.5, row["color"], 1)

        for i, name in enumerate(row["boxes"]):
            bx = left + i * (box_w + gap_x)
            by = ry
            _rounded_rect(canvas, (bx, by), (bx + box_w, by + box_h),
                          THEME["panel"], radius=8)
            _rounded_rect(canvas, (bx, by), (bx + box_w, by + box_h),
                          row["color"], radius=8, thickness=1)
            # Two-line text
            lines = name.split("\n")
            if len(lines) == 1:
                _text(canvas, lines[0], (bx + 10, by + box_h // 2 + 5),
                      0.44, THEME["text"], 1)
            else:
                _text(canvas, lines[0], (bx + 10, by + 17),
                      0.42, THEME["text"], 1)
                _text(canvas, lines[1], (bx + 10, by + 33),
                      0.42, THEME["text"], 1)

            # Arrow to next box
            if i < n_boxes - 1:
                ax1 = bx + box_w + 1
                ax2 = bx + box_w + gap_x - 1
                cv2.arrowedLine(canvas, (ax1, cy), (ax2, cy),
                                THEME["text_dim"], 1, cv2.LINE_AA,
                                tipLength=0.35)

    return canvas


BG_TEMPLATE = _build_background_template()
_cached_plot_resized = None
_cached_plot_id = None

# Memory readout — refreshed at most once per second to keep the per-frame
# cost at zero. psutil.Process.memory_info() is cheap but we don't need
# 25 samples/second of it.
_mem_last_update = 0.0
_mem_text = "--"


def _get_mem_text():
    global _mem_last_update, _mem_text
    if _mem_proc is None:
        return "Memory Usage: n/a"
    now = time.time()
    if now - _mem_last_update >= 1.0:
        try:
            rss_mb = _mem_proc.memory_info().rss / (1024 * 1024)
            _mem_text = f"Memory Usage: {rss_mb:6.1f} MB"
        except Exception:
            _mem_text = "Memory Usage: err"
        _mem_last_update = now
    return _mem_text


def build_dashboard(thermal_bgr, plot_bgr, nasal_brpm, thorac_brpm,
                    nasal_brpm_avg, thorac_brpm_avg,
                    nose_ok, chest_ok, activity_now, fps_est, frame_idx):
    global _cached_plot_resized, _cached_plot_id
    L = LAYOUT
    gap = L["gap"]
    W = L["W"]

    canvas = BG_TEMPLATE.copy()

    # Dynamic: timestamp + memory usage (header right, inset from the corner)
    ts = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    _text(canvas, ts, (W - gap - 290, gap + 24), 0.55, THEME["text_dim"], 1)
    _text(canvas, _get_mem_text(), (W - gap - 290, gap + 48), 0.5,
          THEME["text_dim"], 1)

    # Thermal image — already 512x384, blit directly into inner region
    tx, ty = L["tx"], L["ty"]
    inset = L["inset"]
    if thermal_bgr.shape[1] != L["inner_w"] or thermal_bgr.shape[0] != L["inner_h"]:
        thermal_bgr = cv2.resize(thermal_bgr, (L["inner_w"], L["inner_h"]),
                                 interpolation=cv2.INTER_NEAREST)
    canvas[ty + inset:ty + inset + L["inner_h"],
           tx + inset:tx + inset + L["inner_w"]] = thermal_bgr

    # Plot — cache the resized version; only redo when the underlying plot changes
    px, py = L["px"], L["py"]
    if plot_bgr is not None:
        if id(plot_bgr) != _cached_plot_id:
            _cached_plot_resized = cv2.resize(
                plot_bgr, (L["plot_inner_w"], L["plot_inner_h"]),
                interpolation=cv2.INTER_NEAREST,
            )
            _cached_plot_id = id(plot_bgr)
        canvas[py + 6:py + 6 + L["plot_inner_h"],
               px + 6:px + 6 + L["plot_inner_w"]] = _cached_plot_resized
    else:
        _text(canvas, "Acquiring signal...",
              (px + L["plot_w"] // 2 - 90, py + L["body_h"] // 2),
              0.7, THEME["text_dim"], 1)

    # Dynamic: metric card values
    fy = L["fy"]
    card_w = L["card_w"]

    # Rate cards: LIVE (left) and 1M (right), both shown at equal prominence.
    rate_cards = [
        (nasal_brpm,  nasal_brpm_avg,  THEME["nasal"]),
        (thorac_brpm, thorac_brpm_avg, THEME["thorac"]),
    ]
    for i, (live, avg, col) in enumerate(rate_cards):
        cx = gap + 14 + (card_w + 12) * i
        cy = fy + 7
        _text(canvas, f"{live:4.1f}", (cx + 10, cy + 86), 0.85, col, 2)
        _text(canvas, f"{avg:4.1f}",
              (cx + card_w // 2 - 4, cy + 86), 0.85, col, 2)

    # FPS + Frame cards (single big number as before)
    other_values = [
        (f"{fps_est:5.1f}", THEME["accent"]),
        (f"{frame_idx:5d}", THEME["text"]),
    ]
    for j, (val, col) in enumerate(other_values):
        i = 2 + j
        cx = gap + 14 + (card_w + 12) * i
        cy = fy + 7
        _text(canvas, val, (cx + 14, cy + 64), 1.3, col, 2)

    # Dynamic: status dots
    sx = gap + 14 + (card_w + 12) * 4 + 8
    cv2.circle(canvas, (sx + 16, fy + 18 + 14), 6,
               THEME["ok"] if nose_ok else THEME["bad"], -1)
    cv2.circle(canvas, (sx + 16, fy + 58 + 14), 6,
               THEME["ok"] if chest_ok else THEME["bad"], -1)

    # Activity dot: red when the current frame shows chest activity (above
    # threshold), green when it's below (no activity).
    lx = sx + 160
    cv2.circle(canvas, (lx + 8, fy + 58 + 14), 5,
               THEME["bad"] if activity_now else THEME["ok"], -1)

    return canvas


# ---------- Capture setup ---------------------------------------------------
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
    "-",
]
process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, bufsize=10 ** 8)

width, height = 256, 384
frame_number = 0
respiration_signal = deque(maxlen=500)
nostril_visibility = deque(maxlen=500)
thorac_respiration = deque(maxlen=500)
chest_tracking = deque(maxlen=500)
# Per-frame thermal activity scalar — mean absolute difference between this
# frame's chest ROI and the previous frame's chest ROI. Cheap: one absdiff +
# one mean on a ~300×192 slice per frame. Used to detect "no chest activity"
# independently of the optical-flow signal.
chest_activity = deque(maxlen=500)

respiration_signal.append(0)
nostril_visibility.append(0)
thorac_respiration.append(0)
chest_tracking.append(0)
chest_activity.append(0.0)

infer_time = []
start_time = time.time()
prev_gray = None
prev_pts = None
prev = None
resp_plot = None
nasal_brpm = 0.0
thorac_brpm = 0.0
# 1-minute smoothed rate history. update_plot fires every 25 frames (~1/s at 25 fps),
# so 60 samples ≈ 60 seconds of history.
nasal_brpm_history = deque(maxlen=60)
thorac_brpm_history = deque(maxlen=60)
nasal_brpm_avg = 0.0
thorac_brpm_avg = 0.0
_last_plot_version = 0
# Countdown used to hold the live "Activity" dot red for ACTIVITY_HOLD_FRAMES
# frames after each raw trigger, mirroring the plot-side hold.
_activity_hold_counter = 0
fps_est = 0.0
last_fps_t = time.time()
fps_counter = 0

lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs("Data", exist_ok=True)
video_writer = None
video_path = os.path.join("Data", f"demo_ApneaEyeEnhanced_{timestamp}.avi")

# ---------- Background video writer ----------------------------------------
# cv2.VideoWriter.write is synchronous and costs several ms/frame at this
# resolution; we push frames to a bounded queue so the capture loop is never
# blocked by encoder work. The writer thread drains the queue. Frames are
# dropped (rather than the queue growing unbounded) if the encoder can't keep
# up — preferring low latency over perfect recording.
_video_queue: "queue.Queue" = queue.Queue(maxsize=8)
_video_stop = threading.Event()


def _video_writer_loop():
    while not _video_stop.is_set():
        try:
            frame = _video_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if frame is None:
            break
        if video_writer is not None:
            try:
                video_writer.write(frame)
            except Exception as e:
                print(f"Error writing frame to video: {e}")
        _video_queue.task_done()


_video_thread = threading.Thread(target=_video_writer_loop, daemon=True)
_video_thread.start()

box = np.zeros((0, 6))  # safe default so first-frame skip branch doesn't crash

while True:
    frame_size = width * height * 3
    buffer = bytearray(frame_size)
    process.stdout.readinto(buffer)
    frame = np.ndarray((height, width, 3), dtype=np.uint8, buffer=buffer)

    if np.sum(frame) == 0:
        print("No frame received. Exiting...")
        break

    imdata, _ = np.array_split(frame, 2)
    imdata = cv2.resize(imdata, (512, 384), interpolation=cv2.INTER_LINEAR)
    # Keep a pristine copy *before* any annotations are drawn. All signal
    # extraction (nose ROI, Canny edges, optical flow) must use this clean
    # frame — otherwise LK features latch onto our own label pills/boxes.
    clean = imdata.copy()

    chest = pd.DataFrame([[100, 192, 400, 384]], columns=['x1', 'y1', 'x2', 'y2'])

    # Thermal activity in the chest ROI (pairwise absdiff, mean over ROI).
    # Skipped on the first frame when `prev` doesn't exist yet. Appended
    # later in the loop, after the thoracic-signal branches, so all deques
    # stay in strict lock-step (one entry per frame).
    if prev is not None:
        _cx1 = int(chest['x1'].iloc[0])
        _cy1 = int(chest['y1'].iloc[0])
        _cx2 = int(chest['x2'].iloc[0])
        _cy2 = int(chest['y2'].iloc[0])
        _roi_now = clean[_cy1:_cy2, _cx1:_cx2]
        _roi_prev = prev[_cy1:_cy2, _cx1:_cx2]
        _chest_activity_now = float(cv2.absdiff(_roi_now, _roi_prev).mean())
    else:
        _chest_activity_now = 0.0

    if frame_number == 0 or frame_number % 5 == 0 or np.mean(cv2.absdiff(clean, prev)) > 0.75:
        with torch.no_grad():
            result = yolo_model(imdata, verbose=False)
        box = result[0].boxes.data.cpu().numpy()
        infer_time.append(result[0].speed['inference']
                          + result[0].speed['preprocess']
                          + result[0].speed['postprocess'])

    IM_H, IM_W = imdata.shape[:2]

    def _draw_labeled_box(x1, y1, x2, y2, color, label, side="top"):
        # Clamp the box to the image so the rectangle never draws outside.
        x1 = max(0, min(IM_W - 1, int(x1)))
        y1 = max(0, min(IM_H - 1, int(y1)))
        x2 = max(0, min(IM_W - 1, int(x2)))
        y2 = max(0, min(IM_H - 1, int(y2)))
        cv2.rectangle(imdata, (x1, y1), (x2, y2), color, 2)
        (tw, th_), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        pad_x, pad_y = 10, 8
        pill_w = tw + pad_x
        pill_h = th_ + pad_y

        if side == "right":
            # Pill to the right of the box, vertically centered on the top edge.
            px1 = x2 + 4
            if px1 + pill_w > IM_W:
                # No room on the right — fall back to left side of the box.
                px1 = max(0, x1 - pill_w - 4)
            px2 = px1 + pill_w
            py1 = max(0, min(IM_H - pill_h, y1))
            py2 = py1 + pill_h
        else:
            # Default: label above the box, falling back to below.
            if y1 - pill_h >= 0:
                py1 = y1 - pill_h
                py2 = y1
            else:
                py1 = y2
                py2 = min(IM_H - 1, y2 + pill_h)
            px1 = max(0, min(IM_W - pill_w, x1))
            px2 = px1 + pill_w

        cv2.rectangle(imdata, (px1, py1), (px2, py2), color, -1)
        cv2.putText(imdata, label, (px1 + 5, py2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 30), 1, cv2.LINE_AA)

    nose_ok = False
    face_box = None  # (x1, y1, x2, y2) of the current face detection, if any
    if box.shape[0] > 0:
        box_df = pd.DataFrame(box, columns=['x1', 'y1', 'x2', 'y2', 'confidence', 'class'])
        box_df['area'] = (box_df['x2'] - box_df['x1']) * (box_df['y2'] - box_df['y1'])

        # Draw biggest face (class 1) — informational only, not used in signal.
        if (box_df['class'] == 1).any():
            biggest_face_idx = box_df[box_df['class'] == 1]['area'].idxmax()
            face = box_df.loc[biggest_face_idx]
            face_box = (int(face['x1']), int(face['y1']),
                        int(face['x2']), int(face['y2']))
            _draw_labeled_box(face['x1'], face['y1'], face['x2'], face['y2'],
                              (120, 220, 255), "FACE", side="right")

        if (box_df['class'] == 2).any():
            biggest_nose_idx = box_df[box_df['class'] == 2]['area'].idxmax()
            nose = box_df.loc[biggest_nose_idx]
            _draw_labeled_box(nose['x1'], nose['y1'], nose['x2'], nose['y2'],
                              (120, 255, 170), "NOSTRIL", side="right")

            respiration_signal.append(np.mean(clean[int(nose['y1']):int(nose['y2']),
                                                    int(nose['x1']):int(nose['x2'])]))
            nostril_visibility.append(0)
            nose_ok = True
        else:
            respiration_signal.append(respiration_signal[-1] if respiration_signal else 0)
            nostril_visibility.append(1)
    else:
        respiration_signal.append(respiration_signal[-1] if respiration_signal else 0)
        nostril_visibility.append(1)
        chest_tracking.append(1)
        thorac_respiration.append(thorac_respiration[-1] if thorac_respiration else 0)

    # Use the clean (unannotated) frame for every downstream pixel operation.
    img_gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    chest_ok = False
    if chest is not None:
        edges = cv2.Canny(img_gray, 150, 200)
        edges[:, :int(chest['x1'].iloc[0])] = 0
        edges[:, int(chest['x2'].iloc[0]):] = 0
        edges[:int(chest['y1'].iloc[0])] = 0
        edges[int(chest['y2'].iloc[0]):] = 0

        if prev_gray is not None:
            if prev_pts is None or len(prev_pts) < 30:
                prev_pts = cv2.goodFeaturesToTrack(edges, maxCorners=50, qualityLevel=0.05,
                                                   minDistance=5, blockSize=7)

            if prev_pts is not None:
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, img_gray, prev_pts,
                                                               None, **lk_params)

            if next_pts is not None and status is not None and prev_pts is not None or \
                    np.mean(cv2.absdiff(
                        clean[int(chest['y1']):int(chest['y2']),
                              int(chest['x1']):int(chest['x2'])],
                        prev[int(chest['y1']):int(chest['y2']),
                             int(chest['x1']):int(chest['x2'])])) > 10.0:
                dx = next_pts[:, 0, 0] - prev_pts[:, 0, 0]
                dy = next_pts[:, 0, 1] - prev_pts[:, 0, 1]
                mag = np.sqrt(dx ** 2 + dy ** 2)
                angle = np.arctan2(dy, dx)
                sign = np.sign(np.cos(angle))
                thorac_respiration.append(np.mean(mag * sign))

                cx1 = int(chest['x1'].iloc[0])
                cy1 = int(chest['y1'].iloc[0])
                cx2 = int(chest['x2'].iloc[0])
                cy2 = int(chest['y2'].iloc[0])
                for new, _old in zip(next_pts, prev_pts):
                    a, b = new.ravel()
                    inside_chest = cx1 <= a <= cx2 and cy1 <= b <= cy2
                    inside_face = (face_box is not None
                                   and face_box[0] <= a <= face_box[2]
                                   and face_box[1] <= b <= face_box[3])
                    if inside_chest and not inside_face:
                        cv2.drawMarker(imdata, (int(a), int(b)), THEME["thorac"],
                                       markerType=cv2.MARKER_TILTED_CROSS,
                                       markerSize=8, thickness=2,
                                       line_type=cv2.LINE_AA)
                    prev_pts = next_pts[status == 1].reshape(-1, 1, 2)

                chest_tracking.append(0)
                chest_ok = True

        _draw_labeled_box(chest['x1'].iloc[0], chest['y1'].iloc[0],
                          chest['x2'].iloc[0], chest['y2'].iloc[0],
                          THEME["thorac"], "CHEST", side="right")
    else:
        thorac_respiration.append(thorac_respiration[-1] if thorac_respiration else 0)
        chest_tracking.append(1)

    prev_gray = img_gray.copy()
    prev = clean  # no copy needed — clean is not mutated further this iteration

    # --- Lock-step buffer alignment ------------------------------------------
    # Some branches above skip appending to thorac_respiration / chest_tracking
    # (e.g., first frame, prev_gray None, or prev_pts None). That would cause
    # these deques to drift out of sync with respiration_signal / chest_activity
    # and break the activity-detection alignment. Top them off here so every
    # deque grows by exactly one entry per frame.
    target_len = len(respiration_signal)
    while len(thorac_respiration) < target_len:
        thorac_respiration.append(thorac_respiration[-1] if thorac_respiration else 0)
    while len(chest_tracking) < target_len:
        chest_tracking.append(chest_tracking[-1] if chest_tracking else 1)
    chest_activity.append(_chest_activity_now)

    # Update plot every 25 frames
    # Kick off plot render in a background thread (one at a time). Main loop
    # continues using the previously rendered plot until the worker finishes.
    if (len(respiration_signal) > 50
            and frame_number % 25 == 0
            and (_plot_worker is None or not _plot_worker.is_alive())):
        resp_snap = np.fromiter(respiration_signal, dtype=float)
        thorac_snap = np.fromiter(thorac_respiration, dtype=float)
        nostril_snap = np.fromiter(nostril_visibility, dtype=bool)
        chest_snap = np.fromiter(chest_tracking, dtype=bool)
        act_snap = np.fromiter(chest_activity, dtype=float)
        _plot_worker = threading.Thread(
            target=_render_plot_worker,
            args=(resp_snap, thorac_snap, nostril_snap, chest_snap, act_snap),
            daemon=True,
        )
        _plot_worker.start()

    # Pick up the latest rendered plot if the worker produced a new one.
    with _plot_state_lock:
        cur_version = _plot_state["version"]
        if cur_version != _last_plot_version:
            resp_plot = _plot_state["img"]
            nasal_brpm = _plot_state["nasal"]
            thorac_brpm = _plot_state["thorac"]
            _last_plot_version = cur_version
            nasal_brpm_history.append(nasal_brpm)
            thorac_brpm_history.append(thorac_brpm)
            nasal_brpm_avg = float(np.mean(nasal_brpm_history))
            thorac_brpm_avg = float(np.mean(thorac_brpm_history))

    # FPS estimate (rolling)
    fps_counter += 1
    now = time.time()
    if now - last_fps_t >= 0.5:
        fps_est = fps_counter / (now - last_fps_t)
        fps_counter = 0
        last_fps_t = now

    if _chest_activity_now >= CHEST_ACTIVITY_THRESHOLD:
        _activity_hold_counter = ACTIVITY_HOLD_FRAMES
    activity_now = _activity_hold_counter > 0
    if _activity_hold_counter > 0:
        _activity_hold_counter -= 1
    dashboard = build_dashboard(
        imdata, resp_plot, nasal_brpm, thorac_brpm,
        nasal_brpm_avg, thorac_brpm_avg,
        nose_ok, chest_ok, activity_now, fps_est, frame_number,
    )

    cv2.imshow("ApneaEye - GUI Monitor", dashboard)

    if video_writer is None and resp_plot is not None:
        h, w = dashboard.shape[:2]
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'),
                                       25, (w, h), isColor=True)
        if not video_writer.isOpened():
            print(f"Warning: VideoWriter failed to open {video_path} with size {(w, h)}")
            video_writer = None

    if video_writer is not None:
        try:
            _video_queue.put_nowait(dashboard.copy())
        except queue.Full:
            # Encoder is behind — drop this frame rather than stall capture.
            pass

    frame_number += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        # Drain the writer queue, then stop the thread, then release.
        try:
            _video_queue.join()
        except Exception:
            pass
        _video_stop.set()
        _video_thread.join(timeout=1)
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
try:
    _video_queue.join()
except Exception:
    pass
_video_stop.set()
_video_thread.join(timeout=1)
if video_writer is not None:
    try:
        video_writer.release()
    except Exception as e:
        print(f"Warning releasing VideoWriter during cleanup: {e}")
    video_writer = None
cv2.waitKey(1)
