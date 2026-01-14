#!/usr/bin/env python3
"""
parking_with_tk_yolo_only.py

Parking monitor with Tkinter control UI. DETECTOR IS YOLO-ONLY (no bgsub fallback).

- Draw ROIs (rect or polygon-4)
- Optionally detect only inside ROIs (per-ROI crop)
- Tracks vehicles, detects parked vehicles in ROIs, sends alerts
- Alerts logged to console, GUI log, and alerts.csv
- Added: optionally save alert payloads in JSON (line-delimited) format
       and/or save each alert as a separate pretty JSON file in output/json_alerts/
Run:
    pip install opencv-python numpy ultralytics mysql-connector-python pika
    python parking_with_tk_yolo_only.py
"""
import os
import sys
import time
import json
import math
import threading
import queue
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox

import cv2
import numpy as np

# ---- additional imports to match intrusion-style program ----
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None
try:
    # keep same import pattern as your intrusion script
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from shared.rabbitmq_utils import publish_to_queues
except Exception:
    # publish_to_queues optional — script will still run but publishing will be skipped
    def publish_to_queues(payload):
        print("[WARN] publish_to_queues not available; payload would be:")
        print(json.dumps(payload, default=str, indent=2))

# If you use DB for ROI store later, DB_CONFIG placeholder present
DB_CONFIG = {
    'host': '192.168.202.37',
    'user': 'mirador',
    'password': 'stpl2024',
    'database': 'mirador_demo'
}

# Directories for saving images (kept similar to intrusion program)
DEBUG_FRAMES = os.environ.get("DEBUG_FRAMES", "/u1/code/frames")
INTRUDED_IMAGES_ROOT = os.environ.get("INTRUDED_IMAGES_ROOT", "/u1/code/intrusion/reader/intruded_images")
os.makedirs(DEBUG_FRAMES, exist_ok=True)
os.makedirs(INTRUDED_IMAGES_ROOT, exist_ok=True)

# --------------------------- Defaults / Config ----------------------------
VIDEO_SOURCE = 0
YOLO_WEIGHTS = None  # optional override path for weights

PARK_THRESHOLD_SECONDS = 300
ALERT_INTERVAL_SECONDS = 60
MIN_OVERLAP_RATIO = 0.30
SPEED_THRESH = 2.0
SPEED_WINDOW = 5
MAX_LOST_FRAMES = 30
DETECT_EVERY_N_FRAMES = 3

IOU_MATCH_THRESHOLD = 0.3
CENTROID_DIST_THRESHOLD = 100

ROI_SAVE_FILE = "rois.json"
ALERTS_CSV = "alerts.csv"
ALERTS_JSONL = "output/alerts.jsonl"            # JSON-lines file (one JSON per line)
ALERTS_JSON_DIR = "output/json_alerts"          # Directory to save separate JSON files
HYSTERESIS_SECONDS = 5

MAX_FRAME_WIDTH = 1280
DETECTOR_MODE = "yolo"

# Ensure output dirs exist
os.makedirs(os.path.dirname(ALERTS_JSONL), exist_ok=True)
os.makedirs(ALERTS_JSON_DIR, exist_ok=True)

# --------------------------- Utilities -----------------------------------
def now_ts() -> float:
    return time.time()

def timestamp_to_str(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

def iou(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    boxAArea = max(1e-6, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1e-6, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return interArea / float(boxAArea + boxBArea - interArea)

def rect_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

def intersection_area(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    return interW * interH

# --------------------------- Alerts (augmented to publish + JSON) ----------------
def save_frame_to_folder(frame, date_object, camera_ip, frame_id):
    """Save full annotated frame into structured folders and return path"""
    try:
        date_folder = date_object.strftime("%Y-%m-%d")
        hour_folder = f"{date_object.hour:02d}-{(date_object.hour+1)%24:02d}"
        formatted_ip = camera_ip.replace('.', '_') if camera_ip else "local"
        base_path = os.path.join(DEBUG_FRAMES, formatted_ip, date_folder, hour_folder)
        os.makedirs(base_path, exist_ok=True)
        timestamp = date_object.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{formatted_ip}_{timestamp}_{frame_id}.jpg"
        full_path = os.path.join(base_path, filename)
        cv2.imwrite(full_path, frame)
        return os.path.abspath(full_path)
    except Exception as e:
        print(f"Error saving frame: {e}")
        return None

def save_intruder_crop(frame, detection, date_object, camera_ip, track_id):
    """Save cropped intruder/vehicle image with structured folders"""
    try:
        x1, y1, x2, y2 = map(int, detection)
        # sanitize coords
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        date_folder = date_object.strftime("%Y-%m-%d")
        hour_folder = f"{date_object.hour:02d}-{(date_object.hour+1)%24:02d}"
        formatted_ip = camera_ip.replace('.', '_') if camera_ip else "local"
        crop_path = os.path.join(INTRUDED_IMAGES_ROOT, formatted_ip, date_folder, hour_folder)
        os.makedirs(crop_path, exist_ok=True)
        timestamp = date_object.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"intruder_track_{track_id}_{timestamp}.jpg"
        full_path = os.path.join(crop_path, filename)
        cropped_img = frame[y1:y2, x1:x2]
        cv2.imwrite(full_path, cropped_img)
        return os.path.abspath(full_path)
    except Exception as e:
        print(f"Error saving intruder crop: {e}")
        return None

def send_alert(vehicle_id: int,
               roi_id: int,
               parked_seconds: float,
               alert_type: str = "PARKED",
               log_queue: Optional[queue.Queue] = None,
               frame_path: Optional[str]=None,
               crop_path: Optional[str]=None,
               camera_ip: Optional[str]=None,
               save_json: bool=False,
               save_json_files: bool=False):
    """
    Send alert: prints, writes CSV, publishes payload to queues via publish_to_queues(),
    and optionally appends payload as JSON line to ALERTS_JSONL file or saves each alert as a separate JSON file.
    The JSON structure matches the requested format.
    """
    ts = now_ts()
    # human CSV/log line
    row = {
        "timestamp": timestamp_to_str(ts),
        "vehicle_id": vehicle_id,
        "roi_id": roi_id,
        "parked_seconds": round(parked_seconds, 1),
        "alert_type": alert_type,
    }
    line = f'{row["timestamp"]},{row["vehicle_id"]},{row["roi_id"]},{row["parked_seconds"]},{row["alert_type"]}'
    print("[ALERT]", line)
    if log_queue is not None:
        log_queue.put("[ALERT] " + line)
    write_header = not os.path.exists(ALERTS_CSV)
    try:
        with open(ALERTS_CSV, "a") as f:
            if write_header:
                f.write("timestamp,vehicle_id,roi_id,parked_seconds,alert_type\n")
            f.write(line + "\n")
    except Exception as e:
        print(f"[WARN] failed to write CSV: {e}")

    # Build payload in the exact shape you requested
    payload = {
        "readerId": 0.0,
        "type": "parking_alert",
        "frameId": f"{int(ts*1000)}",
        "zoneId": 0.0,
        "locationId": 0.0,
        "detectionTime": timestamp_to_str(ts),
        "vehicleId": vehicle_id,
        "roiId": roi_id,
        "parkedSeconds": float(round(parked_seconds, 3)),
        "alertType": alert_type,
        "frameLocation": frame_path,
        "cropLocation": crop_path,
        "cameraIp": camera_ip
    }

    # Publish (optional)
    try:
        publish_to_queues(payload)
        print(f"[INFO] Published parking alert: vehicle {vehicle_id} roi {roi_id}")
    except Exception as e:
        print(f"[WARN] publish_to_queues failed: {e}")

    # append JSON line
    if save_json:
        try:
            with open(ALERTS_JSONL, 'a', encoding='utf-8') as jf:
                jf.write(json.dumps(payload, default=str) + '\n')
            if log_queue:
                log_queue.put(f"[INFO] Alert JSON appended to {ALERTS_JSONL}")
        except Exception as e:
            print(f"[WARN] Failed to write alert JSON: {e}")

    # save each alert as separate JSON file
    if save_json_files:
        try:
            os.makedirs(ALERTS_JSON_DIR, exist_ok=True)
            # filename: alert_<timestamp_ms>_veh<vehicle_id>_roi<roi_id>.json
            fname = f"alert_{int(ts*1000)}_veh{vehicle_id}_roi{roi_id}.json"
            fullf = os.path.join(ALERTS_JSON_DIR, fname)
            with open(fullf, 'w', encoding='utf-8') as jf:
                json.dump(payload, jf, default=str, indent=2)
            if log_queue:
                log_queue.put(f"[INFO] Alert JSON file saved: {fullf}")
            print(f"[INFO] Alert JSON file saved: {fullf}")
        except Exception as e:
            print(f"[WARN] Failed to write alert JSON file: {e}")

# small helper to open folder across platforms
def open_alerts_folder():
    """Open ALERTS_JSON_DIR in the OS file browser."""
    path = os.path.abspath(ALERTS_JSON_DIR)
    try:
        if sys.platform.startswith('win'):
            os.startfile(path)
        elif sys.platform.startswith('darwin'):
            import subprocess
            subprocess.Popen(['open', path])
        else:
            import subprocess
            subprocess.Popen(['xdg-open', path])
    except Exception as e:
        print(f"[WARN] Failed to open folder {path}: {e}")

# --------------------------- ROI Drawer ---------------------------------
class ROIDrawer:
    """
    ROI drawer supporting two modes:
      - rect (click-drag)
      - poly4 (click 4 points)
    """
    def __init__(self, window_name="Draw ROIs (press q to continue)"):
        self.window_name = window_name
        self.rois: List[Tuple[int,int,int,int]] = []
        self.mode = "rect"  # or "poly4"
        self._drawing = False
        self._ix = self._iy = 0
        self._current = None
        self._poly_points: List[Tuple[int,int]] = []
        try:
            cv2.namedWindow(self.window_name)
            cv2.setMouseCallback(self.window_name, self._mouse_cb)
        except Exception as e:
            # If cv2 has no GUI backend, we still allow the drawer object but drawing won't work.
            print(f"[WARN] OpenCV GUI unavailable for ROI drawing: {e}")
            self.window_name = None

    def _mouse_cb(self, event, x, y, flags, param):
        if self.mode == "rect":
            if event == cv2.EVENT_LBUTTONDOWN:
                self._drawing = True
                self._ix, self._iy = x, y
                self._current = (x, y, x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
                self._current = (self._ix, self._iy, x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                self._drawing = False
                x1, y1 = self._ix, self._iy
                x2, y2 = x, y
                x1, x2 = min(x1,x2), max(x1,x2)
                y1, y2 = min(y1,y2), max(y1,y2)
                if (x2-x1) > 10 and (y2-y1) > 10:
                    self.rois.append((x1,y1,x2,y2))
                self._current = None
        else: # poly4
            if event == cv2.EVENT_LBUTTONDOWN:
                self._poly_points.append((x,y))
                if len(self._poly_points) >= 4:
                    xs = [p[0] for p in self._poly_points[:4]]
                    ys = [p[1] for p in self._poly_points[:4]]
                    x1, x2 = min(xs), max(xs)
                    y1, y2 = min(ys), max(ys)
                    if (x2-x1) > 10 and (y2-y1) > 10:
                        self.rois.append((x1,y1,x2,y2))
                    self._poly_points = []

    def toggle_mode(self):
        self.mode = "poly4" if self.mode == "rect" else "rect"
        print(f"[INFO] ROI drawer mode -> {self.mode}")

    def draw_on(self, canvas):
        overlay = canvas.copy()
        for idx, r in enumerate(self.rois):
            x1,y1,x2,y2 = r
            cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,200,0), 2)
            cv2.putText(overlay, f"ROI {idx}", (x1+4, y1+16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0),1)
        if self.mode == "rect" and self._current is not None:
            x1,y1,x2,y2 = self._current
            cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,200,255), 1)
            cv2.putText(overlay, f"Mode: rect (m to toggle)", (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200),1)
        else:
            if len(self._poly_points) > 0:
                for p in self._poly_points:
                    cv2.circle(overlay, p, 4, (0,50,200), -1)
                for i in range(1, len(self._poly_points)):
                    cv2.line(overlay, self._poly_points[i-1], self._poly_points[i], (0,50,200), 1)
            cv2.putText(overlay, f"Mode: poly4 (click 4 points) (m to toggle)", (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200),1)
        return overlay

    def reset(self):
        self.rois = []
        self._poly_points = []

    def save(self, filename=ROI_SAVE_FILE):
        with open(filename, "w") as f:
            json.dump({"rois": self.rois}, f)
        print(f"[INFO] Saved {len(self.rois)} ROIs to {filename}")

    def load(self, filename=ROI_SAVE_FILE):
        if not os.path.exists(filename):
            print(f"[WARN] ROI file not found: {filename}")
            return
        with open(filename) as f:
            j = json.load(f)
            self.rois = [tuple(map(int, r)) for r in j.get("rois", [])]
        print(f"[INFO] Loaded {len(self.rois)} ROIs from {filename}")

# --------------------------- Detector (YOLO ONLY) ------------------------------
class Detector:
    """
    YOLO-only detector using ultralytics.YOLO (like your intrusion program)
    """

    def __init__(self, weights=None, target_classes=None, log_queue: Optional[queue.Queue] = None):
        self.weights = weights
        self.target_classes = (target_classes or ["car", "truck", "bus", "motorbike", "bicycle"])  # vehicle classes
        self.log_queue = log_queue
        self.have_yolo = False

        if YOLO is None:
            msg = "[ERROR] ultralytics not available. Install with: pip install ultralytics"
            print(msg)
            if self.log_queue: self.log_queue.put(msg)
            raise RuntimeError(msg)

        try:
            if self.weights:
                self.model = YOLO(self.weights)
            else:
                self.model = YOLO("yolov8n.pt")
            self.have_yolo = True
            msg = "[INFO] Using YOLO-only detector (no fallback)."
            print(msg)
            if self.log_queue: self.log_queue.put(msg)
        except Exception as e:
            msg = f"[ERROR] Failed to load YOLO weights: {e}"
            print(msg)
            if self.log_queue: self.log_queue.put(msg)
            raise RuntimeError(msg)

    def detect(self, frame_rgb):
        """
        Returns list of {'bbox':[x1,y1,x2,y2], 'conf':float, 'class':str}
        frame_rgb should be HxWx3 (RGB)
        """
        if not self.have_yolo:
            raise RuntimeError("YOLO model not initialized")

        try:
            results = self.model.predict(source=frame_rgb, imgsz=max(frame_rgb.shape[:2]), conf=0.25, verbose=False)
        except Exception as e:
            msg = f"[ERROR] YOLO inference failed: {e}"
            print(msg)
            if self.log_queue: self.log_queue.put(msg)
            raise RuntimeError(msg)

        detections = []
        for r in results:
            boxes = getattr(r, "boxes", [])
            for b in boxes:
                try:
                    xyxy = b.xyxy[0].cpu().numpy()
                    conf = float(b.conf[0].cpu().numpy()) if hasattr(b, "conf") else float(b.conf)
                    cls_idx = int(b.cls[0].cpu().numpy()) if hasattr(b, "cls") else int(b.cls)
                except Exception:
                    vals = b.xyxy[0].numpy() if hasattr(b, "xyxy") else None
                    if vals is None:
                        continue
                    xyxy = vals
                    conf = float(getattr(b, "conf", 0.0))
                    cls_idx = int(getattr(b, "cls", 0))

                cls_name = self.model.names[cls_idx] if hasattr(self.model, "names") else str(cls_idx)
                if cls_name.lower() not in [c.lower() for c in self.target_classes]:
                    continue
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                detections.append({"bbox":[x1,y1,x2,y2], "conf":conf, "class":cls_name})
        return detections

# --------------------------- Tracker -------------------------------------
@dataclass
class Track:
    track_id: int
    bbox: Tuple[int,int,int,int]
    last_seen_ts: float
    missed_frames: int = 0
    positions: deque = field(default_factory=lambda: deque(maxlen=50))
    in_roi_flags: Dict[int,bool] = field(default_factory=dict)
    enter_timestamp: Dict[int, Optional[float]] = field(default_factory=dict)
    parked_since: Dict[int, Optional[float]] = field(default_factory=dict)
    last_alert_timestamp: Dict[int, Optional[float]] = field(default_factory=dict)

class Tracker:
    def __init__(self):
        self.next_id = 1
        self.tracks: Dict[int,Track] = {}

    @staticmethod
    def bbox_to_centroid(bbox):
        x1,y1,x2,y2 = bbox
        return int((x1+x2)/2), int((y1+y2)/2)

    def update(self, detections: List[Dict], ts: float):
        det_bboxes = [d["bbox"] for d in detections]
        matched_det_idx = set()
        matched_track_ids = set()
        if len(self.tracks)>0 and len(det_bboxes)>0:
            track_ids = list(self.tracks.keys())
            track_bboxes = [self.tracks[t].bbox for t in track_ids]
            iou_matrix = np.zeros((len(track_bboxes), len(det_bboxes)), dtype=float)
            for i,tb in enumerate(track_bboxes):
                for j,db in enumerate(det_bboxes):
                    iou_matrix[i,j] = iou(tb, db)
            while True:
                i,j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                if iou_matrix[i,j] < IOU_MATCH_THRESHOLD:
                    break
                track_id = track_ids[i]
                det_idx = j
                self._assign_detection_to_track(track_id, detections[det_idx], ts)
                matched_det_idx.add(det_idx)
                matched_track_ids.add(track_id)
                iou_matrix[i,:] = -1
                iou_matrix[:,j] = -1
        unmatched_dets = [idx for idx in range(len(det_bboxes)) if idx not in matched_det_idx]
        unmatched_tracks = [tid for tid in self.tracks if tid not in matched_track_ids]
        if unmatched_dets and unmatched_tracks:
            for det_idx in unmatched_dets[:]:
                db = det_bboxes[det_idx]
                dcx,dcy = self.bbox_to_centroid(db)
                best_tid = None
                best_dist = float("inf")
                for tid in unmatched_tracks:
                    tb = self.tracks[tid].bbox
                    tcx,tcy = self.bbox_to_centroid(tb)
                    dist = math.hypot(tcx-dcx, tcy-dcy)
                    if dist < best_dist:
                        best_dist = dist
                        best_tid = tid
                if best_dist < CENTROID_DIST_THRESHOLD:
                    self._assign_detection_to_track(best_tid, detections[det_idx], ts)
                    matched_det_idx.add(det_idx)
                    if best_tid in unmatched_tracks:
                        unmatched_tracks.remove(best_tid)
        for det_idx, det in enumerate(detections):
            if det_idx in matched_det_idx:
                continue
            self._create_track(det, ts)
        for tid in list(self.tracks.keys()):
            if tid not in matched_track_ids:
                if self.tracks[tid].last_seen_ts != ts:
                    self.tracks[tid].missed_frames += 1
        to_delete = []
        for tid,tr in list(self.tracks.items()):
            if tr.missed_frames > MAX_LOST_FRAMES:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

    def _assign_detection_to_track(self, track_id:int, detection:Dict, ts:float):
        tr = self.tracks.get(track_id)
        if tr is None:
            return
        tr.bbox = tuple(map(int, detection["bbox"]))
        tr.last_seen_ts = ts
        tr.missed_frames = 0
        cx,cy = self.bbox_to_centroid(tr.bbox)
        tr.positions.append((cx,cy,ts))

    def _create_track(self, detection:Dict, ts:float):
        tid = self.next_id; self.next_id += 1
        bbox = tuple(map(int, detection["bbox"]))
        tr = Track(track_id=tid, bbox=bbox, last_seen_ts=ts)
        cx,cy = self.bbox_to_centroid(bbox)
        tr.positions.append((cx,cy,ts))
        self.tracks[tid] = tr

    def get_tracks(self)->Dict[int,Track]:
        return self.tracks

# --------------------------- Parking helpers ------------------------------
def centroid_in_roi(centroid, roi):
    x,y = centroid
    x1,y1,x2,y2 = roi
    return x1 <= x <= x2 and y1 <= y <= y2

def bbox_roi_overlap_ratio(bbox, roi):
    inter = intersection_area(bbox, roi)
    denom = rect_area(bbox)
    if denom <= 0:
        return 0.0
    return inter / denom

def compute_speed_px_per_sec(positions: deque, window=SPEED_WINDOW):
    if len(positions) < 2:
        return 0.0
    pts = list(positions)[-window:]
    if len(pts) < 2:
        return 0.0
    total_dist = 0.0
    total_time = 1e-6
    for i in range(1, len(pts)):
        x1,y1,t1 = pts[i-1]
        x2,y2,t2 = pts[i]
        total_dist += math.hypot(x2-x1, y2-y1)
        total_time += max(1e-6, t2-t1)
    return total_dist / total_time

def overlay_info(frame, tracks: Dict[int,Track], rois: List[Tuple[int,int,int,int]], roi_parked_count: Dict[int,int],
                 park_threshold_local:int, alert_interval_local:int, speed_thresh_local:float, min_overlap_local:float):
    canvas = frame.copy()
    for idx, (x1,y1,x2,y2) in enumerate(rois):
        cv2.rectangle(canvas, (x1,y1), (x2,y2), (0,160,0), 2)
        cv2.putText(canvas, f"ROI {idx}: parked={roi_parked_count.get(idx,0)}", (x1+6, y1+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,160,0),1)
    for tid,tr in tracks.items():
        x1,y1,x2,y2 = tr.bbox
        cv2.rectangle(canvas, (x1,y1), (x2,y2), (255,200,0), 2)
        cv2.putText(canvas, f"ID {tid}", (x1+4,y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,0),1)
        y_offset = y2 + 14
        for rid, parked_since in tr.parked_since.items():
            if parked_since:
                parked_seconds = int(time.time() - parked_since)
                cv2.putText(canvas, f"ROI{rid} parked {parked_seconds}s", (x1+4, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255),1)
                y_offset += 14
    legend_y = 20
    legend = [
        "Keys: r=reset ROIs, m=toggle ROI mode (draw stage), q/ESC=quit",
        f"PARK_THRESHOLD={park_threshold_local}s  ALERT_INTERVAL={alert_interval_local}s",
        f"MIN_OVERLAP={min_overlap_local}  HYST={HYSTERESIS_SECONDS}s"
    ]
    for i,text in enumerate(legend):
        cv2.putText(canvas, text, (8, legend_y + 18*i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200),1)
    return canvas

# --------------------------- Processing thread ---------------------------
def processing_thread_func(config: dict, stop_event: threading.Event, log_queue: queue.Queue):
    source = config.get("source", 0)
    weights = config.get("weights", YOLO_WEIGHTS)
    park_threshold = int(config.get("park_threshold", PARK_THRESHOLD_SECONDS))
    alert_interval = int(config.get("alert_interval", ALERT_INTERVAL_SECONDS))
    speed_thresh = float(config.get("speed_thresh", SPEED_THRESH))
    min_overlap = float(config.get("min_overlap", MIN_OVERLAP_RATIO))
    detect_every = int(config.get("detect_every", DETECT_EVERY_N_FRAMES))
    detect_only_in_rois = bool(config.get("detect_only_in_rois", False))
    save_alert_json = bool(config.get("save_alert_json", False))
    save_json_files = bool(config.get("save_json_files", False))

    try:
        src = int(source)
    except Exception:
        src = source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        msg = f"[ERROR] Cannot open video source: {source}"
        print(msg); log_queue.put(msg)
        return

    ret, first_frame = cap.read()
    if not ret:
        msg = "[ERROR] Cannot read from video source."
        print(msg); log_queue.put(msg)
        cap.release()
        return

    orig_h, orig_w = first_frame.shape[:2]
    scaling = 1.0
    if max(orig_w, orig_h) > MAX_FRAME_WIDTH:
        scaling = MAX_FRAME_WIDTH / max(orig_w, orig_h)
        first_frame = cv2.resize(first_frame, (int(orig_w*scaling), int(orig_h*scaling)))
    working_w, working_h = first_frame.shape[1], first_frame.shape[0]

    drawer = ROIDrawer(window_name="Draw ROIs (press q to continue)")
    log_queue.put("[INSTR] Draw ROIs; press m to toggle mode; press q to continue.")
    # if OpenCV GUI unavailable, skip interactive draw
    if drawer.window_name:
        while not stop_event.is_set():
            display = drawer.draw_on(first_frame)
            cv2.imshow(drawer.window_name, display)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                drawer.reset(); log_queue.put("[INFO] ROIs reset.")
            elif key == ord('s'):
                drawer.save(); log_queue.put("[INFO] saved rois.")
            elif key == ord('l'):
                drawer.load(); log_queue.put("[INFO] loaded rois.")
            elif key == ord('m'):
                drawer.toggle_mode(); log_queue.put(f"[INFO] ROI mode -> {drawer.mode}")
        if drawer.window_name:
            cv2.destroyWindow(drawer.window_name)
    else:
        log_queue.put("[WARN] ROI drawing not available (no cv2 GUI). Use saved rois.json or load via UI.")

    rois = list(drawer.rois)
    log_queue.put(f"[INFO] Using {len(rois)} ROIs.")
    detector = Detector(weights=weights, log_queue=log_queue)
    tracker = Tracker()
    frame_idx = 0
    roi_parked_count = defaultdict(int)
    last_inside_ts: Dict[int, Dict[int, Optional[float]]] = defaultdict(dict)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or math.isnan(fps):
        fps = 25.0
    log_queue.put(f"[INFO] Source FPS (est): {fps}")

    window_name = "Parking Monitor"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    except Exception:
        window_name = None
        log_queue.put("[WARN] OpenCV GUI unavailable — display disabled.")

    camera_ip = config.get("source", "local")

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            log_queue.put("[INFO] End of stream or read error.")
            break
        orig_h, orig_w = frame.shape[:2]
        if scaling != 1.0:
            small = cv2.resize(frame, (working_w, working_h))
        else:
            small = frame.copy()
        ts = time.time()
        frame_idx += 1

        detections = []
        if frame_idx % detect_every == 0:
            rgb_full = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            if detect_only_in_rois and len(rois)>0:
                PAD = 8
                for rid, roi in enumerate(rois):
                    x1,y1,x2,y2 = roi
                    xa = max(0, x1-PAD); ya = max(0, y1-PAD)
                    xb = min(working_w-1, x2+PAD); yb = min(working_h-1, y2+PAD)
                    crop_rgb = rgb_full[ya:yb, xa:xb]
                    if crop_rgb.size == 0:
                        continue
                    dets = detector.detect(crop_rgb)
                    for d in dets:
                        bx1,by1,bx2,by2 = d["bbox"]
                        mx1 = xa + int(bx1); my1 = ya + int(by1)
                        mx2 = xa + int(bx2); my2 = ya + int(by2)
                        cx = int((mx1+mx2)/2); cy = int((my1+my2)/2)
                        if centroid_in_roi((cx,cy), roi) or (bbox_roi_overlap_ratio((mx1,my1,mx2,my2), roi) >= min_overlap):
                            detections.append({"bbox":[mx1,my1,mx2,my2], "conf":d.get("conf",0.5), "class":d.get("class","vehicle")})
            else:
                dets = detector.detect(rgb_full)
                for d in dets:
                    bbox = d["bbox"]
                    bbox = [max(0,int(bbox[0])), max(0,int(bbox[1])), min(working_w-1, int(bbox[2])), min(working_h-1, int(bbox[3]))]
                    detections.append({"bbox":bbox, "conf":d.get("conf",0.5), "class": d.get("class","vehicle")})

        tracker.update(detections, ts)
        tracks = tracker.get_tracks()
        roi_parked_count = defaultdict(int)

        for tid, tr in tracks.items():
            cx,cy = tracker.bbox_to_centroid(tr.bbox)
            for rid, roi in enumerate(rois):
                inside = centroid_in_roi((cx,cy), roi) and (bbox_roi_overlap_ratio(tr.bbox, roi) >= min_overlap)
                if inside:
                    last_inside_ts[tid][rid] = ts
                else:
                    last_ts = last_inside_ts.get(tid, {}).get(rid, None)
                    if last_ts is not None and (ts - last_ts) <= HYSTERESIS_SECONDS:
                        inside = True
                    else:
                        inside = False
                        last_inside_ts[tid][rid] = None
                tr.in_roi_flags[rid] = bool(inside)
                if inside:
                    if tr.enter_timestamp.get(rid) is None:
                        tr.enter_timestamp[rid] = ts
                    parked_since = tr.parked_since.get(rid)
                    speed = compute_speed_px_per_sec(tr.positions, window=SPEED_WINDOW)
                    inside_time = ts - tr.enter_timestamp.get(rid, ts)
                    if parked_since is None:
                        if inside_time >= park_threshold and speed <= speed_thresh:
                            tr.parked_since[rid] = ts
                            tr.last_alert_timestamp[rid] = None
                            # Save frame and crop for alert
                            date_object = datetime.fromtimestamp(ts)
                            frame_path = save_frame_to_folder(small, date_object, str(camera_ip), f"{tid}_{int(ts*1000)}")
                            crop_path = save_intruder_crop(small, tr.bbox, date_object, str(camera_ip), tid)
                            send_alert(tr.track_id, rid, inside_time,
                                       alert_type="PARKED_START",
                                       log_queue=log_queue,
                                       frame_path=frame_path,
                                       crop_path=crop_path,
                                       camera_ip=str(camera_ip),
                                       save_json=save_alert_json,
                                       save_json_files=save_json_files)
                            tr.last_alert_timestamp[rid] = ts
                    else:
                        parked_seconds = ts - tr.parked_since[rid]
                        if tr.last_alert_timestamp.get(rid) is None:
                            date_object = datetime.fromtimestamp(ts)
                            frame_path = save_frame_to_folder(small, date_object, str(camera_ip), f"{tid}_{int(ts*1000)}")
                            crop_path = save_intruder_crop(small, tr.bbox, date_object, str(camera_ip), tid)
                            send_alert(tr.track_id, rid, parked_seconds,
                                       alert_type="PARKED_RENEW",
                                       log_queue=log_queue,
                                       frame_path=frame_path,
                                       crop_path=crop_path,
                                       camera_ip=str(camera_ip),
                                       save_json=save_alert_json,
                                       save_json_files=save_json_files)
                            tr.last_alert_timestamp[rid] = ts
                        else:
                            if ts - tr.last_alert_timestamp[rid] >= alert_interval:
                                date_object = datetime.fromtimestamp(ts)
                                frame_path = save_frame_to_folder(small, date_object, str(camera_ip), f"{tid}_{int(ts*1000)}")
                                crop_path = save_intruder_crop(small, tr.bbox, date_object, str(camera_ip), tid)
                                send_alert(tr.track_id, rid, parked_seconds,
                                           alert_type="PARKED_RENEW",
                                           log_queue=log_queue,
                                           frame_path=frame_path,
                                           crop_path=crop_path,
                                           camera_ip=str(camera_ip),
                                           save_json=save_alert_json,
                                           save_json_files=save_json_files)
                                tr.last_alert_timestamp[rid] = ts
                        roi_parked_count[rid] += 1
                else:
                    tr.in_roi_flags[rid] = False
                    tr.enter_timestamp[rid] = None
                    tr.parked_since[rid] = None
                    tr.last_alert_timestamp[rid] = None

        canvas = overlay_info(small, tracks, rois, roi_parked_count, park_threshold, alert_interval, speed_thresh, min_overlap)
        if scaling != 1.0:
            big = cv2.resize(canvas, (orig_w, orig_h))
        else:
            big = canvas

        if window_name:
            cv2.imshow(window_name, big)

        k = cv2.waitKey(1) & 0xFF if window_name else 0
        if k == ord('q') or k == 27:
            log_queue.put("[INFO] Stop requested via OpenCV window.")
            stop_event.set()
            break
        elif k == ord('r'):
            drawer.reset(); rois = []; log_queue.put("[INFO] ROIs cleared.")
        elif k == ord('s'):
            drawer.rois = rois; drawer.save(); log_queue.put("[INFO] ROIs saved.")
        elif k == ord('l'):
            drawer.load(); rois = drawer.rois; log_queue.put("[INFO] ROIs loaded.")
        elif k == ord('m'):
            drawer.toggle_mode(); log_queue.put(f"[INFO] ROI mode -> {drawer.mode}")

    cap.release()
    if window_name:
        cv2.destroyWindow(window_name)
    log_queue.put("[INFO] Processing thread exited.")

# --------------------------- Tk UI ---------------------------------------
class ParkingApp:
    def __init__(self, root):
        self.root = root
        root.title("Parking Monitor Control Panel")
        self.log_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None

        # variables
        self.source_var = tk.StringVar(value=str(VIDEO_SOURCE))
        self.detector_var = tk.StringVar(value=DETECTOR_MODE)
        self.park_thresh_var = tk.IntVar(value=PARK_THRESHOLD_SECONDS)
        self.alert_interval_var = tk.IntVar(value=ALERT_INTERVAL_SECONDS)
        self.min_overlap_var = tk.DoubleVar(value=MIN_OVERLAP_RATIO)
        self.detect_only_var = tk.BooleanVar(value=False)
        self.save_json_var = tk.BooleanVar(value=False)       # Save alerts to output/alerts.jsonl
        self.save_json_files_var = tk.BooleanVar(value=False) # Save separate JSON files to output/json_alerts/

        frm = ttk.Frame(root, padding=8)
        frm.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frm, text="Source (path or camera index):").grid(row=0,column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.source_var, width=50).grid(row=0,column=1, columnspan=2, sticky="we")
        ttk.Button(frm, text="Browse...", command=self.browse_source).grid(row=0,column=3)

        ttk.Label(frm, text="Detector:").grid(row=1,column=0, sticky="w")
        ttk.OptionMenu(frm, self.detector_var, self.detector_var.get(), "yolo").grid(row=1,column=1, sticky="w")

        ttk.Label(frm, text="Park threshold (sec):").grid(row=2,column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.park_thresh_var).grid(row=2,column=1, sticky="w")

        ttk.Label(frm, text="Alert interval (sec):").grid(row=2,column=2, sticky="w")
        ttk.Entry(frm, textvariable=self.alert_interval_var).grid(row=2,column=3, sticky="w")

        ttk.Label(frm, text="Min overlap ratio:").grid(row=3,column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.min_overlap_var).grid(row=3,column=1, sticky="w")

        ttk.Checkbutton(
            frm,
            text="Detect only inside ROIs (faster, may miss partials)",
            variable=self.detect_only_var
        ).grid(row=4, column=0, columnspan=3, sticky="w")

        # JSON output controls
        ttk.Checkbutton(frm, text="Save alert payloads to JSON (output/alerts.jsonl)", variable=self.save_json_var).grid(row=5, column=0, columnspan=3, sticky="w")
        ttk.Checkbutton(frm, text="Save each alert as separate JSON file (output/json_alerts/)", variable=self.save_json_files_var).grid(row=5, column=3, columnspan=1, sticky="w")

        ttk.Button(frm, text="Start", command=self.start_processing).grid(row=6,column=0)
        ttk.Button(frm, text="Stop", command=self.stop_processing).grid(row=6,column=1)
        ttk.Button(frm, text="Open Alerts Folder", command=self.open_alerts_folder).grid(row=6, column=2)

        ttk.Label(frm, text="Log:").grid(row=7,column=0, sticky="w")
        self.log_text = scrolledtext.ScrolledText(frm, width=90, height=18)
        self.log_text.grid(row=8, column=0, columnspan=4, pady=6)

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(frm, textvariable=self.status_var).grid(row=9, column=0, columnspan=4, sticky="we")

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(200, self.poll_log_queue)

    def browse_source(self):
        path = filedialog.askopenfilename(title="Select video file", filetypes=[("Video files","*.mp4 *.avi *.mov *.mkv"), ("All files","*.*")])
        if path:
            self.source_var.set(path)

    def start_processing(self):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Info","Processing already running.")
            return
        config = {
            "source": self.source_var.get(),
            "weights": YOLO_WEIGHTS,
            "park_threshold": int(self.park_thresh_var.get()),
            "alert_interval": int(self.alert_interval_var.get()),
            "speed_thresh": float(SPEED_THRESH),
            "min_overlap": float(self.min_overlap_var.get()),
            "detect_every": int(DETECT_EVERY_N_FRAMES),
            "detect_only_in_rois": bool(self.detect_only_var.get()),
            "save_alert_json": bool(self.save_json_var.get()),
            "save_json_files": bool(self.save_json_files_var.get()),
        }
        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=processing_thread_func, args=(config, self.stop_event, self.log_queue), daemon=True)
        self.worker_thread.start()
        self.status_var.set("Running")
        self.log_text.insert(tk.END, f"[INFO] Started processing source: {config['source']}\n"); self.log_text.see(tk.END)

    def stop_processing(self):
        if self.worker_thread and self.worker_thread.is_alive():
            self.stop_event.set()
            self.status_var.set("Stopping...")
            self.log_text.insert(tk.END, "[INFO] Stop requested.\n"); self.log_text.see(tk.END)
            self.root.after(500, self.check_thread_stopped)
        else:
            self.log_text.insert(tk.END, "[INFO] No processing thread running.\n"); self.log_text.see(tk.END)

    def check_thread_stopped(self):
        if self.worker_thread and not self.worker_thread.is_alive():
            self.status_var.set("Idle")
            self.log_text.insert(tk.END, "[INFO] Processing stopped.\n"); self.log_text.see(tk.END)
        else:
            self.root.after(500, self.check_thread_stopped)

    def save_rois(self):
        if os.path.exists(ROI_SAVE_FILE):
            dest = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON","*.json")])
            if dest:
                with open(ROI_SAVE_FILE,"r") as f: data = json.load(f)
                with open(dest,"w") as f: json.dump(data, f)
                self.log_text.insert(tk.END, f"[INFO] ROIs saved to {dest}\n"); self.log_text.see(tk.END)
        else:
            self.log_text.insert(tk.END, "[WARN] No ROI file present. Draw ROIs first.\n"); self.log_text.see(tk.END)

    def load_rois(self):
        path = filedialog.askopenfilename(title="Load ROIs", filetypes=[("JSON","*.json")])
        if path:
            try:
                with open(path) as f: j = json.load(f)
                with open(ROI_SAVE_FILE,"w") as f: json.dump(j, f)
                self.log_text.insert(tk.END, f"[INFO] ROIs loaded from {path}\n"); self.log_text.see(tk.END)
            except Exception as e:
                self.log_text.insert(tk.END, f"[ERROR] Load ROIs failed: {e}\n"); self.log_text.see(tk.END)

    def poll_log_queue(self):
        while not self.log_queue.empty():
            try:
                msg = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log_text.insert(tk.END, msg + "\n"); self.log_text.see(tk.END)
        self.root.after(200, self.poll_log_queue)

    def on_close(self):
        if self.worker_thread and self.worker_thread.is_alive():
            if not messagebox.askyesno("Quit","Processing still running. Stop and exit?"):
                return
            self.stop_event.set()
            time.sleep(0.3)
        self.root.destroy()

    def open_alerts_folder(self):
        """UI button handler to open the alerts folder."""
        try:
            open_alerts_folder()
            self.log_text.insert(tk.END, f"[INFO] Opened alerts folder: {os.path.abspath(ALERTS_JSON_DIR)}\n")
            self.log_text.see(tk.END)
        except Exception as e:
            self.log_text.insert(tk.END, f"[ERROR] Could not open alerts folder: {e}\n")
            self.log_text.see(tk.END)

def main():
    root = tk.Tk()
    app = ParkingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
