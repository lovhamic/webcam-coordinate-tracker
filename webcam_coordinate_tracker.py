"""
Webcam Coordinate Tracker
=========================
A computer vision tool that converts an overhead 2D camera feed into a real-world
metric coordinate-measuring system. Computes real-world (X, Y) displacements in
centimeters relative to a defined virtual origin at the center of the camera field of view.

Supports:
- USB webcams (device indices: 0, 1, ...)
- Network / IP camera streams (HTTP, RTSP, MJPEG)
- Single-point and averaged jitter-free measurements
- In-situ ruler calibration with sub-pixel zoom inspection
- Automatic lens focal length estimation from 35mm optical specifications
- Persistent resolution-aware calibration storage (JSON)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


# ==============================================================================
# Mathematical and Projection Core
# ==============================================================================

class PinholeProjector:
    """
    Computes 2D real-world coordinates on an orthogonal plane using a pinhole camera model.
    
    The coordinate system assumes:
    - Camera optical axis is strictly perpendicular (normal) to the table surface plane.
    - Virtual origin (0, 0) is positioned at the optical center (frame center).
    - Table surface is flat and located at a known perpendicular distance H (camera height).
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        camera_height_cm: float,
        focal_px: Optional[float] = None,
        focal_source: str = "uninitialized",
    ) -> None:
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.camera_height_cm = float(camera_height_cm)
        self.origin_px = (self.frame_width // 2, self.frame_height // 2)
        self.focal_px = float(focal_px) if focal_px is not None else 0.0
        self.focal_calibrated = False
        self.focal_source = focal_source

    @staticmethod
    def estimate_focal_from_spec(
        frame_width: int,
        focal_length_35mm_equiv: float = 26.0,
        full_frame_sensor_width_mm: float = 36.0,
    ) -> float:
        """
        Estimates focal length in pixels using the camera's 35mm equivalent focal length.
        Formula:
            HFOV = 2 * atan(sensor_width / (2 * focal_length_35mm))
            focal_px = (frame_width / 2) / tan(HFOV / 2)
        """
        if focal_length_35mm_equiv <= 0:
            raise ValueError("Focal length must be strictly positive.")
        hfov_rad = 2.0 * math.atan(full_frame_sensor_width_mm / (2.0 * focal_length_35mm_equiv))
        focal_px = (frame_width / 2.0) / math.tan(hfov_rad / 2.0)
        return float(focal_px)

    @staticmethod
    def compute_focal_from_ruler(
        pixel_distance: float,
        camera_height_cm: float,
        real_distance_cm: float,
    ) -> float:
        """
        Computes focal length in pixels from a known real-world distance on the target plane.
        Formula:
            focal_px = (pixel_distance * camera_height_cm) / real_distance_cm
        """
        if real_distance_cm <= 0:
            raise ValueError("Real-world distance must be greater than zero.")
        if pixel_distance <= 0:
            raise ValueError("Pixel distance must be greater than zero.")
        if camera_height_cm <= 0:
            raise ValueError("Camera height must be greater than zero.")
        return float((pixel_distance * camera_height_cm) / real_distance_cm)

    def pixel_to_world_cm(self, px: float, py: float) -> Tuple[float, float]:
        """
        Projects image plane pixel coordinates (u, v) to real-world plane (X, Y) in centimeters.
        Formula:
            X_cm = (u - cx) * H / f
            Y_cm = (v - cy) * H / f
        """
        if self.focal_px <= 0:
            raise ValueError("Focal length in pixels must be positive to compute world coordinates.")
        u_rel = px - self.origin_px[0]
        v_rel = py - self.origin_px[1]
        x_cm = (u_rel * self.camera_height_cm) / self.focal_px
        y_cm = (v_rel * self.camera_height_cm) / self.focal_px
        return float(x_cm), float(y_cm)

    def get_scale_cm_per_px(self) -> float:
        """Returns the real-world scale factor in cm per pixel at the configured camera height."""
        if self.focal_px <= 0:
            return 0.0
        return float(self.camera_height_cm / self.focal_px)


# ==============================================================================
# Calibration Persistence Manager
# ==============================================================================

class CalibrationManager:
    """Manages saving and loading calibration data to and from a JSON file."""

    def __init__(self, filepath: str) -> None:
        self.filepath = os.path.abspath(filepath)

    def load(self, frame_width: int, frame_height: int) -> Optional[Dict[str, Any]]:
        """
        Loads calibration data if the file exists and resolution matches.
        Returns None if file does not exist, is corrupted, or resolution mismatches.
        """
        if not os.path.exists(self.filepath):
            return None
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            stored_w = data.get("frame_w")
            stored_h = data.get("frame_h")
            if stored_w == frame_width and stored_h == frame_height:
                return data
            return {
                "mismatch": True,
                "stored_w": stored_w,
                "stored_h": stored_h,
            }
        except (json.JSONDecodeError, KeyError, ValueError, OSError) as err:
            sys.stderr.write(f"Warning: Failed to read calibration file {self.filepath}: {err}\n")
            return None

    def save(
        self,
        focal_px: float,
        frame_width: int,
        frame_height: int,
        camera_height_cm: float,
    ) -> bool:
        """Saves verified calibration parameters to disk."""
        data = {
            "focal_px": float(focal_px),
            "frame_w": int(frame_width),
            "frame_h": int(frame_height),
            "camera_height_cm_when_calibrated": float(camera_height_cm),
        }
        try:
            parent_dir = os.path.dirname(self.filepath)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return True
        except OSError as err:
            sys.stderr.write(f"Error: Unable to save calibration file to {self.filepath}: {err}\n")
            return False


# ==============================================================================
# Computer Vision / Object Detection Core
# ==============================================================================

class ObjectDetector:
    """Detects foreground objects on a static background using MOG2 background subtraction."""

    def __init__(
        self,
        history: int = 300,
        var_threshold: float = 40.0,
        detect_shadows: bool = True,
        min_area: int = 900,
    ) -> None:
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self.min_area = int(min_area)

    @staticmethod
    def compute_centroid(contour: np.ndarray) -> Optional[Tuple[int, int]]:
        """Calculates contour spatial centroid (cx, cy) via image moments."""
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return None
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return cx, cy

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """
        Applies background subtraction, thresholding, and morphological filtering.
        Returns:
            fg_mask: Processed binary foreground mask.
            largest_contour: Largest foreground contour above min_area, or None.
            centroid: Centroid (x, y) of largest contour, or None.
        """
        raw_mask = self.subtractor.apply(frame)
        _, fg_mask = cv2.threshold(raw_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, None, iterations=2)
        fg_mask = cv2.dilate(fg_mask, None, iterations=3)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return fg_mask, None, None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) >= self.min_area:
            centroid = self.compute_centroid(largest)
            return fg_mask, largest, centroid

        return fg_mask, None, None


# ==============================================================================
# GUI Utilities and Overlays
# ==============================================================================

def render_zoom_inset(
    display_frame: np.ndarray,
    source_frame: np.ndarray,
    cursor_pos: Tuple[int, int],
    zoom_half_size: int = 60,
    zoom_scale: int = 6,
) -> None:
    """Renders a magnified zoom inset around the cursor in the top-right corner."""
    cx, cy = cursor_pos
    fh, fw = display_frame.shape[:2]
    x0, x1 = max(0, cx - zoom_half_size), min(fw, cx + zoom_half_size)
    y0, y1 = max(0, cy - zoom_half_size), min(fh, cy + zoom_half_size)

    patch = source_frame[y0:y1, x0:x1]
    if patch.size == 0:
        return

    zoomed = cv2.resize(patch, None, fx=zoom_scale, fy=zoom_scale, interpolation=cv2.INTER_LINEAR)
    zh, zw = zoomed.shape[:2]

    # Draw crosshair in zoomed inset
    cv2.line(zoomed, (zw // 2, 0), (zw // 2, zh), (0, 255, 255), 1)
    cv2.line(zoomed, (0, zh // 2), (zw, zh // 2), (0, 255, 255), 1)
    cv2.rectangle(zoomed, (0, 0), (zw - 1, zh - 1), (0, 165, 255), 2)

    inset_ox = fw - zw - 10
    inset_oy = 10
    if inset_oy + zh <= fh and inset_ox >= 0:
        display_frame[inset_oy:inset_oy + zh, inset_ox:inset_ox + zw] = zoomed


def parse_camera_source(source_str: str) -> Union[int, str]:
    """Parses a camera source parameter into an integer device index or a network URL/path."""
    trimmed = source_str.strip()
    if trimmed.isdigit():
        return int(trimmed)
    return trimmed


# ==============================================================================
# Application Controller
# ==============================================================================

class CoordinateTrackerApp:
    """Main application orchestrating video capture, tracking, calibration, and UI."""

    def __init__(
        self,
        source: Union[int, str],
        camera_height_cm: float,
        calibration_file: str,
        focal_spec_35mm: float = 26.0,
        sensor_width_mm: float = 36.0,
        min_obj_area: int = 900,
        sample_frames: int = 25,
    ) -> None:
        self.source = source
        self.camera_height_cm = float(camera_height_cm)
        self.calib_file = calibration_file
        self.focal_spec_35mm = float(focal_spec_35mm)
        self.sensor_width_mm = float(sensor_width_mm)
        self.min_obj_area = int(min_obj_area)
        self.sample_frames_limit = int(sample_frames)

        self.calib_mgr = CalibrationManager(self.calib_file)
        self.detector = ObjectDetector(min_area=self.min_obj_area)
        self.projector: Optional[PinholeProjector] = None

        # Interactive calibration state
        self.calibrating = False
        self.calib_clicks: List[Tuple[int, int]] = []
        self.mouse_pos: Tuple[int, int] = (0, 0)
        self.awaiting_length = False
        self.pending_pixel_len = 0.0
        self.length_queue: queue.Queue[str] = queue.Queue()

        # Measurement averaging state
        self.sampling = False
        self.sample_buffer: List[Tuple[int, int]] = []
        self.frozen_result: Optional[Tuple[float, float, float, float]] = None

        # Tracking state
        self.obj_center: Optional[Tuple[int, int]] = None
        self.obj_contour: Optional[np.ndarray] = None
        self.lost_frames = 0
        self.max_lost_frames = 8
        self.show_mask = False

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Mouse callback handling interactive point selection during calibration."""
        self.mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN and self.calibrating and len(self.calib_clicks) < 2:
            self.calib_clicks.append((x, y))

    def prompt_length_background(self, pixel_len: float) -> None:
        """Executes terminal input prompt on a background thread to prevent UI freezing."""
        try:
            prompt_msg = (
                f"\n>>> Measured distance: {pixel_len:.2f} px.\n"
                f"Enter known real-world distance between selected ruler marks (in cm): "
            )
            val = input(prompt_msg)
            self.length_queue.put(val)
        except (EOFError, KeyboardInterrupt):
            self.length_queue.put("")

    def initialize_projector(self, frame_w: int, frame_h: int) -> None:
        """Initializes projector with calibrated focal length or optical spec estimate."""
        calib_data = self.calib_mgr.load(frame_w, frame_h)

        if calib_data and not calib_data.get("mismatch"):
            loaded_focal = float(calib_data["focal_px"])
            self.projector = PinholeProjector(
                frame_w, frame_h, self.camera_height_cm,
                focal_px=loaded_focal, focal_source="ruler-verified (loaded from JSON)"
            )
            self.projector.focal_calibrated = True
            print(f"[Calibration] Loaded saved calibration from {self.calib_file}")
            print(f"              Focal length: {loaded_focal:.2f} px at resolution {frame_w}x{frame_h}")
        else:
            if calib_data and calib_data.get("mismatch"):
                print(f"[Calibration] Existing calibration resolution mismatch "
                      f"({calib_data.get('stored_w')}x{calib_data.get('stored_h')} vs {frame_w}x{frame_h}). "
                      f"Using optical specification fallback.")
            spec_focal = PinholeProjector.estimate_focal_from_spec(
                frame_w, self.focal_spec_35mm, self.sensor_width_mm
            )
            self.projector = PinholeProjector(
                frame_w, frame_h, self.camera_height_cm,
                focal_px=spec_focal,
                focal_source=f"optical spec ({self.focal_spec_35mm}mm equivalent, unverified)"
            )
            print(f"[Calibration] Initialized using optical spec: {spec_focal:.2f} px.")
            print("              Press 'k' to perform precise in-situ ruler calibration.")

    def run(self) -> int:
        """Main execution loop."""
        print(f"[System] Opening camera source: {self.source} ...")
        cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            sys.stderr.write(
                f"\n[Error] Unable to open camera source: '{self.source}'.\n\n"
                "Troubleshooting steps:\n"
                "1. If using a local webcam (e.g., 0, 1), check if camera permissions are granted and\n"
                "   ensure no other application is currently using the camera.\n"
                "2. If using an IP streaming URL, verify the phone/device is on the same network,\n"
                "   the streaming app is active, and the IP/port are reachable.\n"
                "3. Use the '--source' CLI option to supply a valid camera index or streaming URL.\n"
            )
            return 1

        ret, probe_frame = cap.read()
        if not ret or probe_frame is None:
            sys.stderr.write("[Error] Camera opened successfully but returned no frame data.\n")
            cap.release()
            return 1

        frame_h, frame_w = probe_frame.shape[:2]
        self.initialize_projector(frame_w, frame_h)
        assert self.projector is not None

        window_name = "Webcam Coordinate Tracker"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.on_mouse)

        print("\n" + "=" * 60)
        print("  Webcam Coordinate Tracker - Active")
        print("=" * 60)
        print(f"Resolution       : {frame_w} x {frame_h}")
        print(f"Origin (0,0)     : {self.projector.origin_px} (frame center)")
        print(f"Camera Height    : {self.camera_height_cm:.1f} cm")
        print("Controls:")
        print("  [k]   : Calibrate camera focal length using a physical ruler")
        print("  [s]   : Take averaged frozen measurement (25 frames)")
        print("  [r]   : Reset frozen measurement")
        print("  [m]   : Toggle foreground segmentation debug mask")
        print("  [Esc] : Quit application")
        print("=" * 60 + "\n")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[System] Stream ended or frame lost.")
                    break

                raw_frame = frame.copy()

                # --- 1. Interactive Ruler Calibration ---
                if self.calibrating:
                    cv2.putText(
                        frame,
                        "CALIBRATION: Click 2 ruler marks (magnified inset at top-right)",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 165, 255),
                        2,
                    )
                    for pt in self.calib_clicks:
                        cv2.circle(frame, pt, 5, (0, 165, 255), -1)

                    if len(self.calib_clicks) == 1:
                        cv2.line(frame, self.calib_clicks[0], self.mouse_pos, (0, 165, 255), 1)

                    render_zoom_inset(frame, raw_frame, self.mouse_pos)

                    if len(self.calib_clicks) == 2:
                        cv2.line(frame, self.calib_clicks[0], self.calib_clicks[1], (0, 165, 255), 2)
                        p1, p2 = self.calib_clicks[0], self.calib_clicks[1]
                        self.pending_pixel_len = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
                        self.calibrating = False
                        self.calib_clicks = []
                        self.awaiting_length = True

                        threading.Thread(
                            target=self.prompt_length_background,
                            args=(self.pending_pixel_len,),
                            daemon=True,
                        ).start()

                # --- 2. Asynchronous Terminal Input Handling ---
                if self.awaiting_length:
                    cv2.rectangle(
                        frame,
                        (0, frame_h // 2 - 32),
                        (frame_w, frame_h // 2 + 32),
                        (0, 0, 0),
                        -1,
                    )
                    cv2.putText(
                        frame,
                        "Switch to terminal and input real-world distance in cm (Press Enter)",
                        (20, frame_h // 2 + 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60,
                        (0, 255, 255),
                        2,
                    )
                    try:
                        input_text = self.length_queue.get_nowait()
                        try:
                            real_dist_cm = float(input_text.strip())
                            if real_dist_cm > 0:
                                new_focal = PinholeProjector.compute_focal_from_ruler(
                                    self.pending_pixel_len,
                                    self.camera_height_cm,
                                    real_dist_cm,
                                )
                                self.projector.focal_px = new_focal
                                self.projector.focal_calibrated = True
                                self.projector.focal_source = "ruler-verified (calibrated)"
                                scale_factor = self.projector.get_scale_cm_per_px()

                                print(f"\n[Calibration Success]")
                                print(f"  Focal Length : {new_focal:.2f} px")
                                print(f"  Scale Factor : {scale_factor:.4f} cm/px at H={self.camera_height_cm} cm")

                                if self.calib_mgr.save(new_focal, frame_w, frame_h, self.camera_height_cm):
                                    print(f"  Saved to     : {self.calib_file}")
                            else:
                                print("[Calibration Error] Distance must be greater than zero. Calibration discarded.")
                        except ValueError:
                            print("[Calibration Error] Non-numeric input received. Calibration discarded.")
                        self.awaiting_length = False
                    except queue.Empty:
                        pass

                # --- 3. Object Detection ---
                fg_mask, largest_contour, centroid = self.detector.process_frame(frame)
                if largest_contour is not None and centroid is not None:
                    self.obj_contour = largest_contour
                    self.obj_center = centroid
                    self.lost_frames = 0
                else:
                    self.lost_frames += 1
                    if self.lost_frames > self.max_lost_frames:
                        self.obj_contour = None
                        self.obj_center = None

                # --- 4. Multi-Frame Averaging (Sampling) ---
                if self.sampling:
                    if self.obj_center is not None:
                        self.sample_buffer.append(self.obj_center)

                    if len(self.sample_buffer) >= self.sample_frames_limit:
                        avg_px = sum(pt[0] for pt in self.sample_buffer) / len(self.sample_buffer)
                        avg_py = sum(pt[1] for pt in self.sample_buffer) / len(self.sample_buffer)

                        fx_cm, fy_cm = self.projector.pixel_to_world_cm(avg_px, avg_py)
                        f_res_cm = math.hypot(fx_cm, fy_cm)
                        f_ang_deg = math.degrees(math.atan2(-fy_cm, fx_cm))

                        self.frozen_result = (fx_cm, fy_cm, f_res_cm, f_ang_deg)
                        print(
                            f"[Measurement Frozen] X={fx_cm:+.2f} cm | Y={fy_cm:+.2f} cm | "
                            f"Resultant={f_res_cm:.2f} cm | Angle={f_ang_deg:+.1f} deg "
                            f"(n={len(self.sample_buffer)} frames)"
                        )
                        self.sampling = False
                        self.sample_buffer.clear()

                # --- 5. Render Virtual Origin & Coordinate Axes ---
                origin = self.projector.origin_px
                cv2.drawMarker(
                    frame,
                    origin,
                    (0, 0, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=24,
                    thickness=2,
                )
                cv2.putText(
                    frame,
                    "Origin (0,0)",
                    (origin[0] + 12, origin[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                )

                # --- 6. Render Detected Target & Real-Time Coordinates ---
                if self.obj_contour is not None and self.obj_center is not None:
                    cv2.drawContours(frame, [self.obj_contour], -1, (0, 255, 0), 2)
                    cv2.circle(frame, self.obj_center, 4, (0, 255, 0), -1)
                    cv2.line(frame, origin, self.obj_center, (255, 255, 0), 2)

                    obj_x_cm, obj_y_cm = self.projector.pixel_to_world_cm(*self.obj_center)
                    res_cm = math.hypot(obj_x_cm, obj_y_cm)

                    cv2.putText(
                        frame,
                        f"X: {obj_x_cm:+.1f} cm  Y: {obj_y_cm:+.1f} cm",
                        (self.obj_center[0] + 10, self.obj_center[1] - 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"R: {res_cm:.1f} cm",
                        (self.obj_center[0] + 10, self.obj_center[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 0),
                        2,
                    )

                # --- 7. Status HUD and Banners ---
                if self.sampling:
                    cv2.putText(
                        frame,
                        f"SAMPLING: {len(self.sample_buffer)}/{self.sample_frames_limit} frames "
                        "(hold object stationary)",
                        (10, frame_h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                    )

                if self.frozen_result is not None:
                    fx, fy, fr, fa = self.frozen_result
                    cv2.putText(
                        frame,
                        f"FROZEN: X={fx:+.2f}cm Y={fy:+.2f}cm R={fr:.2f}cm Angle={fa:+.1f}deg",
                        (10, frame_h - 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )

                status_label = (
                    "CALIBRATED (Ruler-verified)"
                    if self.projector.focal_calibrated
                    else f"ESTIMATED ({self.projector.focal_source}) [Press 'k' to calibrate]"
                )
                cv2.putText(
                    frame,
                    f"Focal: {status_label}",
                    (10, frame_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (200, 200, 200),
                    1,
                )

                cv2.imshow(window_name, frame)

                if self.show_mask:
                    cv2.imshow("Detection Mask (Debug)", fg_mask)
                else:
                    try:
                        if cv2.getWindowProperty("Detection Mask (Debug)", 0) >= 0:
                            cv2.destroyWindow("Detection Mask (Debug)")
                    except cv2.error:
                        pass

                # --- 8. Key Event Handling ---
                key = cv2.waitKey(30) & 0xFF
                if key == 27:  # Esc
                    print("[System] Terminated by user.")
                    break
                elif key == ord("k") and not self.calibrating and not self.awaiting_length:
                    self.calibrating = True
                    self.calib_clicks.clear()
                    print("\n[Calibration Started] Click two distinct tick marks on your ruler in the video window.")
                elif key == ord("s") and not self.sampling:
                    self.sampling = True
                    self.sample_buffer.clear()
                    print("[Measurement] Sampling initiated. Averaging positions across consecutive frames...")
                elif key == ord("r"):
                    self.frozen_result = None
                    print("[Measurement] Frozen reading cleared.")
                elif key == ord("m"):
                    self.show_mask = not self.show_mask

        finally:
            cv2.destroyAllWindows()
            cap.release()
            print("[System] Resources released cleanly.")

        return 0


# ==============================================================================
# Synthetic Self-Test Mode (Hardware-independent)
# ==============================================================================

def run_synthetic_self_test() -> int:
    """
    Executes a comprehensive verification test using synthetic images.
    Verifies projection equations, focal calculations, and background subtraction.
    Does not require a physical camera or graphical display.
    """
    print("[Self-Test] Running synthetic mathematical and computer vision tests...")

    # 1. Verification of focal length from spec
    w, h = 1920, 1080
    focal_spec = PinholeProjector.estimate_focal_from_spec(w, 26.0, 36.0)
    expected_focal = (1920 / 2.0) / math.tan(math.atan(36.0 / 52.0))
    assert math.isclose(focal_spec, expected_focal, rel_tol=1e-5), "Focal spec calculation error"
    print(f"  [Pass] Focal spec calculation ({focal_spec:.2f} px)")

    # 2. Verification of ruler calibration formula
    height_cm = 80.0
    pixel_dist = 200.0
    real_dist_cm = 10.0
    focal_calib = PinholeProjector.compute_focal_from_ruler(pixel_dist, height_cm, real_dist_cm)
    assert math.isclose(focal_calib, 1600.0, rel_tol=1e-5), "Ruler calibration math error"
    print(f"  [Pass] Ruler focal calibration calculation ({focal_calib:.2f} px)")

    # 3. Verification of pinhole projection
    projector = PinholeProjector(w, h, height_cm, focal_px=1600.0)
    origin_x, origin_y = projector.pixel_to_world_cm(960, 540)
    assert math.isclose(origin_x, 0.0) and math.isclose(origin_y, 0.0), "Origin projection mismatch"

    # Offset 160 pixels along X -> should be exactly 8.0 cm
    test_x, test_y = projector.pixel_to_world_cm(960 + 160, 540 - 320)
    assert math.isclose(test_x, 8.0, rel_tol=1e-5), f"World X projection error: {test_x}"
    assert math.isclose(test_y, -16.0, rel_tol=1e-5), f"World Y projection error: {test_y}"
    print("  [Pass] Pinhole coordinate projection logic")

    # 4. Verification of background subtractor and detector
    detector = ObjectDetector(min_area=400)
    bg = np.zeros((480, 640, 3), dtype=np.uint8)

    # Train background subtractor with 10 static frames
    for _ in range(10):
        detector.process_frame(bg)

    # Introduce object: filled white circle at (300, 200) with radius 25 (area ~1963 px)
    fg = bg.copy()
    cv2.circle(fg, (300, 200), 25, (255, 255, 255), -1)
    mask, contour, centroid = detector.process_frame(fg)

    assert contour is not None, "Detector failed to find synthetic object contour"
    assert centroid is not None, "Detector failed to compute centroid"
    assert abs(centroid[0] - 300) <= 2 and abs(centroid[1] - 200) <= 2, f"Centroid drift: {centroid}"
    print(f"  [Pass] Object detection & centroid extraction (detected at {centroid})")

    # 5. Verification of calibration manager JSON round-trip
    test_calib_file = "test_calib_tmp.json"
    mgr = CalibrationManager(test_calib_file)
    try:
        mgr.save(1600.0, 1920, 1080, 80.0)
        loaded = mgr.load(1920, 1080)
        assert loaded is not None and math.isclose(loaded["focal_px"], 1600.0)
        # Test resolution mismatch
        mismatched = mgr.load(1280, 720)
        assert mismatched is not None and mismatched.get("mismatch") is True
        print("  [Pass] Calibration persistence manager (save, load, resolution validation)")
    finally:
        if os.path.exists(test_calib_file):
            os.remove(test_calib_file)

    print("\n[Self-Test] All synthetic tests passed successfully.")
    return 0


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def build_cli_parser() -> argparse.ArgumentParser:
    """Builds the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="webcam_coordinate_tracker",
        description="2D Object Coordinate Measurement System via Overhead Camera Feed",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        default="0",
        help="Camera device index (e.g., '0', '1') or streaming URL (e.g., 'http://192.168.1.50:4747/video')",
    )
    parser.add_argument(
        "--height",
        "-H",
        type=float,
        default=79.0,
        help="Perpendicular distance from table surface to camera lens (in centimeters)",
    )
    parser.add_argument(
        "--calib-file",
        "-c",
        type=str,
        default="camera_calibration.json",
        help="Filepath for reading/writing persistent calibration parameters",
    )
    parser.add_argument(
        "--focal-spec",
        type=float,
        default=26.0,
        help="35mm-equivalent lens focal length (in mm) for initial theoretical approximation",
    )
    parser.add_argument(
        "--sensor-width",
        type=float,
        default=36.0,
        help="Reference full-frame sensor width (in mm) used for 35mm equivalent calculation",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=900,
        help="Minimum foreground contour pixel area threshold to trigger object tracking",
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=25,
        help="Number of consecutive frames averaged for a frozen jitter-free measurement",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run headless mathematical and synthetic computer vision self-tests and exit",
    )
    return parser


def main() -> int:
    """Application main entry point."""
    parser = build_cli_parser()
    args = parser.parse_args()

    if args.self_test:
        return run_synthetic_self_test()

    parsed_source = parse_camera_source(args.source)
    app = CoordinateTrackerApp(
        source=parsed_source,
        camera_height_cm=args.height,
        calibration_file=args.calib_file,
        focal_spec_35mm=args.focal_spec,
        sensor_width_mm=args.sensor_width,
        min_obj_area=args.min_area,
        sample_frames=args.sample_frames,
    )
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
