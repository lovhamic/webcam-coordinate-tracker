# Webcam Coordinate Tracker

A Python and OpenCV computer vision application that converts an overhead 2D video feed into a real-world metric coordinate measurement system. It detects objects placed on a planar surface and computes their real-world (X, Y) positions and straight-line displacement in centimeters relative to a calibrated virtual origin.

---

## Overview

The system models the camera as an orthogonal pinhole projector positioned perpendicularly above a flat working plane. By determining the camera focal length (either via manufacturer specifications or through an in-situ ruler calibration), pixel displacements from the image center are mapped directly to physical metric measurements.

The tool supports standard USB webcams, integrated cameras, and network video streams (HTTP/MJPEG/RTSP) from mobile IP camera applications.

---

## Mathematical Principle

Assuming the optical axis is perpendicular to the target surface at a fixed vertical height H:

```
X_cm = (u - cx) * H / f
Y_cm = (v - cy) * H / f
```

Where:
- `(u, v)`: Pixel coordinates of the detected object centroid.
- `(cx, cy)`: Image center coordinates representing the virtual origin `(0, 0)`.
- `H`: Vertical distance from the camera lens to the surface in centimeters (`--height`).
- `f`: Focal length in pixels (`focal_px`).

The distance from the origin (resultant) and angle are computed as:

```
Resultant = sqrt(X_cm^2 + Y_cm^2)
Angle = atan2(-Y_cm, X_cm)
```

### Determining Focal Length

The pixel focal length `f` is an intrinsic parameter of the camera and optical assembly:

1. **Theoretical Estimation (35mm Equivalent):**
   ```
   HFOV = 2 * atan(sensor_width / (2 * focal_length_35mm))
   focal_px = (frame_width / 2) / tan(HFOV / 2)
   ```
2. **Empirical In-Situ Calibration:**
   Placing a physical ruler of known length `L_cm` on the plane yields a measured pixel distance `D_px`. The exact focal length is then:
   ```
   focal_px = (D_px * H) / L_cm
   ```

---

## Features

- **Pinhole Projection Engine:** Real-time conversion from image plane to metric coordinates.
- **Flexible Video Ingestion:** Direct compatibility with USB webcams, RTSP streams, and mobile IP camera apps.
- **In-Situ Calibration:** Interactive two-point ruler calibration with a sub-pixel magnified cursor inset.
- **Resolution-Aware Persistence:** Calibrations are saved to JSON and automatically reloaded when matching resolution is detected.
- **Background Subtraction:** MOG2 background subtractor with morphological filtering for robust foreground object segmentation.
- **Jitter-Free Sampling:** Multi-frame coordinate averaging (default: 25 frames) to mitigate sensor noise and provide frozen readings.
- **Modular Architecture:** Fully decoupled projection, detection, calibration, and UI modules with high unit test coverage.
- **Headless Self-Test:** Built-in synthetic test mode verifying all mathematical calculations and detector routines without requiring physical hardware.

---

## Repository Structure

```
webcam_coordinate_tracker/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI workflow
├── tests/
│   ├── __init__.py
│   └── test_tracker.py          # Automated unit test suite
├── .gitignore                   # Git exclusion rules
├── CONTRIBUTING.md              # Contribution standards and workflow
├── LICENSE                      # MIT License
├── pyproject.toml               # PEP 517/621 package and build metadata
├── README.md                    # Project documentation
├── requirements.txt             # Core production dependencies
├── requirements-dev.txt         # Development and testing dependencies
└── webcam_coordinate_tracker.py # Main application and core engine
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Camera device (built-in webcam, external USB camera, or a mobile phone running an IP streaming application)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/webcam_coordinate_tracker.git
   cd webcam_coordinate_tracker
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Hardware Setup

1. Mount the camera directly above the working surface facing straight down. Ensure the optical axis is as close to 90 degrees (perpendicular) to the surface as possible.
2. Measure the vertical distance from the tabletop surface to the camera lens in centimeters.
3. If using a phone as an IP camera:
   - Install an IP webcam application (e.g., IP Webcam on Android or similar).
   - Ensure the computer and phone are connected to the same local network.
   - Note the stream endpoint URL (e.g., `http://192.168.1.50:4747/video`).

---

## Usage

### Basic Execution

To launch with a local USB webcam (default index `0`):
```bash
python webcam_coordinate_tracker.py
```

To specify an IP camera stream and measured height:
```bash
python webcam_coordinate_tracker.py --source "http://192.168.1.50:4747/video" --height 82.5
```

To use a secondary USB camera index:
```bash
python webcam_coordinate_tracker.py --source 1 --height 75.0
```

### Command-Line Arguments

| Argument | Short | Type | Default | Description |
|---|---|---|---|---|
| `--source` | `-s` | String | `0` | Device index (`0`, `1`) or streaming URL (`http://...`, `rtsp://...`) |
| `--height` | `-H` | Float | `79.0` | Vertical distance from table to lens in centimeters |
| `--calib-file` | `-c` | String | `camera_calibration.json` | Path to JSON file for calibration persistence |
| `--focal-spec` | | Float | `26.0` | 35mm-equivalent lens focal length in millimeters for theoretical estimation |
| `--sensor-width` | | Float | `36.0` | Full-frame reference sensor width in millimeters |
| `--min-area` | | Integer | `900` | Minimum pixel area required to track an object contour |
| `--sample-frames`| | Integer | `25` | Number of consecutive frames averaged for frozen measurement |
| `--self-test` | | Flag | `False` | Run synthetic mathematical and detector self-tests headlessly |

---

## Interactive Controls

The application window supports the following keyboard hotkeys:

| Key | Function | Description |
|---|---|---|
| `k` | Calibrate | Enters calibration mode. Click two ruler tick marks in the video feed, then enter the real distance in cm in the terminal. |
| `s` | Sample / Freeze | Accumulates coordinates across consecutive frames and freezes the averaged jitter-free measurement. |
| `r` | Reset | Clears the frozen measurement reading from the screen. |
| `m` | Debug Mask | Toggles display of the binary foreground segmentation mask. |
| `Esc` | Exit | Closes all windows and exits cleanly. |

---

## Calibration Workflow

For highest accuracy, perform a one-time ruler calibration:

1. Place a standard physical metric ruler flat on the working surface within the camera field of view.
2. Press `k` in the video window to activate calibration mode.
3. Click two distinct graduation marks on the ruler (for example, 0 cm and 15 cm). Use the magnified inset in the top-right corner to position crosshairs precisely.
4. When prompted in the terminal, type the physical distance between the clicked points in centimeters (e.g., `15.0`) and press Enter.
5. The computed focal length and scale factor will be saved to `camera_calibration.json`. On subsequent runs at the same resolution, this file is automatically loaded.

---

## Running Tests

### Automated Unit Tests

Run the complete test suite using the standard Python test discovery runner:

```bash
python -m unittest discover tests
```

### Synthetic Self-Test

Verify mathematical integrity and computer vision pipeline without hardware dependencies:

```bash
python webcam_coordinate_tracker.py --self-test
```

---

## Troubleshooting

### Camera Source Fails to Open
- **Webcams:** Verify that no other application (e.g., Zoom, Teams, browser) is using the camera. On Windows, check privacy settings to confirm camera access is enabled.
- **IP Cameras:** Confirm the host machine can reach the mobile device IP address using `ping <ip-address>`. Open the stream URL directly in a web browser or media player (such as VLC) to verify video transmission.

### Object Detection Instability
- **Lighting:** Avoid flickering overhead lights or direct sunlight creating rapid illumination changes.
- **Background Training:** The background subtractor requires a few seconds of initial stillness to adapt to the static environment.
- **Sensitivity:** Adjust `--min-area` to filter out smaller noise artifacts.

### Measurement Inaccuracy
- **Mount Angle:** Ensure the camera is pointed straight down. Any optical tilt introduces perspective distortion that violates planar pinhole assumptions.
- **Surface Elevation:** The object being tracked must reside on the same horizontal plane as the calibration ruler.
- **Resolution Changes:** If you change camera streaming resolution, repeat the calibration (`k`) to recalculate `focal_px`.

---

## Contributing

Contributions, bug reports, and enhancements are welcome. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for details on code standards and pull request workflows.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
