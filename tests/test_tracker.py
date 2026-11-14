"""
Unit Tests for Webcam Coordinate Tracker
========================================
Comprehensive automated tests covering projection equations, focal calculations,
persistence, contour centroid extraction, and CLI parser behavior.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import unittest

import cv2
import numpy as np

from webcam_coordinate_tracker import (
    CalibrationManager,
    ObjectDetector,
    PinholeProjector,
    build_cli_parser,
    parse_camera_source,
)


class TestPinholeProjector(unittest.TestCase):
    """Tests for PinholeProjector coordinate projections and focal length math."""

    def setUp(self) -> None:
        self.frame_w = 1280
        self.frame_h = 720
        self.camera_height = 80.0  # cm
        self.focal_px = 1000.0     # px
        self.projector = PinholeProjector(
            frame_width=self.frame_w,
            frame_height=self.frame_h,
            camera_height_cm=self.camera_height,
            focal_px=self.focal_px,
        )

    def test_origin_maps_to_zero(self) -> None:
        """The optical center (frame center) must project exactly to (0.0, 0.0) cm."""
        cx = self.frame_w // 2
        cy = self.frame_h // 2
        x_cm, y_cm = self.projector.pixel_to_world_cm(cx, cy)
        self.assertAlmostEqual(x_cm, 0.0, places=5)
        self.assertAlmostEqual(y_cm, 0.0, places=5)

    def test_quadrant_displacements(self) -> None:
        """Verify positive and negative offsets along both axes."""
        cx = self.frame_w // 2
        cy = self.frame_h // 2

        # Displace +100 px along X, -50 px along Y
        # X = 100 * 80 / 1000 = 8.0 cm
        # Y = -50 * 80 / 1000 = -4.0 cm
        x_cm, y_cm = self.projector.pixel_to_world_cm(cx + 100, cy - 50)
        self.assertAlmostEqual(x_cm, 8.0, places=5)
        self.assertAlmostEqual(y_cm, -4.0, places=5)

    def test_height_scaling_proportionality(self) -> None:
        """Doubling camera height must double real-world displacement for same pixel offset."""
        proj_high = PinholeProjector(
            frame_width=self.frame_w,
            frame_height=self.frame_h,
            camera_height_cm=160.0,
            focal_px=self.focal_px,
        )
        x1, y1 = self.projector.pixel_to_world_cm(740, 460)
        x2, y2 = proj_high.pixel_to_world_cm(740, 460)
        self.assertAlmostEqual(x2, 2.0 * x1, places=5)
        self.assertAlmostEqual(y2, 2.0 * y1, places=5)

    def test_scale_cm_per_px(self) -> None:
        """Scale factor must equal camera_height / focal_px."""
        expected_scale = 80.0 / 1000.0  # 0.08 cm/px
        self.assertAlmostEqual(self.projector.get_scale_cm_per_px(), expected_scale, places=5)

    def test_ruler_calibration_formula(self) -> None:
        """Verify focal length computation from ruler distance."""
        # 250 px corresponds to 20 cm at H=80 cm -> f = (250 * 80) / 20 = 1000 px
        computed_focal = PinholeProjector.compute_focal_from_ruler(250.0, 80.0, 20.0)
        self.assertAlmostEqual(computed_focal, 1000.0, places=5)

    def test_ruler_calibration_invalid_inputs(self) -> None:
        """Negative or zero inputs must raise ValueError."""
        with self.assertRaises(ValueError):
            PinholeProjector.compute_focal_from_ruler(-100.0, 80.0, 20.0)
        with self.assertRaises(ValueError):
            PinholeProjector.compute_focal_from_ruler(100.0, -80.0, 20.0)
        with self.assertRaises(ValueError):
            PinholeProjector.compute_focal_from_ruler(100.0, 80.0, 0.0)

    def test_estimate_focal_from_spec(self) -> None:
        """Verify 35mm optical equivalent estimation formula."""
        focal_px = PinholeProjector.estimate_focal_from_spec(
            frame_width=1920,
            focal_length_35mm_equiv=26.0,
            full_frame_sensor_width_mm=36.0,
        )
        hfov = 2.0 * math.atan(36.0 / (2.0 * 26.0))
        expected = (1920.0 / 2.0) / math.tan(hfov / 2.0)
        self.assertAlmostEqual(focal_px, expected, places=5)

    def test_estimate_focal_invalid_spec(self) -> None:
        """Zero or negative 35mm focal spec must raise ValueError."""
        with self.assertRaises(ValueError):
            PinholeProjector.estimate_focal_from_spec(1920, 0.0)


class TestCalibrationManager(unittest.TestCase):
    """Tests for persistent calibration storage and resolution verification."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.calib_file = os.path.join(self.temp_dir.name, "calib.json")
        self.manager = CalibrationManager(self.calib_file)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_save_and_load_successful_match(self) -> None:
        """Saved calibration must reload correctly when resolution matches."""
        success = self.manager.save(
            focal_px=1450.5,
            frame_width=1920,
            frame_height=1080,
            camera_height_cm=75.0,
        )
        self.assertTrue(success)
        self.assertTrue(os.path.exists(self.calib_file))

        loaded = self.manager.load(1920, 1080)
        self.assertIsNotNone(loaded)
        self.assertAlmostEqual(loaded["focal_px"], 1450.5)
        self.assertEqual(loaded["frame_w"], 1920)
        self.assertEqual(loaded["frame_h"], 1080)

    def test_resolution_mismatch_detected(self) -> None:
        """Loading with mismatched resolution must return mismatch dictionary."""
        self.manager.save(1450.5, 1920, 1080, 75.0)
        loaded = self.manager.load(1280, 720)
        self.assertIsNotNone(loaded)
        self.assertTrue(loaded.get("mismatch"))
        self.assertEqual(loaded.get("stored_w"), 1920)
        self.assertEqual(loaded.get("stored_h"), 1080)

    def test_missing_file_returns_none(self) -> None:
        """Loading a non-existent file returns None without throwing."""
        non_existent = CalibrationManager(os.path.join(self.temp_dir.name, "missing.json"))
        self.assertIsNone(non_existent.load(1920, 1080))

    def test_corrupted_json_returns_none(self) -> None:
        """Malformed JSON file should be gracefully handled and return None."""
        with open(self.calib_file, "w") as f:
            f.write("{invalid_json: true")
        self.assertIsNone(self.manager.load(1920, 1080))


class TestObjectDetector(unittest.TestCase):
    """Tests for computer vision background subtraction and centroid calculation."""

    def setUp(self) -> None:
        self.detector = ObjectDetector(min_area=300)

    def test_compute_centroid_valid_contour(self) -> None:
        """Moments of a synthetic square contour must produce the correct center."""
        # Square from (100, 100) to (140, 140) -> center is (120, 120)
        pts = np.array([[[100, 100]], [[140, 100]], [[140, 140]], [[100, 140]]], dtype=np.int32)
        cx, cy = self.detector.compute_centroid(pts)
        self.assertEqual(cx, 120)
        self.assertEqual(cy, 120)

    def test_compute_centroid_degenerate(self) -> None:
        """Contour with zero area returns None."""
        degenerate = np.array([[[10, 10]]], dtype=np.int32)
        self.assertIsNone(self.detector.compute_centroid(degenerate))

    def test_detection_pipeline_synthetic_scene(self) -> None:
        """Synthetic moving object must be detected after background model adapts."""
        canvas = np.zeros((400, 600, 3), dtype=np.uint8)
        # Adapt background model
        for _ in range(8):
            self.detector.process_frame(canvas)

        # Place target object: circle at (250, 200) with radius 20 (area ~1256 px)
        fg_scene = canvas.copy()
        cv2.circle(fg_scene, (250, 200), 20, (255, 255, 255), -1)

        mask, contour, centroid = self.detector.process_frame(fg_scene)
        self.assertIsNotNone(contour)
        self.assertIsNotNone(centroid)
        self.assertAlmostEqual(centroid[0], 250, delta=2)
        self.assertAlmostEqual(centroid[1], 200, delta=2)


class TestCLIAndUtils(unittest.TestCase):
    """Tests for CLI parameter parsing and utility helpers."""

    def test_parse_camera_source_numeric(self) -> None:
        """Numeric strings must be cast to integer device indices."""
        self.assertEqual(parse_camera_source("0"), 0)
        self.assertEqual(parse_camera_source(" 2 "), 2)

    def test_parse_camera_source_url(self) -> None:
        """URLs and paths must remain strings."""
        url = "http://192.168.1.100:4747/video"
        self.assertEqual(parse_camera_source(url), url)
        path = "samples/recording.mp4"
        self.assertEqual(parse_camera_source(path), path)

    def test_cli_parser_defaults(self) -> None:
        """Verify default argument parser values."""
        parser = build_cli_parser()
        args = parser.parse_args([])
        self.assertEqual(args.source, "0")
        self.assertAlmostEqual(args.height, 79.0)
        self.assertEqual(args.calib_file, "camera_calibration.json")
        self.assertAlmostEqual(args.focal_spec, 26.0)
        self.assertEqual(args.min_area, 900)
        self.assertEqual(args.sample_frames, 25)
        self.assertFalse(args.self_test)


if __name__ == "__main__":
    unittest.main()
