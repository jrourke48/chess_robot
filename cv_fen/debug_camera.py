"""
Debug script to save what the camera sees and detect markers
Saves the raw image so you can check if markers are visible
"""

import cv2
import numpy as np
from picamera2 import Picamera2
import time

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Initialize camera
picam2 = Picamera2()
full_w, full_h = picam2.camera_properties["PixelArraySize"]
config = picam2.create_preview_configuration(
    main={"size": (1640, 1232), "format": "RGB888"},
    controls={"ScalerCrop": (0, 0, full_w, full_h)}
)
picam2.configure(config)
picam2.start()
time.sleep(2)

print("Capturing image...")

# Capture frame
frame = picam2.capture_array()
img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# Save raw capture
cv2.imwrite('camera_view_raw.jpg', img_bgr)
print("✓ Saved: camera_view_raw.jpg")

# Try to detect markers with relaxed parameters for better detection
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Use more lenient detection parameters
params = cv2.aruco.DetectorParameters()
params.adaptiveThreshWinSizeMin = 3
params.adaptiveThreshWinSizeMax = 23
params.adaptiveThreshWinSizeStep = 10
params.minMarkerPerimeterRate = 0.03
params.maxMarkerPerimeterRate = 4.0
params.polygonalApproxAccuracyRate = 0.05
params.minCornerDistanceRate = 0.05
params.minDistanceToBorder = 3
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

detector = cv2.aruco.ArucoDetector(DICT, params)
corners, ids, rejected = detector.detectMarkers(gray)

# Draw markers if found
img_marked = img_bgr.copy()
if ids is not None:
    cv2.aruco.drawDetectedMarkers(img_marked, corners, ids)
    print(f"✓ Found {len(ids)} markers: {ids.flatten().tolist()}")
else:
    print("✗ No markers detected")

# Draw rejected candidates
if len(rejected) > 0:
    cv2.aruco.drawDetectedMarkers(img_marked, rejected, borderColor=(0, 0, 255))
    print(f"⚠ {len(rejected)} rejected marker candidates (shown in red)")

cv2.imwrite('camera_view_markers.jpg', img_marked)
print("✓ Saved: camera_view_markers.jpg")

print("\nCheck the images:")
print("  - camera_view_raw.jpg: What the camera sees")
print("  - camera_view_markers.jpg: Detected markers (green) and rejected (red)")

if ids is None:
    print("\nTroubleshooting:")
    print("  1. Are all 4 ArUco markers visible in the camera view?")
    print("  2. Are they printed clearly and not blurry?")
    print("  3. Is the lighting good (no glare/shadows)?")
    print("  4. Are you using DICT_4X4_50 markers?")
    print("  5. Try moving the camera closer/farther from the board")

picam2.stop()
