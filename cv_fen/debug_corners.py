"""
Debug script to visualize which marker corners are being selected for warping
"""

import cv2
import numpy as np
from picamera2 import Picamera2
import time

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ID_TL, ID_TR, ID_BR, ID_BL = 0, 1, 2, 3

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

# Capture
frame = picam2.capture_array()
img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# Detect markers
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
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
corners, ids, _ = detector.detectMarkers(gray)

if ids is None or len(ids) < 4:
    print(f"Error: Only found {len(ids) if ids is not None else 0}/4 markers")
    picam2.stop()
    exit()

# Draw all markers
img_marked = img_bgr.copy()
cv2.aruco.drawDetectedMarkers(img_marked, corners, ids)

# Build found dictionary
found = {}
for c, mid in zip(corners, ids.flatten()):
    pts = c.reshape(4, 2).astype(np.float32)
    found[int(mid)] = pts

# Draw the corners being used for warping
colors = {
    ID_TL: (255, 0, 0),    # Blue
    ID_TR: (0, 255, 0),    # Green  
    ID_BR: (0, 0, 255),    # Red
    ID_BL: (255, 255, 0)   # Cyan
}

corner_indices = {
    ID_TL: 2,  # bottom-right of marker
    ID_TR: 3,  # bottom-left of marker
    ID_BR: 0,  # top-left of marker
    ID_BL: 1   # top-right of marker
}

labels = {
    ID_TL: "TL",
    ID_TR: "TR",
    ID_BR: "BR",
    ID_BL: "BL"
}

selected_corners = []
for marker_id in [ID_TL, ID_TR, ID_BR, ID_BL]:
    if marker_id in found:
        marker_pts = found[marker_id]
        corner_idx = corner_indices[marker_id]
        corner_pt = marker_pts[corner_idx]
        
        # Draw big circle on selected corner
        cv2.circle(img_marked, tuple(corner_pt.astype(int)), 15, colors[marker_id], -1)
        
        # Draw label
        text_pos = (int(corner_pt[0]) - 30, int(corner_pt[1]) - 20)
        cv2.putText(img_marked, labels[marker_id], text_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, 1, colors[marker_id], 3)
        
        selected_corners.append(corner_pt)
        print(f"Marker {marker_id} ({labels[marker_id]}): using corner {corner_idx} at {corner_pt}")

# Draw lines connecting the board corners
if len(selected_corners) == 4:
    pts = np.array(selected_corners, dtype=np.int32)
    cv2.polylines(img_marked, [pts], True, (255, 0, 255), 3)

cv2.imwrite('corner_selection_debug.jpg', img_marked)
print(f"\n✓ Saved: corner_selection_debug.jpg")
print("\nThe colored circles show which marker corner is being used:")
print("  Blue (TL) = Top-Left board corner")
print("  Green (TR) = Top-Right board corner")
print("  Red (BR) = Bottom-Right board corner")
print("  Cyan (BL) = Bottom-Left board corner")
print("\nThe magenta line shows the quadrilateral that will be warped.")
print("\nIf these corners don't match your actual board corners,")
print("you need to adjust the corner_indices in the code.")

picam2.stop()
