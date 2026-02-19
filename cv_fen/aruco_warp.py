import time
import cv2
import numpy as np
from picamera2 import Picamera2

# ---- Settings ----
BOARD_PIX = 800  # output warp size (800x800)
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Initialize Pi Camera
picam2 = Picamera2()
full_w, full_h = picam2.camera_properties["PixelArraySize"]
config = picam2.create_preview_configuration(
    main={"size": (1640, 1232), "format": "RGB888"},
    controls={"ScalerCrop": (0, 0, full_w, full_h)}
)
picam2.configure(config)
picam2.start()

# Marker IDs mapped to board corners (TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT)
ID_TL, ID_TR, ID_BR, ID_BL = 0, 1, 2, 3

# Destination points in the warped image
DST = np.array([
    [0, 0],                 # TL
    [BOARD_PIX - 1, 0],      # TR
    [BOARD_PIX - 1, BOARD_PIX - 1],  # BR
    [0, BOARD_PIX - 1],      # BL
], dtype=np.float32)

def capture_frame():
    """Capture a frame from the Pi camera"""
    frame = picam2.capture_array()
    # Convert from RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return img_bgr

def detect_markers(img_bgr):
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
    if ids is None:
        return {}, corners, ids

    ids = ids.flatten()
    found = {}
    for c, mid in zip(corners, ids):
        # c shape: (1,4,2)
        pts = c.reshape(4, 2).astype(np.float32)
        found[int(mid)] = pts
    return found, corners, ids

def marker_corner_for_board(marker_pts, which):
    """
    marker_pts: 4x2 points in order [top-left, top-right, bottom-right, bottom-left] for the marker itself.
    which: which corner of the marker to use.
    For board homography, we want the marker corner closest to the board corner.
    If you place markers outside the board, the marker corner nearest the board is:
      TL marker -> its bottom-right corner (index 2)
      TR marker -> its bottom-left corner  (index 3)
      BR marker -> its top-left corner     (index 0)
      BL marker -> its top-right corner    (index 1)
    """
    return marker_pts[which]

def warp_board(img_bgr, found):
    required = [ID_TL, ID_TR, ID_BR, ID_BL]
    if not all(mid in found for mid in required):
        return None

    # Choose the marker corner nearest the board area (assuming markers are OUTSIDE the board)
    src = np.array([
        marker_corner_for_board(found[ID_TL], 2),  # TL marker: bottom-right
        marker_corner_for_board(found[ID_TR], 3),  # TR marker: bottom-left
        marker_corner_for_board(found[ID_BR], 0),  # BR marker: top-left
        marker_corner_for_board(found[ID_BL], 1),  # BL marker: top-right
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, DST)
    warped = cv2.warpPerspective(img_bgr, H, (BOARD_PIX, BOARD_PIX))
    return warped

def main():
    print("Press q to quit.")
    try:
        while True:
            img = capture_frame()
            if img is None:
                print("Failed to capture frame...")
                time.sleep(0.2)
                continue

            found, corners, ids = detect_markers(img)

            # Draw detected markers on original image for debugging
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(img, corners, ids)

            warped = warp_board(img, found)

            cv2.imshow("Pi Camera (markers)", img)
            if warped is not None:
                cv2.imshow("Warped board (top-down)", warped)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
