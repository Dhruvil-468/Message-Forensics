"""
QR Code Reader module for Message Forensics.

Extracts text/URL payload from raw image bytes using OpenCV's built-in QRCodeDetector
without requiring external system libraries like zbar.
"""

import cv2
import numpy as np


def decode_qr(image_bytes: bytes) -> str | None:
    """
    Decodes a QR code from image bytes.
    Returns the decoded text/URL string if found, or None if no QR code could be decoded.
    """
    if not image_bytes or not isinstance(image_bytes, (bytes, bytearray)):
        return None

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None

        # Build list of detectors to try
        detectors = [cv2.QRCodeDetector()]
        if hasattr(cv2, "QRCodeDetectorAruco"):
            try:
                detectors.append(cv2.QRCodeDetectorAruco())
            except Exception:
                pass

        # Helper to test a detector on an image
        def _scan(detector, target_img):
            try:
                text, points, _ = detector.detectAndDecode(target_img)
                if text and isinstance(text, str) and text.strip():
                    return text.strip()
            except Exception:
                pass
            return None

        # 1. Try raw image
        for d in detectors:
            res = _scan(d, img)
            if res:
                return res

        # 2. Try grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for d in detectors:
            res = _scan(d, gray)
            if res:
                return res

        # 3. Try Otsu thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        for d in detectors:
            res = _scan(d, thresh)
            if res:
                return res

        # 4. Try resized (scaled up if small, scaled down if huge)
        h, w = gray.shape[:2]
        if max(h, w) > 2000:
            scale = 1600.0 / max(h, w)
            resized = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            for d in detectors:
                res = _scan(d, resized)
                if res:
                    return res
        elif min(h, w) < 300:
            scale = 600.0 / max(h, w)
            resized = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
            for d in detectors:
                res = _scan(d, resized)
                if res:
                    return res

        return None
    except Exception:
        return None
