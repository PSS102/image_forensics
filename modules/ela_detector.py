"""
Error Level Analysis (ELA) Module
----------------------------------
Detects tampering by re-saving an image at a known compression quality
and measuring the pixel-level difference. Authentic regions compress
uniformly; tampered regions show anomalous error levels.
"""

import numpy as np
import cv2
from PIL import Image
import io
import os
import tempfile


class ELADetector:
    """
    Performs Error Level Analysis on JPEG images.

    ELA works by:
      1. Re-saving the image at a fixed JPEG quality (default 90).
      2. Computing the absolute pixel difference between original and re-saved.
      3. Amplifying the difference for visualization.
      4. Scoring suspicious regions based on std-dev of error levels.
    """

    def __init__(self, quality: int = 90, amplification: int = 10):
        self.quality = quality
        self.amplification = amplification

    def analyze(self, image_path: str) -> dict:
        """
        Run ELA on an image.

        Returns:
            dict with keys:
              - ela_image (np.ndarray): amplified error map (RGB)
              - score (float): 0-100 manipulation probability
              - heatmap (np.ndarray): normalized single-channel heatmap
              - suspicious_regions (list): bounding boxes of anomalous zones
              - details (dict): raw stats
        """
        original = Image.open(image_path).convert("RGB")
        orig_array = np.array(original, dtype=np.float32)

        # Re-save at target quality into memory buffer
        buffer = io.BytesIO()
        original.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        recompressed = Image.open(buffer).convert("RGB")
        recomp_array = np.array(recompressed, dtype=np.float32)

        # Compute absolute difference and amplify
        diff = np.abs(orig_array - recomp_array) * self.amplification
        diff_clipped = np.clip(diff, 0, 255).astype(np.uint8)

        # Grayscale heatmap for analysis
        gray_diff = cv2.cvtColor(diff_clipped, cv2.COLOR_RGB2GRAY)
        heatmap_norm = gray_diff.astype(np.float32) / 255.0

        # Score: images with consistent low error are likely authentic
        mean_err = float(np.mean(gray_diff))
        std_err = float(np.std(gray_diff))

        # Higher std deviation = more regional inconsistency = more suspicious
        score = min(100.0, (std_err / 15.0) * 100)

        suspicious_regions = self._find_suspicious_regions(gray_diff)

        return {
            "ela_image": diff_clipped,
            "heatmap": heatmap_norm,
            "score": round(score, 2),
            "suspicious_regions": suspicious_regions,
            "details": {
                "mean_error_level": round(mean_err, 4),
                "std_error_level": round(std_err, 4),
                "quality_used": self.quality,
                "amplification": self.amplification,
            },
        }

    def _find_suspicious_regions(
        self, gray_diff: np.ndarray, threshold_percentile: float = 90.0
    ) -> list:
        """
        Identify bounding boxes around high-error regions.

        Returns list of (x, y, w, h) tuples.
        """
        threshold = np.percentile(gray_diff, threshold_percentile)
        binary = (gray_diff > threshold).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        min_area = gray_diff.shape[0] * gray_diff.shape[1] * 0.001  # 0.1% of image
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                regions.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})

        return regions
