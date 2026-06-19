"""
Noise Inconsistency Detection Module
--------------------------------------
Authentic camera images have spatially consistent sensor noise patterns
(Photo Response Non-Uniformity – PRNU). Manipulated regions (spliced,
copy-moved, or AI-generated) typically break this consistency because
they originate from a different sensor or pipeline.

This module extracts a noise residual using wavelet denoising and
analyzes local noise variance to detect inconsistencies.
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage.restoration import denoise_wavelet, estimate_sigma


class NoiseAnalyzer:
    """
    Detects noise-level inconsistencies across image blocks.

    Steps:
      1. Convert to grayscale and extract noise residual.
      2. Divide image into overlapping blocks.
      3. Compute local noise variance per block.
      4. Flag blocks with outlier variance as potentially tampered.
    """

    def __init__(self, block_size: int = 64, overlap: int = 32):
        self.block_size = block_size
        self.overlap = overlap

    def analyze(self, image_path: str) -> dict:
        """
        Analyze noise inconsistency in an image.

        Returns:
            dict with keys:
              - noise_map (np.ndarray): per-pixel noise variance map
              - score (float): 0-100 manipulation probability
              - inconsistency_mask (np.ndarray): binary mask of suspect regions
              - suspicious_regions (list): bounding boxes
              - details (dict): block-level statistics
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # Extract noise residual (image - denoised)
        denoised = denoise_wavelet(
            gray, method="BayesShrink", mode="soft", channel_axis=None
        )
        noise_residual = gray - denoised

        # Build block-level variance map
        variance_map = self._compute_block_variance(noise_residual)

        # Detect outlier blocks
        mean_var = np.mean(variance_map)
        std_var = np.std(variance_map)

        if std_var < 1e-10:
            score = 0.0
        else:
            # Z-score based outlier detection
            z_scores = np.abs((variance_map - mean_var) / (std_var + 1e-10))
            outlier_ratio = float(np.mean(z_scores > 2.5))
            score = min(100.0, outlier_ratio * 500)

        # Upsample variance map to image size for visualization
        noise_map_full = cv2.resize(
            variance_map, (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

        # Create inconsistency mask
        threshold = mean_var + 2.5 * std_var
        low_threshold = max(0, mean_var - 2.5 * std_var)
        inconsistency_map = ((variance_map > threshold) | (variance_map < low_threshold)).astype(np.uint8)

        inconsistency_mask_full = cv2.resize(
            inconsistency_map * 255,
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint8)

        suspicious_regions = self._extract_regions(inconsistency_mask_full)

        global_sigma = float(estimate_sigma(gray, channel_axis=None))

        return {
            "noise_map": noise_map_full,
            "score": round(score, 2),
            "inconsistency_mask": inconsistency_mask_full,
            "suspicious_regions": suspicious_regions,
            "details": {
                "global_noise_sigma": round(global_sigma, 6),
                "block_variance_mean": round(float(mean_var), 6),
                "block_variance_std": round(float(std_var), 6),
                "outlier_block_ratio": round(score / 500 if score < 100 else 1.0, 4),
            },
        }

    def _compute_block_variance(self, noise_residual: np.ndarray) -> np.ndarray:
        """Compute per-block noise variance with sliding window."""
        h, w = noise_residual.shape
        step = self.block_size - self.overlap

        rows = max(1, (h - self.overlap) // step)
        cols = max(1, (w - self.overlap) // step)
        variance_map = np.zeros((rows, cols), dtype=np.float32)

        for i in range(rows):
            for j in range(cols):
                y_start = i * step
                x_start = j * step
                y_end = min(y_start + self.block_size, h)
                x_end = min(x_start + self.block_size, w)
                block = noise_residual[y_start:y_end, x_start:x_end]
                variance_map[i, j] = float(np.var(block))

        return variance_map

    def _extract_regions(self, mask: np.ndarray) -> list:
        """Extract bounding boxes from binary mask."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        min_area = mask.shape[0] * mask.shape[1] * 0.002
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                regions.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})

        return regions
