"""
Frequency Domain Analysis Module
-----------------------------------
Analyzes the DCT (Discrete Cosine Transform) and FFT spectrum of images
to detect:

  1. Double JPEG compression — leaves periodic artifacts in DCT histograms
     (tell-tale "dips" at multiples of the original quality quantization step).

  2. AI-generated image artifacts — generative models (GANs, diffusion)
     produce characteristic high-frequency spectral signatures that differ
     from real camera images.

  3. Upscaling/resizing artifacts — interpolation leaves spectral peaks
     at frequencies corresponding to the scale factor.
"""

import numpy as np
import cv2
from scipy.fft import fft2, fftshift
from scipy.signal import find_peaks


class FrequencyAnalyzer:
    """
    DCT and FFT-based forensic analysis.
    """

    def analyze(self, image_path: str) -> dict:
        """
        Perform frequency domain analysis.

        Returns:
            dict with keys:
              - fft_spectrum (np.ndarray): log-scale FFT magnitude image
              - dct_histogram (np.ndarray): DCT coefficient histogram
              - score (float): 0-100 manipulation probability
              - findings (list): human-readable anomaly descriptions
              - details (dict): raw statistics
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        findings = []
        scores = []

        # --- FFT Analysis ---
        fft_result = fft2(gray)
        fft_shifted = fftshift(fft_result)
        magnitude_spectrum = np.log1p(np.abs(fft_shifted))
        magnitude_spectrum_norm = cv2.normalize(
            magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        fft_score, fft_findings = self._analyze_fft(magnitude_spectrum)
        scores.append(fft_score)
        findings.extend(fft_findings)

        # --- DCT Analysis ---
        dct_coeffs = cv2.dct(gray)
        dct_hist, dct_score, dct_findings = self._analyze_dct(dct_coeffs)
        scores.append(dct_score)
        findings.extend(dct_findings)

        # --- Periodic noise detection ---
        periodic_score, periodic_findings = self._detect_periodic_noise(magnitude_spectrum)
        scores.append(periodic_score)
        findings.extend(periodic_findings)

        # Aggregate score
        overall_score = min(100.0, max(scores) * 0.6 + np.mean(scores) * 0.4)

        return {
            "fft_spectrum": magnitude_spectrum_norm,
            "dct_histogram": dct_hist,
            "score": round(overall_score, 2),
            "findings": findings,
            "details": {
                "fft_score": round(fft_score, 2),
                "dct_score": round(dct_score, 2),
                "periodic_noise_score": round(periodic_score, 2),
                "image_size": f"{gray.shape[1]}x{gray.shape[0]}",
            },
        }

    def _analyze_fft(self, magnitude_spectrum: np.ndarray) -> tuple:
        """Analyze FFT for AI-generation and resizing artifacts."""
        findings = []
        score = 0.0

        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2

        # Compare high-frequency energy ratio
        # Natural images have steep falloff; AI images and resized images differ
        inner_radius = min(h, w) // 8
        outer_radius = min(h, w) // 3

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)

        inner_mask = dist_from_center < inner_radius
        mid_mask = (dist_from_center >= inner_radius) & (dist_from_center < outer_radius)
        outer_mask = dist_from_center >= outer_radius

        inner_energy = float(np.mean(magnitude_spectrum[inner_mask]))
        mid_energy = float(np.mean(magnitude_spectrum[mid_mask]))
        outer_energy = float(np.mean(magnitude_spectrum[outer_mask]))

        total = inner_energy + mid_energy + outer_energy + 1e-10
        high_freq_ratio = outer_energy / total

        # Natural images: high_freq_ratio usually < 0.15
        if high_freq_ratio > 0.25:
            score += 40
            findings.append(
                f"Unusually high-frequency energy ({high_freq_ratio:.2%}) — "
                "possible AI generation or aggressive sharpening"
            )
        elif high_freq_ratio > 0.18:
            score += 15
            findings.append(
                f"Elevated high-frequency content ({high_freq_ratio:.2%}) — "
                "possible upscaling or enhancement"
            )

        return min(100.0, score), findings

    def _analyze_dct(self, dct_coeffs: np.ndarray) -> tuple:
        """
        Analyze DCT coefficients for double-compression artifacts.

        Double-JPEG leaves periodic dips in the DCT coefficient histogram
        at multiples of the quantization step of the first compression.
        """
        findings = []
        score = 0.0

        # Focus on AC coefficients (exclude DC at [0,0])
        ac_coeffs = dct_coeffs.flatten()
        ac_coeffs = ac_coeffs[np.abs(ac_coeffs) < np.percentile(np.abs(ac_coeffs), 99)]

        # Build histogram
        hist_range = (-200, 200)
        hist, bin_edges = np.histogram(ac_coeffs, bins=400, range=hist_range)
        hist_norm = hist.astype(np.float32) / (hist.sum() + 1e-10)

        # Detect periodic dips (double compression signature)
        # Smooth histogram and find valleys
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(hist_norm, sigma=2)

        # Find local minima in center region (skip extremes)
        center_start, center_end = 150, 250
        center_hist = smoothed[center_start:center_end]

        if len(center_hist) > 10:
            mean_val = np.mean(center_hist)
            dip_threshold = mean_val * 0.6
            dips = np.where(center_hist < dip_threshold)[0]

            # Check for periodicity in dips
            if len(dips) >= 4:
                dip_diffs = np.diff(dips)
                if len(dip_diffs) >= 3:
                    diff_std = np.std(dip_diffs)
                    diff_mean = np.mean(dip_diffs)
                    if diff_std < diff_mean * 0.5 and 4 < diff_mean < 30:
                        score += 60
                        findings.append(
                            f"Periodic DCT histogram dips detected (spacing ≈ {diff_mean:.1f} bins) — "
                            "strong indicator of double JPEG compression"
                        )

        # Check coefficient distribution kurtosis
        kurtosis = float(self._kurtosis(ac_coeffs))
        if kurtosis < 5.0:
            score += 20
            findings.append(
                f"Low DCT coefficient kurtosis ({kurtosis:.2f}) — "
                "may indicate heavy re-compression or AI generation"
            )

        # Return histogram as image
        hist_img = self._histogram_to_image(hist_norm)

        return hist_img, min(100.0, score), findings

    def _detect_periodic_noise(self, magnitude_spectrum: np.ndarray) -> tuple:
        """Detect periodic noise patterns (grid-like artifacts from splicing)."""
        findings = []
        score = 0.0

        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2

        # Check horizontal and vertical line profiles through spectrum center
        h_profile = magnitude_spectrum[center_y, :]
        v_profile = magnitude_spectrum[:, center_x]

        # Find sharp peaks away from center (periodic grid noise)
        def count_off_center_peaks(profile, center, min_dist=10):
            profile_float = profile.astype(np.float64)
            mean_val = np.mean(profile_float)
            std_val = np.std(profile_float)
            threshold = mean_val + 3 * std_val
            peaks, _ = find_peaks(profile_float, height=threshold, distance=5)
            off_center = [p for p in peaks if abs(p - center) > min_dist]
            return len(off_center)

        h_peaks = count_off_center_peaks(h_profile, center_x)
        v_peaks = count_off_center_peaks(v_profile, center_y)

        if h_peaks >= 4 or v_peaks >= 4:
            score += 50
            findings.append(
                f"Periodic spectral peaks detected (H:{h_peaks}, V:{v_peaks}) — "
                "possible grid noise from image stitching or splicing"
            )
        elif h_peaks >= 2 or v_peaks >= 2:
            score += 20
            findings.append(
                f"Minor spectral anomalies detected (H:{h_peaks}, V:{v_peaks})"
            )

        return min(100.0, score), findings

    def _kurtosis(self, x: np.ndarray) -> float:
        """Compute excess kurtosis."""
        n = len(x)
        if n < 4:
            return 0.0
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-10:
            return 0.0
        return float(np.mean(((x - mean) / std) ** 4) - 3)

    def _histogram_to_image(self, hist_norm: np.ndarray, h: int = 128, w: int = 400) -> np.ndarray:
        """Render a histogram as a grayscale image array."""
        img = np.ones((h, len(hist_norm)), dtype=np.uint8) * 255
        max_val = max(hist_norm.max(), 1e-10)
        for i, val in enumerate(hist_norm):
            bar_h = int((val / max_val) * h)
            img[h - bar_h:, i] = 50
        return cv2.resize(img, (w, h))
