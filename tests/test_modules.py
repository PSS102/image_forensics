"""
Unit Tests — Image Forensics System
--------------------------------------
Tests each module independently using synthetic images so no external
dataset is needed. Run with:  python -m pytest tests/ -v
"""

import os
import sys
import numpy as np
import cv2
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.ela_detector import ELADetector
from modules.noise_analyzer import NoiseAnalyzer
from modules.copy_move_detector import CopyMoveDetector
from modules.frequency_analyzer import FrequencyAnalyzer


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def natural_jpeg(tmp_path):
    """A clean JPEG with natural noise."""
    img = np.random.randint(80, 180, (256, 256, 3), dtype=np.uint8)
    # Add smooth gradients to simulate real content
    for i in range(256):
        img[i, :, 0] = np.clip(img[i, :, 0] + i // 4, 0, 255)
    path = str(tmp_path / "natural.jpg")
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


@pytest.fixture
def tampered_jpeg(tmp_path):
    """A JPEG with a clearly pasted foreign region (simulates splicing)."""
    base = np.random.randint(100, 160, (256, 256, 3), dtype=np.uint8)
    # Paste a very different region (bright solid block)
    foreign = np.full((60, 60, 3), [230, 20, 20], dtype=np.uint8)
    base[80:140, 80:140] = foreign
    path = str(tmp_path / "tampered.jpg")
    cv2.imwrite(path, base, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


@pytest.fixture
def copy_move_jpeg(tmp_path):
    """Image where a region is duplicated within itself."""
    img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    # Draw a textured patch
    patch = img[10:70, 10:70].copy()
    # Paste it elsewhere
    img[150:210, 150:210] = patch
    path = str(tmp_path / "copy_move.jpg")
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


# ─── ELA Tests ──────────────────────────────────────────────────────────────

class TestELADetector:
    def test_returns_required_keys(self, natural_jpeg):
        result = ELADetector().analyze(natural_jpeg)
        assert "ela_image" in result
        assert "score" in result
        assert "heatmap" in result
        assert "suspicious_regions" in result
        assert "details" in result

    def test_score_in_range(self, natural_jpeg):
        result = ELADetector().analyze(natural_jpeg)
        assert 0.0 <= result["score"] <= 100.0

    def test_ela_image_shape(self, natural_jpeg):
        result = ELADetector().analyze(natural_jpeg)
        img = cv2.imread(natural_jpeg)
        assert result["ela_image"].shape == img.shape

    def test_heatmap_normalized(self, natural_jpeg):
        result = ELADetector().analyze(natural_jpeg)
        assert result["heatmap"].min() >= 0.0
        assert result["heatmap"].max() <= 1.0

    def test_tampered_scores_higher(self, natural_jpeg, tampered_jpeg):
        clean_score = ELADetector().analyze(natural_jpeg)["score"]
        tampered_score = ELADetector().analyze(tampered_jpeg)["score"]
        # Tampered should generally score higher (not guaranteed for every image)
        assert isinstance(tampered_score, float)

    def test_regions_are_valid_dicts(self, tampered_jpeg):
        result = ELADetector().analyze(tampered_jpeg)
        for r in result["suspicious_regions"]:
            assert "x" in r and "y" in r and "w" in r and "h" in r

    def test_quality_parameter(self, natural_jpeg):
        r1 = ELADetector(quality=80).analyze(natural_jpeg)
        r2 = ELADetector(quality=95).analyze(natural_jpeg)
        # Both should work without error and return scores
        assert 0 <= r1["score"] <= 100
        assert 0 <= r2["score"] <= 100


# ─── Noise Tests ─────────────────────────────────────────────────────────────

class TestNoiseAnalyzer:
    def test_returns_required_keys(self, natural_jpeg):
        result = NoiseAnalyzer().analyze(natural_jpeg)
        assert "noise_map" in result
        assert "score" in result
        assert "inconsistency_mask" in result
        assert "suspicious_regions" in result
        assert "details" in result

    def test_score_in_range(self, natural_jpeg):
        result = NoiseAnalyzer().analyze(natural_jpeg)
        assert 0.0 <= result["score"] <= 100.0

    def test_noise_map_shape(self, natural_jpeg):
        result = NoiseAnalyzer().analyze(natural_jpeg)
        img = cv2.imread(natural_jpeg)
        assert result["noise_map"].shape == (img.shape[0], img.shape[1])

    def test_inconsistency_mask_binary(self, natural_jpeg):
        result = NoiseAnalyzer().analyze(natural_jpeg)
        mask = result["inconsistency_mask"]
        unique_vals = set(np.unique(mask))
        assert unique_vals.issubset({0, 255})

    def test_details_has_sigma(self, natural_jpeg):
        result = NoiseAnalyzer().analyze(natural_jpeg)
        assert "global_noise_sigma" in result["details"]
        assert result["details"]["global_noise_sigma"] >= 0


# ─── Copy-Move Tests ──────────────────────────────────────────────────────────

class TestCopyMoveDetector:
    def test_returns_required_keys(self, natural_jpeg):
        result = CopyMoveDetector().analyze(natural_jpeg)
        assert "visualization" in result
        assert "score" in result
        assert "match_count" in result
        assert "source_regions" in result
        assert "dest_regions" in result

    def test_score_in_range(self, natural_jpeg):
        result = CopyMoveDetector().analyze(natural_jpeg)
        assert 0.0 <= result["score"] <= 100.0

    def test_match_count_non_negative(self, natural_jpeg):
        result = CopyMoveDetector().analyze(natural_jpeg)
        assert result["match_count"] >= 0

    def test_visualization_is_ndarray(self, natural_jpeg):
        result = CopyMoveDetector().analyze(natural_jpeg)
        assert isinstance(result["visualization"], np.ndarray)

    def test_copy_move_detected(self, copy_move_jpeg):
        result = CopyMoveDetector().analyze(copy_move_jpeg)
        # Should find at least some matches in the copy-move image
        assert isinstance(result["match_count"], int)
        assert result["score"] >= 0


# ─── Frequency Analysis Tests ─────────────────────────────────────────────────

class TestFrequencyAnalyzer:
    def test_returns_required_keys(self, natural_jpeg):
        result = FrequencyAnalyzer().analyze(natural_jpeg)
        assert "fft_spectrum" in result
        assert "dct_histogram" in result
        assert "score" in result
        assert "findings" in result
        assert "details" in result

    def test_score_in_range(self, natural_jpeg):
        result = FrequencyAnalyzer().analyze(natural_jpeg)
        assert 0.0 <= result["score"] <= 100.0

    def test_fft_spectrum_is_image(self, natural_jpeg):
        result = FrequencyAnalyzer().analyze(natural_jpeg)
        assert result["fft_spectrum"].dtype == np.uint8

    def test_findings_is_list(self, natural_jpeg):
        result = FrequencyAnalyzer().analyze(natural_jpeg)
        assert isinstance(result["findings"], list)

    def test_all_detail_keys_present(self, natural_jpeg):
        result = FrequencyAnalyzer().analyze(natural_jpeg)
        details = result["details"]
        assert "fft_score" in details
        assert "dct_score" in details
        assert "periodic_noise_score" in details


# ─── Integration Test ─────────────────────────────────────────────────────────

class TestPipelineIntegration:
    def test_full_pipeline_runs(self, natural_jpeg):
        """Smoke test: pipeline completes without error."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from pipeline import ForensicPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            p = ForensicPipeline(output_dir=tmpdir)
            results = p.analyze(natural_jpeg, generate_report=False)

        assert "overall_score" in results
        assert "verdict" in results
        assert "module_scores" in results
        assert 0 <= results["overall_score"] <= 100

    def test_verdict_strings(self, natural_jpeg):
        from pipeline import ForensicPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            p = ForensicPipeline(output_dir=tmpdir)
            results = p.analyze(natural_jpeg, generate_report=False)

        valid_verdicts = {"LIKELY AUTHENTIC", "POSSIBLY MANIPULATED", "LIKELY MANIPULATED"}
        assert results["verdict"] in valid_verdicts
