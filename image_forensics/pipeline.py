"""
Image Forensics Pipeline
--------------------------
Orchestrates all detection modules and produces a unified forensic result:

  1. Metadata Analysis
  2. Error Level Analysis (ELA)
  3. Noise Inconsistency Detection
  4. Copy-Move Forgery Detection
  5. Frequency Domain Analysis

Each module returns a score (0-100). Scores are fused via weighted ensemble
into an overall manipulation probability with verdict and summary.
"""

import os
import cv2
import numpy as np
from PIL import Image
from datetime import datetime

from modules.ela_detector import ELADetector
from modules.noise_analyzer import NoiseAnalyzer
from modules.copy_move_detector import CopyMoveDetector
from modules.metadata_analyzer import MetadataAnalyzer
from modules.frequency_analyzer import FrequencyAnalyzer
from modules.report_generator import ForensicReportGenerator


# Module weights for ensemble fusion
MODULE_WEIGHTS = {
    "Metadata Analysis":        0.20,
    "Error Level Analysis":     0.25,
    "Noise Inconsistency":      0.20,
    "Copy-Move Detection":      0.20,
    "Frequency Analysis":       0.15,
}

VERDICT_THRESHOLDS = {
    70: "LIKELY MANIPULATED",
    40: "POSSIBLY MANIPULATED",
    0:  "LIKELY AUTHENTIC",
}


class ForensicPipeline:
    """
    End-to-end image forensics pipeline.
    """

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.modules = {
            "Metadata Analysis":    MetadataAnalyzer(),
            "Error Level Analysis": ELADetector(),
            "Noise Inconsistency":  NoiseAnalyzer(),
            "Copy-Move Detection":  CopyMoveDetector(),
            "Frequency Analysis":   FrequencyAnalyzer(),
        }
        self.report_gen = ForensicReportGenerator()

    def analyze(self, image_path: str, generate_report: bool = True) -> dict:
        """
        Run full forensic analysis on an image.

        Args:
            image_path: path to the image file
            generate_report: whether to produce a PDF report

        Returns:
            Comprehensive results dict.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        print(f"\n{'='*60}")
        print(f"  IMAGE FORENSICS ANALYSIS")
        print(f"  File: {os.path.basename(image_path)}")
        print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

        # Get image info
        img_info = self._get_image_info(image_path)

        module_results = {}
        module_scores = {}
        image_outputs = {}

        # --- Run each module ---
        for module_name, module in self.modules.items():
            print(f"[→] Running {module_name}...")
            try:
                result = module.analyze(image_path)
                module_results[module_name] = result
                score = result.get("score", 0.0)

                # Extract key finding
                flags = result.get("flags", [])
                findings = result.get("findings", [])
                all_items = flags + findings
                key_finding = all_items[0] if all_items else "No anomalies detected"

                module_scores[module_name] = {
                    "score": score,
                    "key_finding": key_finding,
                    "flags": flags,
                    "findings": findings,
                }
                print(f"    Score: {score:.1f}% | {key_finding[:70]}")

            except Exception as e:
                print(f"    [ERROR] {module_name} failed: {e}")
                module_scores[module_name] = {
                    "score": 0.0,
                    "key_finding": f"Module error: {str(e)[:60]}",
                    "flags": [],
                    "findings": [],
                }

        # --- Collect image outputs for visualization ---
        ela_result = module_results.get("Error Level Analysis", {})
        noise_result = module_results.get("Noise Inconsistency", {})
        cm_result = module_results.get("Copy-Move Detection", {})
        freq_result = module_results.get("Frequency Analysis", {})

        # Annotated original with all suspicious region boxes
        annotated = self._build_annotated_image(image_path, module_results)

        image_outputs = {
            "Original (Annotated)": annotated,
            "ELA — Error Level Map": ela_result.get("ela_image"),
            "Noise — Inconsistency Heatmap": self._normalize_to_color(
                noise_result.get("noise_map")
            ),
            "Copy-Move — Match Visualization": cm_result.get("visualization"),
            "FFT — Frequency Spectrum": freq_result.get("fft_spectrum"),
            "DCT — Coefficient Histogram": freq_result.get("dct_histogram"),
        }

        # --- Fuse scores ---
        overall_score = self._fuse_scores(module_scores)
        verdict = self._get_verdict(overall_score)
        manipulation_type = self._classify_manipulation(module_scores, module_results)
        summary_text = self._build_summary(overall_score, module_scores, manipulation_type)

        print(f"\n{'='*60}")
        print(f"  OVERALL SCORE:   {overall_score:.1f}%")
        print(f"  VERDICT:         {verdict}")
        print(f"  MANIPULATION:    {manipulation_type}")
        print(f"{'='*60}\n")

        results = {
            "image_path": image_path,
            "image_size": img_info["size"],
            "image_format": img_info["format"],
            "overall_score": overall_score,
            "verdict": verdict,
            "manipulation_type": manipulation_type,
            "summary_text": summary_text,
            "module_scores": module_scores,
            "module_results": module_results,
            "image_outputs": image_outputs,
            "raw_metadata": module_results.get("Metadata Analysis", {}).get("metadata", {}),
            "timestamp": datetime.now().isoformat(),
        }

        # --- Generate PDF report ---
        if generate_report:
            basename = os.path.splitext(os.path.basename(image_path))[0]
            report_path = os.path.join(
                self.output_dir,
                f"forensic_report_{basename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )
            print(f"[→] Generating PDF report...")
            self.report_gen.generate(results, report_path)
            results["report_path"] = report_path
            print(f"    Report saved: {report_path}")

        return results

    def _fuse_scores(self, module_scores: dict) -> float:
        """Weighted ensemble fusion of module scores."""
        total_weight = 0.0
        weighted_sum = 0.0

        for name, weight in MODULE_WEIGHTS.items():
            if name in module_scores:
                score = module_scores[name]["score"]
                weighted_sum += score * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        base_score = weighted_sum / total_weight

        # Boost if multiple modules are highly suspicious (corroboration)
        high_count = sum(
            1 for info in module_scores.values() if info["score"] >= 60
        )
        if high_count >= 3:
            base_score = min(100, base_score * 1.15)
        elif high_count >= 2:
            base_score = min(100, base_score * 1.08)

        return round(base_score, 2)

    def _get_verdict(self, score: float) -> str:
        for threshold in sorted(VERDICT_THRESHOLDS.keys(), reverse=True):
            if score >= threshold:
                return VERDICT_THRESHOLDS[threshold]
        return "LIKELY AUTHENTIC"

    def _classify_manipulation(self, module_scores: dict, module_results: dict) -> str:
        """Identify which manipulation type(s) are most likely."""
        types = []

        # Check metadata flags
        meta_flags = module_scores.get("Metadata Analysis", {}).get("flags", [])
        for f in meta_flags:
            if "editing software" in f.lower():
                types.append("Software Editing")
                break

        # Check ELA score
        ela_score = module_scores.get("Error Level Analysis", {}).get("score", 0)
        if ela_score >= 50:
            types.append("Regional Pixel Editing")

        # Check copy-move
        cm_result = module_results.get("Copy-Move Detection", {})
        if cm_result.get("match_count", 0) >= 5:
            types.append("Copy-Move Forgery")

        # Check noise
        noise_score = module_scores.get("Noise Inconsistency", {}).get("score", 0)
        if noise_score >= 50:
            types.append("Image Splicing")

        # Check frequency
        freq_findings = module_scores.get("Frequency Analysis", {}).get("findings", [])
        for f in freq_findings:
            if "double jpeg" in f.lower() or "double compression" in f.lower():
                types.append("Double JPEG Compression")
                break
            if "ai generation" in f.lower():
                types.append("AI Generation")
                break

        return ", ".join(types) if types else "None Detected"

    def _build_summary(
        self, score: float, module_scores: dict, manipulation_type: str
    ) -> str:
        if score >= 70:
            intro = (
                f"Forensic analysis indicates a HIGH probability ({score:.1f}%) that this image "
                "has been digitally manipulated. "
            )
        elif score >= 40:
            intro = (
                f"Forensic analysis reveals MODERATE indicators ({score:.1f}%) of possible "
                "image manipulation. "
            )
        else:
            intro = (
                f"Forensic analysis found LOW evidence ({score:.1f}%) of manipulation. "
                "The image appears largely authentic. "
            )

        # Add module highlights
        highlights = []
        for name, info in module_scores.items():
            if info["score"] >= 40 and info.get("key_finding") and "error" not in info["key_finding"].lower():
                highlights.append(f"{name} ({info['score']:.0f}%): {info['key_finding'][:80]}")

        if highlights:
            intro += "Key findings: " + "; ".join(highlights[:3]) + "."

        return intro

    def _get_image_info(self, image_path: str) -> dict:
        try:
            img = Image.open(image_path)
            return {
                "size": f"{img.width}×{img.height}px",
                "format": img.format or os.path.splitext(image_path)[1].upper().strip("."),
            }
        except Exception:
            return {"size": "Unknown", "format": "Unknown"}

    def _build_annotated_image(self, image_path: str, module_results: dict) -> np.ndarray:
        """Draw all suspicious regions from all modules onto the original image."""
        img = cv2.imread(image_path)
        if img is None:
            return None

        overlay = img.copy()
        colors_map = {
            "Error Level Analysis": (0, 165, 255),   # orange
            "Noise Inconsistency":  (255, 0, 255),    # magenta
            "Copy-Move Detection":  (0, 0, 255),      # red
        }

        for module_name, color in colors_map.items():
            result = module_results.get(module_name, {})
            regions = result.get("suspicious_regions", [])
            for r in regions:
                cv2.rectangle(
                    overlay,
                    (r["x"], r["y"]),
                    (r["x"] + r["w"], r["y"] + r["h"]),
                    color, 2
                )

        # Add legend
        legend_items = [
            ("ELA Regions", (0, 165, 255)),
            ("Noise Anomaly", (255, 0, 255)),
            ("Copy-Move", (0, 0, 255)),
        ]
        y_pos = 20
        for label, color in legend_items:
            cv2.rectangle(overlay, (10, y_pos - 12), (30, y_pos + 2), color, -1)
            cv2.putText(overlay, label, (35, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            y_pos += 22

        return cv2.addWeighted(overlay, 0.85, img, 0.15, 0)

    def _normalize_to_color(self, array: np.ndarray) -> np.ndarray:
        """Convert a grayscale float map to a color heatmap."""
        if array is None:
            return None
        norm = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)
