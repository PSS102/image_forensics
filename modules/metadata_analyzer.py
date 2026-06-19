"""
Metadata Analysis Module
--------------------------
Examines EXIF and file-level metadata for signs of manipulation.

Red flags include:
  - Missing EXIF (stripped by editing software)
  - Software tag revealing editing tools (Photoshop, GIMP, etc.)
  - Timestamp anomalies (modified > created, GPS mismatch)
  - Thumbnail mismatch with main image
  - JPEG comment fields with unusual content
  - Make/Model absent or inconsistent with other fields
"""

import os
import struct
import piexif
import exifread
from PIL import Image
from datetime import datetime


# Known editing software signatures
EDITING_SOFTWARE = [
    "adobe photoshop", "photoshop", "gimp", "lightroom",
    "affinity photo", "capture one", "darktable", "rawtherapee",
    "paint.net", "snapseed", "facetune", "meitu", "picsart",
    "canva", "pixlr", "fotor", "stable diffusion", "midjourney",
    "dall-e", "firefly", "imagemagick", "opencv", "pillow",
]


class MetadataAnalyzer:
    """
    Parses image metadata and flags forensic indicators.
    """

    def analyze(self, image_path: str) -> dict:
        """
        Analyze image metadata for tampering indicators.

        Returns:
            dict with keys:
              - score (float): 0-100 manipulation probability
              - flags (list): list of red-flag strings
              - metadata (dict): extracted raw metadata
              - details (dict): structured field analysis
        """
        flags = []
        metadata = {}
        details = {}

        # --- Basic file info ---
        file_stat = os.stat(image_path)
        file_size = file_stat.st_size
        mod_time = datetime.fromtimestamp(file_stat.st_mtime).isoformat()

        # --- EXIF via exifread ---
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f, details=False)

        # Convert tags to serializable dict
        for k, v in tags.items():
            try:
                metadata[k] = str(v)
            except Exception:
                metadata[k] = "[unreadable]"

        # --- EXIF via piexif for structured access ---
        try:
            exif_dict = piexif.load(image_path)
            has_exif = bool(exif_dict.get("Exif") or exif_dict.get("0th"))
        except Exception:
            exif_dict = {}
            has_exif = False

        # --- Flag: Missing EXIF ---
        if not has_exif or not tags:
            flags.append("EXIF data is absent or stripped — common after editing")
            details["exif_present"] = False
        else:
            details["exif_present"] = True

        # --- Flag: Editing software in tags ---
        software_tag = metadata.get("Image Software", "").lower()
        processing_tag = metadata.get("Image ProcessingSoftware", "").lower()
        combined_software = software_tag + " " + processing_tag

        detected_software = []
        for sw in EDITING_SOFTWARE:
            if sw in combined_software:
                detected_software.append(sw.title())

        if detected_software:
            flags.append(f"Editing software detected: {', '.join(detected_software)}")
        details["software_tags"] = {
            "Software": metadata.get("Image Software", "N/A"),
            "ProcessingSoftware": metadata.get("Image ProcessingSoftware", "N/A"),
        }

        # --- Camera make/model ---
        make = metadata.get("Image Make", "")
        model = metadata.get("Image Model", "")
        if not make and not model:
            flags.append("Camera Make/Model absent — may indicate synthetic or edited origin")
        details["camera"] = {"make": make or "N/A", "model": model or "N/A"}

        # --- Timestamp analysis ---
        dt_original = metadata.get("EXIF DateTimeOriginal", "")
        dt_digitized = metadata.get("EXIF DateTimeDigitized", "")
        dt_modified = metadata.get("Image DateTime", "")

        details["timestamps"] = {
            "DateTimeOriginal": dt_original or "N/A",
            "DateTimeDigitized": dt_digitized or "N/A",
            "DateTime (modified)": dt_modified or "N/A",
            "File Modified": mod_time,
        }

        if dt_original and dt_modified and dt_original != dt_modified:
            flags.append(
                f"Timestamp mismatch: original={dt_original}, modified={dt_modified}"
            )

        # --- Thumbnail consistency check ---
        thumbnail_flag = self._check_thumbnail(image_path, exif_dict)
        if thumbnail_flag:
            flags.append(thumbnail_flag)

        # --- GPS data ---
        gps_lat = metadata.get("GPS GPSLatitude", None)
        gps_lon = metadata.get("GPS GPSLongitude", None)
        details["gps"] = {
            "latitude": str(gps_lat) if gps_lat else "N/A",
            "longitude": str(gps_lon) if gps_lon else "N/A",
        }

        # --- JPEG Comment field ---
        comment = metadata.get("Image ImageDescription", "") or metadata.get("EXIF UserComment", "")
        if comment:
            details["comment"] = str(comment)
            # Flag if comment mentions AI tools
            for sw in EDITING_SOFTWARE:
                if sw in str(comment).lower():
                    flags.append(f"AI/editing tool referenced in comment field: {comment[:80]}")
                    break
        else:
            details["comment"] = "N/A"

        # --- Compute score ---
        # Each flag contributes weighted points
        flag_weights = {
            "EXIF data is absent": 30,
            "Editing software detected": 50,
            "Camera Make/Model absent": 25,
            "Timestamp mismatch": 35,
            "Thumbnail mismatch": 45,
            "AI/editing tool referenced": 55,
        }

        score = 0.0
        for flag in flags:
            for key, weight in flag_weights.items():
                if key.lower() in flag.lower():
                    score += weight
                    break
            else:
                score += 10  # Generic flag weight

        score = min(100.0, score)

        return {
            "score": round(score, 2),
            "flags": flags,
            "metadata": metadata,
            "details": details,
        }

    def _check_thumbnail(self, image_path: str, exif_dict: dict) -> str:
        """
        Compare embedded EXIF thumbnail dimensions vs main image dimensions.
        A significant mismatch may indicate the main image was replaced.
        """
        try:
            main_img = Image.open(image_path)
            main_size = main_img.size  # (width, height)

            thumbnail_data = exif_dict.get("thumbnail")
            if not thumbnail_data:
                return None

            import io
            thumb = Image.open(io.BytesIO(thumbnail_data))
            thumb_size = thumb.size

            # Check aspect ratio consistency
            main_ratio = main_size[0] / max(main_size[1], 1)
            thumb_ratio = thumb_size[0] / max(thumb_size[1], 1)

            ratio_diff = abs(main_ratio - thumb_ratio)
            if ratio_diff > 0.15:
                return (
                    f"Thumbnail aspect ratio ({thumb_ratio:.2f}) mismatches "
                    f"main image ({main_ratio:.2f}) — image may have been replaced"
                )
        except Exception:
            pass

        return None
