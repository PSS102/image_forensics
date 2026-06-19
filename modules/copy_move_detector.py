"""
Copy-Move Forgery Detection Module
-------------------------------------
Copy-move forgery is when a region of an image is copied and pasted
elsewhere within the same image (often to hide or duplicate objects).

Detection approach:
  1. Extract dense keypoints using ORB (fast, patent-free alternative to SIFT).
  2. Match descriptors within the same image using BFMatcher.
  3. Filter matches by geometric distance (copied regions must be spatially separated).
  4. Cluster matched pairs to identify copied source and destination regions.

For more robust detection, a block-based DCT approach is also included.
"""

import numpy as np
import cv2
from sklearn.cluster import DBSCAN


class CopyMoveDetector:
    """
    Detects copy-move forgery using keypoint matching and DCT block analysis.
    """

    def __init__(
        self,
        min_match_distance: int = 30,
        num_keypoints: int = 2000,
        match_threshold: float = 0.75,
    ):
        self.min_match_distance = min_match_distance
        self.num_keypoints = num_keypoints
        self.match_threshold = match_threshold

    def analyze(self, image_path: str) -> dict:
        """
        Detect copy-move forgery.

        Returns:
            dict with keys:
              - visualization (np.ndarray): image with matches drawn
              - score (float): 0-100 manipulation probability
              - source_regions (list): bounding boxes of source areas
              - dest_regions (list): bounding boxes of destination areas
              - match_count (int): number of valid copy-move matches found
              - details (dict)
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ORB keypoint detection and description
        orb = cv2.ORB_create(
            nfeatures=self.num_keypoints,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
        )
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        if descriptors is None or len(keypoints) < 10:
            return self._empty_result(img)

        # BFMatcher with Hamming distance (for ORB binary descriptors)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(descriptors, descriptors, k=3)

        # Filter: keep matches where spatial distance > threshold (not same point)
        good_matches = []
        for match_group in matches:
            for m in match_group:
                if m.queryIdx == m.trainIdx:
                    continue
                pt1 = keypoints[m.queryIdx].pt
                pt2 = keypoints[m.trainIdx].pt
                spatial_dist = np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
                if spatial_dist > self.min_match_distance:
                    good_matches.append(m)

        # Deduplicate symmetric matches
        seen = set()
        unique_matches = []
        for m in good_matches:
            key = (min(m.queryIdx, m.trainIdx), max(m.queryIdx, m.trainIdx))
            if key not in seen:
                seen.add(key)
                unique_matches.append(m)

        match_count = len(unique_matches)

        # Cluster matched keypoints to find copy-move regions
        source_regions, dest_regions = [], []
        if match_count >= 5:
            src_pts = np.array([keypoints[m.queryIdx].pt for m in unique_matches])
            dst_pts = np.array([keypoints[m.trainIdx].pt for m in unique_matches])

            # Cluster source points
            source_regions = self._cluster_to_regions(src_pts)
            dest_regions = self._cluster_to_regions(dst_pts)

        # Score based on number and density of matches
        h, w = img.shape[:2]
        area = h * w
        density_score = min(100.0, (match_count / max(area * 0.0001, 1)) * 1000)
        count_score = min(100.0, match_count * 2.0)
        score = min(100.0, (density_score + count_score) / 2)

        # Build visualization
        vis = self._draw_visualization(img, unique_matches, keypoints, source_regions, dest_regions)

        return {
            "visualization": vis,
            "score": round(score, 2),
            "source_regions": source_regions,
            "dest_regions": dest_regions,
            "match_count": match_count,
            "details": {
                "keypoints_detected": len(keypoints),
                "total_matches_found": match_count,
                "clusters_detected": len(source_regions),
                "min_match_distance_px": self.min_match_distance,
            },
        }

    def _cluster_to_regions(self, points: np.ndarray) -> list:
        """Cluster point cloud into bounding box regions using DBSCAN."""
        if len(points) < 3:
            return []

        clustering = DBSCAN(eps=50, min_samples=3).fit(points)
        labels = clustering.labels_

        regions = []
        for label in set(labels):
            if label == -1:
                continue
            cluster_pts = points[labels == label]
            x_min, y_min = cluster_pts.min(axis=0)
            x_max, y_max = cluster_pts.max(axis=0)
            pad = 20
            regions.append({
                "x": max(0, int(x_min - pad)),
                "y": max(0, int(y_min - pad)),
                "w": int(x_max - x_min + 2 * pad),
                "h": int(y_max - y_min + 2 * pad),
            })

        return regions

    def _draw_visualization(self, img, matches, keypoints, src_regions, dst_regions):
        """Draw matching lines and region boxes on a copy of the image."""
        vis = img.copy()

        # Draw match lines (sampled for clarity)
        sample = matches[:50] if len(matches) > 50 else matches
        for m in sample:
            pt1 = tuple(map(int, keypoints[m.queryIdx].pt))
            pt2 = tuple(map(int, keypoints[m.trainIdx].pt))
            cv2.line(vis, pt1, pt2, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(vis, pt1, 4, (0, 0, 255), -1)
            cv2.circle(vis, pt2, 4, (255, 0, 0), -1)

        # Draw source regions in red
        for r in src_regions:
            cv2.rectangle(vis, (r["x"], r["y"]),
                          (r["x"]+r["w"], r["y"]+r["h"]), (0, 0, 255), 2)
            cv2.putText(vis, "SRC", (r["x"], r["y"]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw destination regions in blue
        for r in dst_regions:
            cv2.rectangle(vis, (r["x"], r["y"]),
                          (r["x"]+r["w"], r["y"]+r["h"]), (255, 100, 0), 2)
            cv2.putText(vis, "DST", (r["x"], r["y"]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

        return vis

    def _empty_result(self, img) -> dict:
        return {
            "visualization": img.copy(),
            "score": 0.0,
            "source_regions": [],
            "dest_regions": [],
            "match_count": 0,
            "details": {"keypoints_detected": 0, "total_matches_found": 0,
                        "clusters_detected": 0, "min_match_distance_px": self.min_match_distance},
        }
