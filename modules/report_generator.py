"""
Forensic Report Generator
--------------------------
Generates a comprehensive PDF forensic report from analysis results,
including annotated images, charts, findings, and confidence scores.
"""

import os
import io
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, HRFlowable, PageBreak,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


class ForensicReportGenerator:
    """
    Generates a PDF forensic analysis report.
    """

    SEVERITY_COLORS = {
        "HIGH":   colors.HexColor("#DC2626"),
        "MEDIUM": colors.HexColor("#D97706"),
        "LOW":    colors.HexColor("#16A34A"),
        "CLEAN":  colors.HexColor("#2563EB"),
    }

    def generate(self, results: dict, output_path: str) -> str:
        """
        Generate a forensic PDF report.

        Args:
            results: combined output from ForensicPipeline.analyze()
            output_path: where to save the PDF

        Returns:
            Path to generated PDF.
        """
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm,
        )

        styles = self._build_styles()
        story = []

        # --- Cover Header ---
        story += self._build_header(styles, results)
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1E293B")))
        story.append(Spacer(1, 0.4*cm))

        # --- Executive Summary ---
        story += self._build_summary(styles, results)
        story.append(Spacer(1, 0.5*cm))

        # --- Overall Confidence Gauge ---
        gauge_img = self._render_confidence_gauge(results["overall_score"])
        story.append(gauge_img)
        story.append(Spacer(1, 0.5*cm))

        # --- Module-Level Results Table ---
        story += self._build_module_table(styles, results)
        story.append(Spacer(1, 0.5*cm))

        # --- Findings by Module ---
        story += self._build_findings(styles, results)

        # --- Annotated Images ---
        story.append(PageBreak())
        story += self._build_image_section(styles, results)

        # --- Metadata Section ---
        story.append(PageBreak())
        story += self._build_metadata_section(styles, results)

        # --- Footer ---
        story.append(Spacer(1, 1*cm))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
        story.append(Paragraph(
            f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            "Image Forensics System v1.0 | For investigative use only.",
            styles["footer"]
        ))

        doc.build(story)
        return output_path

    def _build_styles(self):
        styles = getSampleStyleSheet()
        custom = {
            "title": ParagraphStyle("title", fontSize=22, fontName="Helvetica-Bold",
                                    textColor=colors.HexColor("#0F172A"), spaceAfter=6),
            "subtitle": ParagraphStyle("subtitle", fontSize=11, fontName="Helvetica",
                                       textColor=colors.HexColor("#475569"), spaceAfter=4),
            "section": ParagraphStyle("section", fontSize=13, fontName="Helvetica-Bold",
                                      textColor=colors.HexColor("#1E293B"), spaceBefore=10, spaceAfter=4),
            "body": ParagraphStyle("body", fontSize=9, fontName="Helvetica",
                                   textColor=colors.HexColor("#334155"), leading=14),
            "flag": ParagraphStyle("flag", fontSize=9, fontName="Helvetica",
                                   textColor=colors.HexColor("#7C3AED"), leftIndent=12, leading=13),
            "footer": ParagraphStyle("footer", fontSize=7, fontName="Helvetica",
                                     textColor=colors.grey, alignment=TA_CENTER),
            "verdict_high": ParagraphStyle("verdict_high", fontSize=14, fontName="Helvetica-Bold",
                                           textColor=colors.HexColor("#DC2626")),
            "verdict_low": ParagraphStyle("verdict_low", fontSize=14, fontName="Helvetica-Bold",
                                          textColor=colors.HexColor("#16A34A")),
        }
        return custom

    def _build_header(self, styles, results):
        image_path = results.get("image_path", "Unknown")
        filename = os.path.basename(image_path)
        return [
            Paragraph("🔍 IMAGE FORENSICS REPORT", styles["title"]),
            Paragraph(f"File: <b>{filename}</b>", styles["subtitle"]),
            Paragraph(
                f"Analysis Date: {datetime.now().strftime('%B %d, %Y')}  |  "
                f"Image Size: {results.get('image_size', 'N/A')}  |  "
                f"Format: {results.get('image_format', 'N/A')}",
                styles["subtitle"]
            ),
            Spacer(1, 0.3*cm),
        ]

    def _build_summary(self, styles, results):
        score = results["overall_score"]
        verdict = results["verdict"]
        manipulation_type = results.get("manipulation_type", "Unknown")

        if score >= 70:
            verdict_style = styles["verdict_high"]
            verdict_text = f"⚠ VERDICT: {verdict} (Confidence: {score:.1f}%)"
        elif score >= 40:
            verdict_style = styles["verdict_high"]
            verdict_text = f"⚠ VERDICT: {verdict} (Confidence: {score:.1f}%)"
        else:
            verdict_style = styles["verdict_low"]
            verdict_text = f"✓ VERDICT: {verdict} (Confidence: {score:.1f}%)"

        elements = [
            Paragraph("Executive Summary", styles["section"]),
            Paragraph(verdict_text, verdict_style),
            Spacer(1, 0.2*cm),
            Paragraph(
                f"<b>Detected Manipulation Type(s):</b> {manipulation_type}",
                styles["body"]
            ),
            Spacer(1, 0.1*cm),
            Paragraph(results.get("summary_text", ""), styles["body"]),
        ]
        return elements

    def _render_confidence_gauge(self, score: float) -> RLImage:
        """Render a horizontal confidence bar chart."""
        fig, ax = plt.subplots(figsize=(7, 1.0))
        fig.patch.set_facecolor("white")

        # Background bar
        ax.barh(0, 100, color="#E2E8F0", height=0.5)

        # Score bar
        color = "#DC2626" if score >= 70 else "#D97706" if score >= 40 else "#16A34A"
        ax.barh(0, score, color=color, height=0.5)

        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel("Manipulation Probability (%)", fontsize=9)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.text(score + 1, 0, f"{score:.1f}%", va="center", fontsize=10,
                fontweight="bold", color=color)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        buf.seek(0)

        return RLImage(buf, width=14*cm, height=2.5*cm)

    def _build_module_table(self, styles, results):
        modules = results.get("module_scores", {})
        data = [["Module", "Score", "Risk Level", "Key Finding"]]

        for name, info in modules.items():
            score = info.get("score", 0)
            risk = "HIGH" if score >= 70 else "MEDIUM" if score >= 40 else "LOW"
            key_finding = info.get("key_finding", "—")[:60]
            data.append([name, f"{score:.1f}%", risk, key_finding])

        table = Table(data, colWidths=[3.5*cm, 2*cm, 2.5*cm, 9*cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F8FAFC"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))

        return [
            Paragraph("Module Analysis Results", styles["section"]),
            table,
        ]

    def _build_findings(self, styles, results):
        elements = [Paragraph("Detailed Findings", styles["section"])]

        for module_name, info in results.get("module_scores", {}).items():
            flags = info.get("flags", [])
            findings_list = info.get("findings", [])
            all_items = flags + findings_list

            if not all_items:
                continue

            elements.append(Paragraph(f"▸ {module_name}", styles["body"]))
            for item in all_items:
                elements.append(Paragraph(f"• {item}", styles["flag"]))
            elements.append(Spacer(1, 0.2*cm))

        return elements

    def _build_image_section(self, styles, results):
        elements = [Paragraph("Visual Analysis", styles["section"])]

        image_outputs = results.get("image_outputs", {})

        for label, img_array in image_outputs.items():
            if img_array is None:
                continue
            try:
                pil_img = self._array_to_pil(img_array)
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                buf.seek(0)
                w = min(14*cm, 14*cm)
                h = w * (pil_img.height / pil_img.width)
                elements.append(Paragraph(label, styles["body"]))
                elements.append(RLImage(buf, width=w, height=h))
                elements.append(Spacer(1, 0.5*cm))
            except Exception as e:
                elements.append(Paragraph(f"[Could not render {label}: {e}]", styles["body"]))

        return elements

    def _build_metadata_section(self, styles, results):
        elements = [Paragraph("Metadata Inventory", styles["section"])]

        metadata = results.get("raw_metadata", {})
        if not metadata:
            elements.append(Paragraph("No EXIF metadata found in image.", styles["body"]))
            return elements

        # Display top 30 metadata fields
        shown = list(metadata.items())[:30]
        data = [["Field", "Value"]]
        for k, v in shown:
            data.append([str(k)[:40], str(v)[:80]])

        table = Table(data, colWidths=[6*cm, 11*cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F8FAFC"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 4),
            ("WORDWRAP", (0, 0), (-1, -1), True),
        ]))

        elements.append(table)
        return elements

    def _array_to_pil(self, arr: np.ndarray):
        from PIL import Image
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        if len(arr.shape) == 2:
            return Image.fromarray(arr, mode="L")
        elif arr.shape[2] == 3:
            # OpenCV BGR → RGB
            return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
        return Image.fromarray(arr)
