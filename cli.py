"""
Command-Line Interface
-----------------------
Run forensic analysis directly from the terminal without the API server.

Usage:
    python cli.py --image path/to/image.jpg
    python cli.py --image path/to/image.jpg --output my_output_dir --no-report
    python cli.py --image path/to/image.jpg --json-only
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import ForensicPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Image Forensics & Tampering Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py --image photo.jpg
  python cli.py --image photo.jpg --output ./results
  python cli.py --image photo.jpg --json-only --no-report
        """,
    )
    parser.add_argument("--image", required=True, help="Path to the image to analyze")
    parser.add_argument("--output", default="output", help="Output directory (default: output/)")
    parser.add_argument(
        "--no-report", action="store_true", help="Skip PDF report generation"
    )
    parser.add_argument(
        "--json-only", action="store_true",
        help="Print JSON results to stdout only (no PDF, no verbose output)"
    )
    return parser.parse_args()


def print_banner():
    print("\n" + "="*60)
    print("  ██████╗  ██████╗ ██████╗ ███████╗███╗   ██╗███████╗")
    print("  ██╔════╝ ██╔═══██╗██╔══██╗██╔════╝████╗  ██║██╔════╝")
    print("  ██║█████╗██║   ██║██████╔╝█████╗  ██╔██╗ ██║███████╗")
    print("  ██║╚════╝██║   ██║██╔══██╗██╔══╝  ██║╚██╗██║╚════██║")
    print("  ╚██████╗ ╚██████╔╝██║  ██║███████╗██║ ╚████║███████║")
    print("   ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝╚══════╝")
    print("  IMAGE FORENSICS & TAMPERING DETECTION SYSTEM v1.0")
    print("="*60 + "\n")


def main():
    args = parse_args()

    if not args.json_only:
        print_banner()

    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    pipeline = ForensicPipeline(output_dir=args.output)

    try:
        results = pipeline.analyze(
            image_path=args.image,
            generate_report=not args.no_report and not args.json_only,
        )
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json_only:
        # Print clean JSON (exclude numpy arrays from image_outputs)
        clean_results = {
            k: v for k, v in results.items()
            if k not in ("image_outputs", "module_results", "raw_metadata")
        }
        print(json.dumps(clean_results, indent=2, default=str))
        return

    # --- Pretty print summary ---
    print("\n" + "="*60)
    print("  FORENSIC ANALYSIS COMPLETE")
    print("="*60)
    print(f"  File         : {os.path.basename(args.image)}")
    print(f"  Size         : {results['image_size']}")
    print(f"  Format       : {results['image_format']}")
    print(f"  Overall Score: {results['overall_score']:.1f}%")
    print(f"  Verdict      : {results['verdict']}")
    print(f"  Manipulation : {results['manipulation_type']}")
    print()
    print("  Module Scores:")
    for name, info in results["module_scores"].items():
        bar = _score_bar(info["score"])
        print(f"    {name:<28} {bar} {info['score']:5.1f}%")

    if results.get("report_path"):
        print(f"\n  PDF Report   : {results['report_path']}")

    print("="*60 + "\n")


def _score_bar(score: float, width: int = 20) -> str:
    filled = int(score / 100 * width)
    empty = width - filled
    if score >= 70:
        char = "█"
    elif score >= 40:
        char = "▓"
    else:
        char = "░"
    return f"[{char * filled}{'·' * empty}]"


if __name__ == "__main__":
    main()
