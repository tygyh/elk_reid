import argparse
import csv
import json
import logging
import re
from datetime import datetime
from pathlib import Path

from megadetector.detection.run_detector_batch import load_and_run_detector_batch

TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$"
)

CSV_FIELDS = [
    "run_id",
    "segment_path",
    "timestamp",
    "captured_at",
    "is_animal",
    "animal_detections",
    "failure",
    "max_confidence",
]

SUMMARY_CSV_FIELDS = [
    "run_id",
    "total_segments",
    "animal_segments",
    "failed_segments",
    "total_animal_detections",
    "max_confidence",
]

logger = logging.getLogger(__name__)


class MegaDetectorRunner:
    def __init__(self, model_file: str):
        self.model_file = model_file

    def run(self, image_paths: list[str]) -> dict:
        return load_and_run_detector_batch(
            model_file=self.model_file,
            image_file_names=image_paths,
        )


def is_animal_detection(det: dict) -> bool:
    return str(det.get("category")) == "1"


def passes_threshold(det: dict, threshold: float) -> bool:
    return float(det.get("conf", 0.0)) >= threshold


def segment_has_animal(result: dict, threshold: float) -> bool:
    for det in result.get("detections", []):
        if is_animal_detection(det) and passes_threshold(det, threshold):
            return True
    return False


def animal_detections(result: dict, threshold: float) -> list[dict]:
    return [
        {
            "category": str(det.get("category")),
            "confidence": float(det.get("conf", 0.0)),
            "bbox": det.get("bbox"),
        }
        for det in result.get("detections", [])
        if is_animal_detection(det) and passes_threshold(det, threshold)
    ]


def extract_animal_segments(md_results: dict, threshold: float) -> list[dict]:
    results = []

    for image_result in md_results.get("images", []):
        segment_path = image_result["file"]

        try:
            timestamp = extract_timestamp_from_filename(segment_path)
            captured_at = parse_timestamp_from_filename(segment_path)
        except ValueError as exc:
            results.append(
                {
                    "segment_path": segment_path,
                    "timestamp": None,
                    "captured_at": None,
                    "is_animal": False,
                    "animal_detections": [],
                    "failure": str(exc),
                }
            )
            continue

        if "failure" in image_result:
            results.append(
                {
                    "segment_path": segment_path,
                    "timestamp": timestamp,
                    "captured_at": captured_at,
                    "is_animal": False,
                    "animal_detections": [],
                    "failure": image_result["failure"],
                }
            )
            continue

        detections = animal_detections(image_result, threshold)

        results.append(
            {
                "segment_path": segment_path,
                "timestamp": timestamp,
                "captured_at": captured_at,
                "is_animal": len(detections) > 0,
                "animal_detections": detections,
                "failure": None,
            }
        )

    return sorted(
        results,
        key=lambda result: (
            result["captured_at"] is None,
            result["captured_at"] or "",
        ),
    )


def detect_animals_in_segments(
        segment_paths: list[str],
        runner,
        threshold: float = 0.5,
) -> list[dict]:
    md_results = runner.run(segment_paths)
    return extract_animal_segments(md_results, threshold)


def extract_timestamp_from_filename(path: str) -> str:
    timestamp = Path(path).stem

    if not TIMESTAMP_PATTERN.match(timestamp):
        raise ValueError(
            f"Expected filename format YYYY-MM-DD_hh-mm-ss, got: {timestamp}"
        )

    try:
        datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S")
    except ValueError as exc:
        raise ValueError(f"Invalid timestamp filename: {timestamp}") from exc

    return timestamp


def parse_timestamp_from_filename(path: str) -> str:
    return parse_timestamp_datetime(path).isoformat()


def parse_timestamp_datetime(path: str) -> datetime:
    timestamp = extract_timestamp_from_filename(path)
    return datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S")


def summarize_detection_results(results: list[dict]) -> dict:
    confidences = [
        detection["confidence"]
        for result in results
        for detection in result["animal_detections"]
    ]

    return {
        "total_segments": len(results),
        "animal_segments": sum(1 for result in results if result["is_animal"]),
        "failed_segments": sum(1 for result in results if result["failure"] is not None),
        "total_animal_detections": len(confidences),
        "max_confidence": max(confidences) if confidences else None,
    }


def write_detection_results_csv(
        results: list[dict],
        csv_path: str | Path,
        append: bool = False,
        run_id: str | None = None,
) -> None:
    csv_path = Path(csv_path)

    mode = "a" if append else "w"
    should_write_header = not append or not csv_path.exists() or csv_path.stat().st_size == 0

    existing_segment_paths = (
        existing_segment_paths_in_csv(csv_path)
        if append
        else set()
    )

    with csv_path.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)

        if should_write_header:
            writer.writeheader()

        for result in results:
            if result["segment_path"] in existing_segment_paths:
                continue

            max_confidence = max_confidence_for_result(result)

            writer.writerow(
                {
                    "run_id": run_id or "",
                    "segment_path": result["segment_path"],
                    "timestamp": result["timestamp"] or "",
                    "captured_at": result["captured_at"] or "",
                    "is_animal": result["is_animal"],
                    "animal_detections": json.dumps(result["animal_detections"]),
                    "failure": result["failure"] or "",
                    "max_confidence": (
                        max_confidence if max_confidence is not None else ""
                    ),
                }
            )

            existing_segment_paths.add(result["segment_path"])


def max_confidence_for_result(result: dict) -> float | None:
    confidences = [
        detection["confidence"]
        for detection in result["animal_detections"]
    ]

    return max(confidences) if confidences else None


def existing_segment_paths_in_csv(csv_path: str | Path) -> set[str]:
    csv_path = Path(csv_path)

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()

    with csv_path.open(newline="") as f:
        return {
            row["segment_path"]
            for row in csv.DictReader(f)
            if row.get("segment_path")
        }


def find_segment_files(
        segment_dir: str | Path,
        recursive: bool = False,
) -> list[str]:
    segment_dir = Path(segment_dir)

    paths = segment_dir.rglob("*") if recursive else segment_dir.glob("*")

    segment_files = []

    for path in paths:
        if not path.is_file():
            continue

        if path.suffix.lower() not in {".jpg", ".jpeg"}:
            continue

        try:
            parse_timestamp_datetime(str(path))
        except ValueError:
            continue

        segment_files.append(path)

    return [
        str(path)
        for path in sorted(
            segment_files,
            key=lambda path: parse_timestamp_datetime(str(path)),
        )
    ]


def generate_run_id() -> str:
    return datetime.now().strftime("run-%Y-%m-%d_%H-%M-%S")


def run_detection_pipeline(
        segment_dir: str | Path,
        output_csv: str | Path,
        runner,
        threshold: float = 0.5,
        recursive: bool = False,
        append: bool = False,
        run_id: str | None = None,
        summary_csv: str | Path | None = None,
) -> list[dict]:
    segment_paths = find_segment_files(segment_dir, recursive=recursive)
    logger.info("Found %s segment files", len(segment_paths))

    run_id = run_id or generate_run_id()

    if not segment_paths:
        results = []
        summary = summarize_detection_results(results)

        write_detection_results_csv(
            results=results,
            csv_path=output_csv,
            append=append,
            run_id=run_id,
        )
        logger.info("Wrote detections CSV: %s", output_csv)

        if summary_csv is not None:
            write_detection_summary_csv(
                summary=summary,
                csv_path=summary_csv,
                run_id=run_id,
                append=append,
            )
            logger.info("Wrote summary CSV: %s", summary_csv)

        logger.info("Animal segments: %s", summary["animal_segments"])
        logger.info("Animal detections: %s", summary["total_animal_detections"])
        logger.info("Failed segments: %s", summary["failed_segments"])

        return results

    md_results = runner.run(segment_paths)

    results = extract_animal_segments(
        md_results=md_results,
        threshold=threshold,
    )

    summary = summarize_detection_results(results)

    write_detection_results_csv(
        results=results,
        csv_path=output_csv,
        append=append,
        run_id=run_id,
    )
    logger.info("Wrote detections CSV: %s", output_csv)

    if summary_csv is not None:
        write_detection_summary_csv(
            summary=summary,
            csv_path=summary_csv,
            run_id=run_id,
            append=append,
        )
        logger.info("Wrote summary CSV: %s", summary_csv)

    logger.info("Animal segments: %s", summary["animal_segments"])
    logger.info("Animal detections: %s", summary["total_animal_detections"])
    logger.info("Failed segments: %s", summary["failed_segments"])

    return results


def write_detection_summary_csv(
        summary: dict,
        csv_path: str | Path,
        run_id: str,
        append: bool = False,
) -> None:
    csv_path = Path(csv_path)

    mode = "a" if append else "w"
    should_write_header = (
            not append or not csv_path.exists() or csv_path.stat().st_size == 0
    )

    existing_run_ids = (
        existing_run_ids_in_summary_csv(csv_path)
        if append
        else set()
    )

    if run_id in existing_run_ids:
        return

    with csv_path.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_CSV_FIELDS)

        if should_write_header:
            writer.writeheader()

        writer.writerow(
            {
                "run_id": run_id,
                "total_segments": summary["total_segments"],
                "animal_segments": summary["animal_segments"],
                "failed_segments": summary["failed_segments"],
                "total_animal_detections": summary["total_animal_detections"],
                "max_confidence": (
                    summary["max_confidence"]
                    if summary["max_confidence"] is not None
                    else ""
                ),
            }
        )


def existing_run_ids_in_summary_csv(csv_path: str | Path) -> set[str]:
    csv_path = Path(csv_path)

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()

    with csv_path.open(newline="") as f:
        return {
            row["run_id"]
            for row in csv.DictReader(f)
            if row.get("run_id")
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect animals in SAM-segmented images using MegaDetector."
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level.",
    )
    parser.add_argument(
        "--segments",
        required=True,
        type=Path,
        help="Directory containing timestamped segment images.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to detections CSV.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional path to summary CSV.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="MegaDetector model file or model name.",
    )
    parser.add_argument(
        "--threshold",
        type=confidence_threshold,
        default=0.5,
        help="Animal confidence threshold between 0.0 and 1.0.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for segment images recursively.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing CSV files.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run identifier.",
    )

    return parser


def confidence_threshold(value: str) -> float:
    threshold = float(value)

    if threshold < 0.0 or threshold > 1.0:
        raise argparse.ArgumentTypeError(
            "threshold must be between 0.0 and 1.0"
        )

    return threshold


def main(argv: list[str] | None = None) -> list[dict]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s:%(name)s:%(message)s",
    )

    runner = MegaDetectorRunner(model_file=args.model)

    return run_detection_pipeline(
        segment_dir=args.segments,
        output_csv=args.output,
        runner=runner,
        threshold=args.threshold,
        recursive=args.recursive,
        append=args.append,
        run_id=args.run_id,
        summary_csv=args.summary,
    )
