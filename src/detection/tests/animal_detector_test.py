from unittest.mock import patch, Mock

import pytest

from detection.animal_detector import *


@pytest.fixture
def image_paths():
    return [
        "segments/2026-05-20_14-30-05.jpg",
        "segments/2026-05-20_14-31-10.jpg",
        "segments/2026-05-20_14-33-17.jpg"
    ]


def animal_detection(confidence, category="1", bbox=None):
    return {
        "category": category,
        "confidence": confidence,
        "bbox": bbox,
    }


def md_detection(confidence, category="1", bbox=None):
    data = {
        "category": category,
        "conf": confidence,
    }

    if bbox is not None:
        data["bbox"] = bbox

    return data


def expected_segment_result(
        segment_path,
        *,
        is_animal=False,
        animal_detections=None,
        failure=None,
):
    timestamp = Path(segment_path).stem
    captured_at = datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S").isoformat()

    return {
        "segment_path": segment_path,
        "timestamp": timestamp,
        "captured_at": captured_at,
        "is_animal": is_animal,
        "animal_detections": animal_detections or [],
        "failure": failure,
    }


def expected_invalid_segment_result(segment_path, failure):
    return {
        "segment_path": segment_path,
        "timestamp": None,
        "captured_at": None,
        "is_animal": False,
        "animal_detections": [],
        "failure": failure,
    }


def expected_csv_row(
        segment_path,
        *,
        run_id="",
        is_animal=False,
        animal_detections=None,
        failure="",
        max_confidence="",
):
    timestamp = Path(segment_path).stem
    captured_at = datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S").isoformat()

    return {
        "run_id": run_id,
        "segment_path": segment_path,
        "timestamp": timestamp,
        "captured_at": captured_at,
        "is_animal": str(is_animal),
        "animal_detections": json.dumps(animal_detections or []),
        "failure": failure,
        "max_confidence": max_confidence,
    }


def expected_summary_row(
        run_id,
        *,
        total_segments,
        animal_segments,
        failed_segments,
        total_animal_detections,
        max_confidence="",
):
    return {
        "run_id": run_id,
        "total_segments": str(total_segments),
        "animal_segments": str(animal_segments),
        "failed_segments": str(failed_segments),
        "total_animal_detections": str(total_animal_detections),
        "max_confidence": str(max_confidence) if max_confidence != "" else "",
    }


def read_csv_rows(csv_path):
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


class FakeMegaDetectorRunner:
    def run(self, image_paths):
        return {
            "images": [
                {
                    "file": image_paths[0],
                    "detections": [
                        md_detection(confidence=0.9, category="2", bbox=[0.0, 0.0, 0.2, 0.2])
                    ],
                },
                {
                    "file": image_paths[1],
                    "detections": [
                        md_detection(confidence=0.92, bbox=[0.1, 0.2, 0.3, 0.4])
                    ],
                },
            ]
        }


def test_animal_category_is_animal():
    det = md_detection(confidence=0.8)
    assert is_animal_detection(det) is True


def test_person_category_is_not_animal():
    det = md_detection(confidence=0.8, category="2")
    assert is_animal_detection(det) is False


def test_detection_passes_threshold():
    assert passes_threshold(md_detection(confidence=0.7), 0.5) is True


def test_detection_fails_threshold():
    assert passes_threshold(md_detection(confidence=0.3), 0.5) is False


def test_segment_has_animal_when_animal_above_threshold(image_paths):
    result = {
        "file": image_paths[0],
        "detections": [md_detection(confidence=0.82)],
    }

    assert segment_has_animal(result, threshold=0.5) is True


def test_segment_has_no_animal_when_only_person(image_paths):
    result = {
        "file": image_paths[1],
        "detections": [md_detection(confidence=0.95, category="2")],
    }

    assert segment_has_animal(result, threshold=0.5) is False


def test_segment_has_no_animal_when_animal_below_threshold(image_paths):
    result = {
        "file": image_paths[1],
        "detections": [md_detection(confidence=0.2)],
    }

    assert segment_has_animal(result, threshold=0.5) is False


def test_extracts_segments_with_animal_status(image_paths):
    md_results = {
        "images": [
            {
                "file": image_paths[0],
                "detections": [md_detection(confidence=0.9, category="3")]
            },
            {
                "file": image_paths[1],
                "detections": [md_detection(confidence=0.88)],
            },
            {
                "file": image_paths[2],
                "detections": [md_detection(confidence=0.1)],
            },
        ]
    }

    animals = extract_animal_segments(md_results, threshold=0.5)

    assert animals == [
        expected_segment_result("segments/2026-05-20_14-30-05.jpg"),
        expected_segment_result(
            "segments/2026-05-20_14-31-10.jpg",
            is_animal=True,
            animal_detections=[animal_detection(confidence=0.88)],
        ),
        expected_segment_result("segments/2026-05-20_14-33-17.jpg"),
    ]


def test_detect_animals_in_segments(image_paths):
    runner = FakeMegaDetectorRunner()

    result = detect_animals_in_segments(
        segment_paths=[
            image_paths[0],
            image_paths[1],
        ],
        runner=runner,
        threshold=0.5,
    )

    assert result == [
        expected_segment_result("segments/2026-05-20_14-30-05.jpg"),
        expected_segment_result(
            "segments/2026-05-20_14-31-10.jpg",
            is_animal=True,
            animal_detections=[animal_detection(confidence=0.92, bbox=[0.1, 0.2, 0.3, 0.4])],
        ),
    ]


def test_megadetector_runner_calls_batch_detector(image_paths):
    fake_md_results = {
        "images": [
            {
                "file": image_paths[0],
                "detections": [],
            },
            {
                "file": image_paths[1],
                "detections": [md_detection(confidence=0.91)],
            },
        ]
    }

    with patch("detection.animal_detector.load_and_run_detector_batch") as mock_run:
        mock_run.return_value = fake_md_results

        runner = MegaDetectorRunner(model_file="MDV5A")
        result = runner.run(image_paths)

    mock_run.assert_called_once_with(
        model_file="MDV5A",
        image_file_names=image_paths,
    )

    assert result == fake_md_results


def test_extract_segments_preserves_failed_images(image_paths):
    md_results = {
        "images": [
            {
                "file": image_paths[0],
                "failure": "could not read image",
            },
            {
                "file": image_paths[1],
                "detections": [md_detection(confidence=0.91)],
            },
        ]
    }

    results = extract_animal_segments(md_results, threshold=0.5)

    assert results == [
        expected_segment_result("segments/2026-05-20_14-30-05.jpg", failure="could not read image"),
        expected_segment_result(
            segment_path="segments/2026-05-20_14-31-10.jpg",
            is_animal=True,
            animal_detections=[animal_detection(confidence=0.91)],
        ),
    ]


def test_extract_segments_returns_all_animal_detections_per_segment(image_paths):
    md_results = {
        "images": [
            {
                "file": image_paths[0],
                "detections": [
                    md_detection(confidence=0.91, bbox=[0.1, 0.2, 0.3, 0.4]),
                    md_detection(confidence=0.99, category="2", bbox=[0.0, 0.0, 0.1, 0.1]),
                    md_detection(confidence=0.72, bbox=[0.5, 0.5, 0.2, 0.2]),
                    md_detection(confidence=0.2, bbox=[0.8, 0.8, 0.1, 0.1]),
                ],
            }
        ]
    }

    results = extract_animal_segments(md_results, threshold=0.5)

    assert results == [
        expected_segment_result(
            "segments/2026-05-20_14-30-05.jpg",
            is_animal=True,
            animal_detections=[
                animal_detection(confidence=0.91, bbox=[0.1, 0.2, 0.3, 0.4]),
                animal_detection(confidence=0.72, bbox=[0.5, 0.5, 0.2, 0.2]),
            ],
        )
    ]


def test_animal_detections_preserves_category_and_bbox():
    result = {
        "file": "segments/2026-05-20_14-30-05.jpg",
        "detections": [
            md_detection(
                category="1",
                confidence=0.91,
                bbox=[0.1, 0.2, 0.3, 0.4],
            )
        ],
    }

    detections = animal_detections(result, threshold=0.5)

    assert detections == [
        animal_detection(
            confidence=0.91,
            bbox=[0.1, 0.2, 0.3, 0.4],
        )
    ]


def test_extract_timestamp_accepts_valid_timestamp_filename():
    assert extract_timestamp_from_filename(
        "segments/2026-05-20_14-30-05.jpg"
    ) == "2026-05-20_14-30-05"


def test_extract_timestamp_rejects_invalid_filename():
    with pytest.raises(ValueError):
        extract_timestamp_from_filename("segments/img__seg_001.png")


def test_extract_timestamp_rejects_invalid_datetime():
    with pytest.raises(ValueError):
        extract_timestamp_from_filename("segments/2026-99-99_88-99-99.jpg")


def test_parse_timestamp_from_filename_returns_iso_datetime():
    assert parse_timestamp_from_filename("segments/2026-05-20_14-30-05.jpg") == "2026-05-20T14:30:05"


def test_parse_timestamp_datetime_returns_datetime():
    assert parse_timestamp_datetime(
        "segments/2026-05-20_14-30-05.jpg"
    ) == datetime(2026, 5, 20, 14, 30, 5)


def test_extract_segments_preserves_invalid_timestamp_as_failure():
    md_results = {
        "images": [
            {
                "file": "segments/not-a-timestamp.jpg",
                "detections": [md_detection(confidence=0.91, bbox=[0.1, 0.2, 0.3, 0.4])],
            }
        ]
    }

    results = extract_animal_segments(md_results, threshold=0.5)

    assert results == [
        expected_invalid_segment_result(
            "segments/not-a-timestamp.jpg",
            failure="Expected filename format YYYY-MM-DD_hh-mm-ss, got: not-a-timestamp",
        )
    ]


def test_extract_segments_sorts_results_by_captured_at():
    md_results = {
        "images": [
            {
                "file": "segments/2026-05-20_14-32-00.jpg",
                "detections": [],
            },
            {
                "file": "segments/2026-05-20_14-30-00.jpg",
                "detections": [],
            },
            {
                "file": "segments/2026-05-20_14-31-00.jpg",
                "detections": [],
            },
        ]
    }

    results = extract_animal_segments(md_results, threshold=0.5)

    assert [result["captured_at"] for result in results] == [
        "2026-05-20T14:30:00",
        "2026-05-20T14:31:00",
        "2026-05-20T14:32:00",
    ]


def test_extract_segments_sorts_invalid_timestamps_last():
    md_results = {
        "images": [
            {
                "file": "segments/not-a-timestamp.jpg",
                "detections": [],
            },
            {
                "file": "segments/2026-05-20_14-30-00.jpg",
                "detections": [],
            },
        ]
    }

    results = extract_animal_segments(md_results, threshold=0.5)

    assert [result["segment_path"] for result in results] == [
        "segments/2026-05-20_14-30-00.jpg",
        "segments/not-a-timestamp.jpg",
    ]


def test_summarize_detection_results():
    results = [
        expected_segment_result("segments/2026-05-20_14-31-00.jpg"),
        expected_segment_result(
            "segments/2026-05-20_14-30-00.jpg",
            is_animal=True,
            animal_detections=[
                animal_detection(confidence=0.91, bbox=[0.1, 0.2, 0.3, 0.4]),
                animal_detection(confidence=0.72, bbox=[0.5, 0.5, 0.2, 0.2]),
            ],
        )
    ]

    summary = summarize_detection_results(results)

    assert summary == {
        "total_segments": 2,
        "animal_segments": 1,
        "failed_segments": 0,
        "total_animal_detections": 2,
        "max_confidence": 0.91,
    }


def test_write_detection_results_csv(tmp_path):
    results = [
        expected_segment_result(
            "segments/2026-05-20_14-30-00.jpg",
            is_animal=True,
            animal_detections=[animal_detection(confidence=0.91, bbox=[0.1, 0.2, 0.3, 0.4])],
        ),
        expected_segment_result("segments/2026-05-20_14-31-00.jpg"),
    ]

    csv_path = tmp_path / "detections.csv"

    write_detection_results_csv(results, csv_path)

    rows = read_csv_rows(csv_path)

    assert rows == [
        expected_csv_row(
            "segments/2026-05-20_14-30-00.jpg",
            is_animal=True,
            animal_detections=[animal_detection(confidence=0.91, bbox=[0.1, 0.2, 0.3, 0.4])],
            failure="",
            max_confidence="0.91",
        ),
        expected_csv_row("segments/2026-05-20_14-31-00.jpg"),
    ]


def test_summarize_detection_results_keeps_max_confidence():
    results = [
        expected_segment_result(
            "segments/2026-05-20_14-30-00.jpg",
            is_animal=True,
            animal_detections=[
                animal_detection(confidence=0.72, bbox=[0.1, 0.2, 0.3, 0.4]),
                animal_detection(confidence=0.91, bbox=[0.5, 0.5, 0.2, 0.2]),
            ],
        )
    ]

    summary = summarize_detection_results(results)

    assert summary == {
        "total_segments": 1,
        "animal_segments": 1,
        "failed_segments": 0,
        "total_animal_detections": 2,
        "max_confidence": 0.91,
    }


def test_write_detection_results_csv_appends_without_rewriting_header(tmp_path):
    csv_path = tmp_path / "detections.csv"

    first_results = [
        expected_segment_result(
            segment_path="segments/2026-05-20_14-30-00.jpg",
            is_animal=True,
            animal_detections=[animal_detection(confidence=0.91, bbox=[0.1, 0.2, 0.3, 0.4])],
        )
    ]

    second_results = [expected_segment_result(segment_path="segments/2026-05-20_14-31-00.jpg")]

    write_detection_results_csv(first_results, csv_path)
    write_detection_results_csv(second_results, csv_path, append=True)

    rows = read_csv_rows(csv_path)

    assert rows == [
        expected_csv_row(
            "segments/2026-05-20_14-30-00.jpg",
            is_animal=True,
            animal_detections=[
                animal_detection(
                    confidence=0.91,
                    bbox=[0.1, 0.2, 0.3, 0.4],
                )
            ],
            max_confidence="0.91",
        ),
        expected_csv_row(
            "segments/2026-05-20_14-31-00.jpg",
        ),
    ]


def test_write_detection_results_csv_skips_duplicate_segment_paths_when_appending(tmp_path):
    csv_path = tmp_path / "detections.csv"

    first_results = [
        expected_segment_result(
            segment_path="segments/2026-05-20_14-30-00.jpg",
            is_animal=True,
            animal_detections=[animal_detection(confidence=0.91, bbox=[0.1, 0.2, 0.3, 0.4])],
        )
    ]

    duplicate_results = [
        expected_segment_result(
            segment_path="segments/2026-05-20_14-30-00.jpg",
            is_animal=True,
            animal_detections=[animal_detection(confidence=0.95, bbox=[0.2, 0.2, 0.3, 0.3])],
        )
    ]

    write_detection_results_csv(first_results, csv_path)
    write_detection_results_csv(duplicate_results, csv_path, append=True)

    rows = read_csv_rows(csv_path)

    assert len(rows) == 1
    assert rows[0]["max_confidence"] == "0.91"


def test_find_segment_files_returns_sorted_timestamped_jpgs(tmp_path):
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()

    valid_later = segments_dir / "2026-05-20_14-31-00.jpg"
    valid_earlier = segments_dir / "2026-05-20_14-30-00.jpg"
    invalid_name = segments_dir / "not-a-timestamp.jpg"
    wrong_extension = segments_dir / "2026-05-20_14-32-00.png"

    valid_later.write_text("")
    valid_earlier.write_text("")
    invalid_name.write_text("")
    wrong_extension.write_text("")

    segment_files = find_segment_files(segments_dir)

    assert segment_files == [
        str(valid_earlier),
        str(valid_later),
    ]


def test_find_segment_files_recursively_returns_sorted_timestamped_jpgs(tmp_path):
    segments_dir = tmp_path / "segments"
    nested_dir = segments_dir / "camera-01"
    nested_dir.mkdir(parents=True)

    root_file = segments_dir / "2026-05-20_14-31-00.jpg"
    nested_file = nested_dir / "2026-05-20_14-30-00.jpg"
    invalid_file = nested_dir / "not-a-timestamp.jpg"

    root_file.write_text("")
    nested_file.write_text("")
    invalid_file.write_text("")

    segment_files = find_segment_files(segments_dir, recursive=True)

    assert segment_files == [
        str(nested_file),
        str(root_file),
    ]


def test_find_segment_files_supports_jpeg_extension(tmp_path):
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()

    jpg_file = segments_dir / "2026-05-20_14-30-00.jpg"
    jpeg_file = segments_dir / "2026-05-20_14-31-00.jpeg"

    jpg_file.write_text("")
    jpeg_file.write_text("")

    segment_files = find_segment_files(segments_dir)

    assert segment_files == [
        str(jpg_file),
        str(jpeg_file),
    ]


def test_find_segment_files_supports_uppercase_extensions(tmp_path):
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()

    jpg_file = segments_dir / "2026-05-20_14-30-00.JPG"
    jpeg_file = segments_dir / "2026-05-20_14-31-00.JPEG"

    jpg_file.write_text("")
    jpeg_file.write_text("")

    segment_files = find_segment_files(segments_dir)

    assert segment_files == [
        str(jpg_file),
        str(jpeg_file),
    ]


def test_run_detection_pipeline_finds_detects_writes_and_returns_results(tmp_path):
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()

    first_file = segments_dir / "2026-05-20_14-30-00.jpg"
    second_file = segments_dir / "2026-05-20_14-31-00.jpg"

    first_file.write_text("")
    second_file.write_text("")

    csv_path = tmp_path / "detections.csv"

    class FakeRunner:
        def run(self, image_paths):
            assert image_paths == [
                str(first_file),
                str(second_file),
            ]

            return {
                "images": [
                    {
                        "file": str(first_file),
                        "detections": [md_detection(confidence=0.91, bbox=[0.1, 0.2, 0.3, 0.4])],
                    },
                    {
                        "file": str(second_file),
                        "detections": [],
                    },
                ]
            }

    results = run_detection_pipeline(
        segment_dir=segments_dir,
        output_csv=csv_path,
        runner=FakeRunner(),
        threshold=0.5,
    )

    assert results == [
        expected_segment_result(
            segment_path=str(first_file),
            is_animal=True,
            animal_detections=[animal_detection(confidence=0.91, bbox=[0.1, 0.2, 0.3, 0.4])],
        ),
        expected_segment_result(segment_path=str(second_file)),
    ]

    rows = read_csv_rows(csv_path)

    assert len(rows) == 2
    assert rows[0]["segment_path"] == str(first_file)
    assert rows[0]["max_confidence"] == "0.91"
    assert rows[1]["segment_path"] == str(second_file)
    assert rows[1]["max_confidence"] == ""


def test_run_detection_pipeline_does_not_call_runner_when_no_segments(tmp_path):
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()

    csv_path = tmp_path / "detections.csv"

    runner = Mock()

    results = run_detection_pipeline(
        segment_dir=segments_dir,
        output_csv=csv_path,
        runner=runner,
        threshold=0.5,
    )

    assert results == []
    runner.run.assert_not_called()

    rows = read_csv_rows(csv_path)

    assert rows == []


def test_run_detection_pipeline_writes_run_id_to_csv(tmp_path):
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()

    segment_file = segments_dir / "2026-05-20_14-30-00.jpg"
    segment_file.write_text("")

    csv_path = tmp_path / "detections.csv"

    class FakeRunner:
        def run(self, image_paths):
            return {
                "images": [
                    {
                        "file": str(segment_file),
                        "detections": [],
                    }
                ]
            }

    run_detection_pipeline(
        segment_dir=segments_dir,
        output_csv=csv_path,
        runner=FakeRunner(),
        threshold=0.5,
        run_id="test-run-001",
    )

    rows = read_csv_rows(csv_path)

    assert rows[0]["run_id"] == "test-run-001"


def test_run_detection_pipeline_generates_run_id_when_missing(tmp_path):
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()

    segment_file = segments_dir / "2026-05-20_14-30-00.jpg"
    segment_file.write_text("")

    csv_path = tmp_path / "detections.csv"

    class FakeRunner:
        def run(self, image_paths):
            return {
                "images": [
                    {
                        "file": str(segment_file),
                        "detections": [],
                    }
                ]
            }

    run_detection_pipeline(
        segment_dir=segments_dir,
        output_csv=csv_path,
        runner=FakeRunner(),
        threshold=0.5,
    )

    rows = read_csv_rows(csv_path)

    assert rows[0]["run_id"].startswith("run-")
    assert rows[0]["run_id"] != ""


def test_write_detection_summary_csv(tmp_path):
    summary = {
        "total_segments": 2,
        "animal_segments": 1,
        "failed_segments": 0,
        "total_animal_detections": 1,
        "max_confidence": 0.91,
    }

    csv_path = tmp_path / "summary.csv"

    write_detection_summary_csv(
        summary=summary,
        csv_path=csv_path,
        run_id="run-test-001",
    )

    rows = read_csv_rows(csv_path)

    assert rows == [
        expected_summary_row(
            run_id="run-test-001",
            total_segments="2",
            animal_segments="1",
            failed_segments="0",
            total_animal_detections="1",
            max_confidence="0.91",
        )
    ]


def test_run_detection_pipeline_writes_summary_csv(tmp_path):
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()

    first_file = segments_dir / "2026-05-20_14-30-00.jpg"
    second_file = segments_dir / "2026-05-20_14-31-00.jpg"

    first_file.write_text("")
    second_file.write_text("")

    detections_csv = tmp_path / "detections.csv"
    summary_csv = tmp_path / "summary.csv"

    class FakeRunner:
        def run(self, image_paths):
            return {
                "images": [
                    {
                        "file": str(first_file),
                        "detections": [md_detection(confidence=0.91, bbox=[0.1, 0.2, 0.3, 0.4])],
                    },
                    {
                        "file": str(second_file),
                        "detections": [],
                    },
                ]
            }

    run_detection_pipeline(
        segment_dir=segments_dir,
        output_csv=detections_csv,
        runner=FakeRunner(),
        threshold=0.5,
        run_id="run-test-001",
        summary_csv=summary_csv,
    )

    with summary_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows == [
        expected_summary_row(
            run_id="run-test-001",
            total_segments="2",
            animal_segments="1",
            failed_segments="0",
            total_animal_detections="1",
            max_confidence="0.91",
        )
    ]


def test_write_detection_summary_csv_skips_duplicate_run_id_when_appending(tmp_path):
    csv_path = tmp_path / "summary.csv"

    first_summary = {
        "total_segments": 2,
        "animal_segments": 1,
        "failed_segments": 0,
        "total_animal_detections": 1,
        "max_confidence": 0.91,
    }

    duplicate_summary = {
        "total_segments": 99,
        "animal_segments": 99,
        "failed_segments": 99,
        "total_animal_detections": 99,
        "max_confidence": 0.99,
    }

    write_detection_summary_csv(
        summary=first_summary,
        csv_path=csv_path,
        run_id="run-001",
    )

    write_detection_summary_csv(
        summary=duplicate_summary,
        csv_path=csv_path,
        run_id="run-001",
        append=True,
    )

    rows = read_csv_rows(csv_path)

    assert rows == [
        expected_summary_row(
            run_id="run-001",
            total_segments="2",
            animal_segments="1",
            failed_segments="0",
            total_animal_detections="1",
            max_confidence="0.91",
        )
    ]


def test_main_runs_detection_pipeline_from_cli_args(tmp_path):
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()

    detections_csv = tmp_path / "detections.csv"
    summary_csv = tmp_path / "summary.csv"

    argv = [
        "--segments",
        str(segments_dir),
        "--output",
        str(detections_csv),
        "--summary",
        str(summary_csv),
        "--model",
        "MDV5A",
        "--threshold",
        "0.7",
        "--recursive",
        "--append",
        "--run-id",
        "run-test-001",
    ]

    with patch("detection.animal_detector.MegaDetectorRunner") as mock_runner_cls:
        with patch("detection.animal_detector.run_detection_pipeline") as mock_pipeline:
            mock_runner = mock_runner_cls.return_value
            mock_pipeline.return_value = []

            main(argv)

    mock_runner_cls.assert_called_once_with(model_file="MDV5A")

    mock_pipeline.assert_called_once_with(
        segment_dir=segments_dir,
        output_csv=detections_csv,
        runner=mock_runner,
        threshold=0.7,
        recursive=True,
        append=True,
        run_id="run-test-001",
        summary_csv=summary_csv,
    )


def test_main_rejects_threshold_below_zero(tmp_path):
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()

    with pytest.raises(SystemExit):
        main(
            [
                "--segments",
                str(segments_dir),
                "--output",
                str(tmp_path / "detections.csv"),
                "--model",
                "MDV5A",
                "--threshold",
                "-0.1",
            ]
        )


def test_main_rejects_threshold_above_one(tmp_path):
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()

    with pytest.raises(SystemExit):
        main(
            [
                "--segments",
                str(segments_dir),
                "--output",
                str(tmp_path / "detections.csv"),
                "--model",
                "MDV5A",
                "--threshold",
                "1.1",
            ]
        )


def test_existing_segment_paths_in_csv_returns_empty_set_when_file_missing(tmp_path):
    csv_path = tmp_path / "missing.csv"

    assert existing_segment_paths_in_csv(csv_path) == set()


def test_existing_segment_paths_in_csv_returns_empty_set_when_file_empty(tmp_path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("")

    assert existing_segment_paths_in_csv(csv_path) == set()


def test_run_detection_pipeline_writes_summary_when_summary_csv_is_provided(tmp_path):
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()

    segment_file = segments_dir / "2026-05-20_14-30-00.jpg"
    segment_file.write_text("")

    detections_csv = tmp_path / "detections.csv"
    summary_csv = tmp_path / "summary.csv"

    class FakeRunner:
        def run(self, image_paths):
            return {
                "images": [
                    {
                        "file": str(segment_file),
                        "detections": [md_detection(confidence=0.91, bbox=[0.1, 0.2, 0.3, 0.4])],
                    }
                ]
            }

    run_detection_pipeline(
        segment_dir=segments_dir,
        output_csv=detections_csv,
        runner=FakeRunner(),
        threshold=0.5,
        run_id="run-test-001",
        summary_csv=summary_csv,
    )

    with summary_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows == [
        expected_summary_row(
            run_id="run-test-001",
            total_segments="1",
            animal_segments="1",
            failed_segments="0",
            total_animal_detections="1",
            max_confidence="0.91",
        )
    ]


def test_run_detection_pipeline_writes_empty_summary_when_no_segments(tmp_path):
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()

    detections_csv = tmp_path / "detections.csv"
    summary_csv = tmp_path / "summary.csv"

    runner = Mock()

    results = run_detection_pipeline(
        segment_dir=segments_dir,
        output_csv=detections_csv,
        runner=runner,
        threshold=0.5,
        run_id="run-empty-001",
        summary_csv=summary_csv,
    )

    assert results == []
    runner.run.assert_not_called()

    with summary_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows == [
        expected_summary_row(
            run_id="run-empty-001",
            total_segments="0",
            animal_segments="0",
            failed_segments="0",
            total_animal_detections="0",
            max_confidence="",
        )
    ]


def test_existing_run_ids_in_summary_csv_returns_empty_set_when_file_missing(tmp_path):
    csv_path = tmp_path / "missing_summary.csv"

    assert existing_run_ids_in_summary_csv(csv_path) == set()


def test_existing_run_ids_in_summary_csv_returns_empty_set_when_file_empty(tmp_path):
    csv_path = tmp_path / "empty_summary.csv"
    csv_path.write_text("")

    assert existing_run_ids_in_summary_csv(csv_path) == set()


def test_existing_run_ids_in_summary_csv_returns_existing_run_ids(tmp_path):
    csv_path = tmp_path / "summary.csv"

    csv_path.write_text(
        "run_id,total_segments,animal_segments,failed_segments,total_animal_detections,max_confidence\n"
        "run-001,2,1,0,1,0.91\n"
        "run-002,3,0,1,0,\n"
    )

    assert existing_run_ids_in_summary_csv(csv_path) == {
        "run-001",
        "run-002",
    }


def test_run_detection_pipeline_logs_summary(tmp_path, caplog):
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()

    segment_file = segments_dir / "2026-05-20_14-30-00.jpg"
    segment_file.write_text("")

    detections_csv = tmp_path / "detections.csv"
    summary_csv = tmp_path / "summary.csv"

    runner = Mock()
    runner.run.return_value = {
        "images": [
            {
                "file": str(segment_file),
                "detections": [
                    md_detection(
                        confidence=0.91,
                        bbox=[0.1, 0.2, 0.3, 0.4],
                    )
                ],
            }
        ]
    }

    with caplog.at_level(logging.INFO):
        run_detection_pipeline(
            segment_dir=segments_dir,
            output_csv=detections_csv,
            runner=runner,
            threshold=0.5,
            run_id="run-test-001",
            summary_csv=summary_csv,
        )

    messages = [record.message for record in caplog.records]

    assert "Found 1 segment files" in messages
    assert "Animal segments: 1" in messages
    assert "Animal detections: 1" in messages
    assert "Failed segments: 0" in messages
    assert f"Wrote detections CSV: {detections_csv}" in messages
    assert f"Wrote summary CSV: {summary_csv}" in messages


def test_main_configures_logging(tmp_path):
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()

    detections_csv = tmp_path / "detections.csv"

    argv = [
        "--segments",
        str(segments_dir),
        "--output",
        str(detections_csv),
        "--model",
        "MDV5A",
    ]

    with patch("detection.animal_detector.logging.basicConfig") as mock_basic_config:
        with patch("detection.animal_detector.MegaDetectorRunner") as mock_runner_cls:
            with patch("detection.animal_detector.run_detection_pipeline") as mock_pipeline:
                mock_pipeline.return_value = []

                main(argv)

    mock_basic_config.assert_called_once_with(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )


def test_main_configures_custom_log_level(tmp_path):
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()

    detections_csv = tmp_path / "detections.csv"

    argv = [
        "--segments",
        str(segments_dir),
        "--output",
        str(detections_csv),
        "--model",
        "MDV5A",
        "--log-level",
        "DEBUG",
    ]

    with patch("detection.animal_detector.logging.basicConfig") as mock_basic_config:
        with patch("detection.animal_detector.MegaDetectorRunner"):
            with patch("detection.animal_detector.run_detection_pipeline") as mock_pipeline:
                mock_pipeline.return_value = []

                main(argv)

    mock_basic_config.assert_called_once_with(
        level=logging.DEBUG,
        format="%(levelname)s:%(name)s:%(message)s",
    )


def test_main_rejects_invalid_log_level(tmp_path):
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()

    with pytest.raises(SystemExit):
        main(
            [
                "--segments",
                str(segments_dir),
                "--output",
                str(tmp_path / "detections.csv"),
                "--model",
                "MDV5A",
                "--log-level",
                "VERBOSE",
            ]
        )
