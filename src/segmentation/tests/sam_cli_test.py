import pytest

from src.segmentation.sam_cli import *


@pytest.mark.parametrize(
    ("flag", "field", "expected"),
    [
        ("--crop", "crop", True),
        ("--quiet", "quiet", True),
        ("--no-multimask-output", "multimask_output", False),
    ],
)
def test_parse_args_boolean_flags(flag, field, expected):
    args = parse_args(default_args("outputs") + [flag])

    assert getattr(args, field) is expected


def patch_segmentation(monkeypatch, result, called=None):
    class FakePredictor:
        pass

    def fake_create_sam_predictor(**kwargs):
        if called is not None:
            called["checkpoint_path"] = kwargs["checkpoint_path"]
            called["model_type"] = kwargs["model_type"]
            called["device"] = kwargs["device"]

        return FakePredictor()

    def fake_segment(
            images,
            predictor,
            output_dir,
            multimask_output,
            crop,
            min_area,
            output_format,
    ):
        if called is not None:
            called.update(
                {
                    "images": images,
                    "predictor": predictor,
                    "output_dir": output_dir,
                    "multimask_output": multimask_output,
                    "crop": crop,
                    "min_area": min_area,
                    "output_format": output_format,
                }
            )

        return result

    monkeypatch.setattr(
        "src.segmentation.sam_cli.create_sam_predictor",
        fake_create_sam_predictor,
    )

    monkeypatch.setattr(
        "src.segmentation.sam_cli.segment_and_save_many_images",
        fake_segment,
    )

    return FakePredictor


def default_args(output_dir):
    return [
        "--image",
        "input.png",
        "--checkpoint",
        "weights/sam_vit_b.pth",
        "--output-dir",
        output_dir,
    ]


def make_fake_runner(tmp_path, called):
    def fake_runner(
            image_paths,
            checkpoint_path,
            output_dir,
            model_type,
            device,
            multimask_output,
            crop,
            min_area,
            output_format,
    ):
        called.update(
            {
                "image_paths": image_paths,
                "checkpoint_path": checkpoint_path,
                "output_dir": output_dir,
                "model_type": model_type,
                "device": device,
                "multimask_output": multimask_output,
                "crop": crop,
                "min_area": min_area,
                "output_format": output_format,
            }
        )

        return [[tmp_path / "image_0" / f"mask_0.{output_format}"]]

    return fake_runner


def assert_runner_args(called, tmp_path, **overrides):
    expected = {
        "image_paths": [Path("input.png")],
        "checkpoint_path": Path("weights/sam_vit_b.pth"),
        "output_dir": tmp_path,
        "model_type": "vit_b",
        "device": "cuda_if_available",
        "multimask_output": True,
        "crop": False,
        "min_area": None,
        "output_format": "png",
    }

    expected.update(overrides)

    assert called == expected


def assert_default_runner_args(called, tmp_path):
    assert_runner_args(called, tmp_path)


def test_cli_calls_pipeline_with_fake_runner(tmp_path):
    called_with = {}

    fake_runner = make_fake_runner(tmp_path, called_with)

    result = main(default_args(str(tmp_path)), runner=fake_runner)

    assert result == [[tmp_path / "image_0" / "mask_0.png"]]
    assert_default_runner_args(called_with, tmp_path)


def test_run_segmentation_cli_loads_image_creates_predictor_and_saves_masks(
        monkeypatch,
        tmp_path,
        image_a,
):
    image_path = tmp_path / "input.png"
    checkpoint_path = tmp_path / "sam.pth"
    output_dir = tmp_path / "outputs"

    image_a.save(image_path)
    checkpoint_path.write_text("fake checkpoint")

    called = {}

    FakePredictor = patch_segmentation(
        monkeypatch,
        [[output_dir / "image_0" / "mask_0.png"]],
        called=called,
    )

    result = run_segmentation_cli(
        image_paths=[image_path],
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
    )

    assert result == [[output_dir / "image_0" / "mask_0.png"]]

    assert called["checkpoint_path"] == checkpoint_path
    assert called["model_type"] == "vit_b"
    assert called["device"] == "cuda_if_available"

    assert len(called["images"]) == 1
    assert called["predictor"].__class__ is FakePredictor
    assert called["output_dir"] == output_dir
    assert called["multimask_output"] is True
    assert called["crop"] is False
    assert called["min_area"] is None
    assert called["output_format"] == "png"


def test_parse_args_accepts_model_type_and_device():
    args = parse_args(default_args("outputs") + ["--model-type", "vit_h", "--device", "cpu"])

    assert args.model_type == "vit_h"
    assert args.device == "cpu"


def test_run_segmentation_cli_rejects_missing_image(tmp_path):
    image_path = tmp_path / "missing.png"
    checkpoint_path = tmp_path / "sam.pth"
    output_dir = tmp_path / "outputs"

    checkpoint_path.write_text("fake checkpoint")

    with pytest.raises(ValueError, match="image is missing"):
        run_segmentation_cli(
            image_paths=[image_path],
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
        )


def test_run_segmentation_cli_rejects_non_image_file(tmp_path):
    image_path = tmp_path / "not_image.txt"
    checkpoint_path = tmp_path / "sam.pth"
    output_dir = tmp_path / "outputs"

    image_path.write_text("not an image")
    checkpoint_path.write_text("fake checkpoint")

    with pytest.raises(ValueError, match="invalid image file"):
        run_segmentation_cli(
            image_paths=[image_path],
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
        )


def test_parse_args_accepts_multiple_images():
    args = parse_args(
        [
            "--image",
            "input1.png",
            "--image",
            "input2.png",
            "--checkpoint",
            "weights/sam_vit_b.pth",
            "--output-dir",
            "outputs",
        ]
    )

    assert args.images == ["input1.png", "input2.png"]


def test_run_segmentation_cli_handles_multiple_images(
        monkeypatch,
        tmp_path,
        image_a,
):
    image_path_1 = tmp_path / "input1.png"
    image_path_2 = tmp_path / "input2.png"
    checkpoint_path = tmp_path / "sam.pth"
    output_dir = tmp_path / "outputs"

    image_a.save(image_path_1)
    image_a.save(image_path_2)
    checkpoint_path.write_text("fake checkpoint")

    patch_segmentation(
        monkeypatch,
        [
            [output_dir / "image_0" / "mask_0.png"],
            [output_dir / "image_1" / "mask_0.png"],
        ],
    )

    result = run_segmentation_cli(
        image_paths=[image_path_1, image_path_2],
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
    )

    assert result == [
        [output_dir / "image_0" / "mask_0.png"],
        [output_dir / "image_1" / "mask_0.png"],
    ]


def test_load_rgb_image_rejects_image_that_cannot_convert_to_rgb(
        monkeypatch,
        tmp_path,
):
    image_path = tmp_path / "image.png"
    image_path.write_text("fake image bytes")

    class FakeImage:
        def convert(self, mode):
            assert mode == "RGB"
            raise ValueError("cannot convert")

    monkeypatch.setattr(
        "src.segmentation.sam_cli.Image.open",
        lambda path: FakeImage(),
    )

    with pytest.raises(ValueError, match="invalid image file"):
        load_rgb_image(image_path)


def test_print_saved_paths(capsys, tmp_path):
    paths = [
        [tmp_path / "image_0" / "mask_0.png"],
        [
            tmp_path / "image_1" / "mask_0.png",
            tmp_path / "image_1" / "mask_1.png",
        ],
    ]

    print_saved_paths(paths)

    captured = capsys.readouterr()

    assert captured.out == (
        f"{tmp_path / 'image_0' / 'mask_0.png'}\n"
        f"{tmp_path / 'image_1' / 'mask_0.png'}\n"
        f"{tmp_path / 'image_1' / 'mask_1.png'}\n"
    )


def test_main_prints_saved_paths(capsys, tmp_path):
    called = {}

    result = main(default_args(str(tmp_path)) + ["--min-area", "5"], runner=make_fake_runner(tmp_path, called))

    captured = capsys.readouterr()

    assert result == [[tmp_path / "image_0" / "mask_0.png"]]
    assert captured.out == f"{tmp_path / 'image_0' / 'mask_0.png'}\n"


def test_cli_entrypoint_returns_zero_on_success():
    def fake_main(args):
        return []

    result = cli_entrypoint(default_args("outputs"), main_func=fake_main)

    assert result == 0


def test_cli_entrypoint_returns_one_on_error(capsys):
    def fake_main(args):
        raise ValueError("image is missing")

    result = cli_entrypoint(default_args("outputs"), main_func=fake_main)

    captured = capsys.readouterr()

    assert result == 1
    assert captured.err == "image is missing\n"


def test_parse_args_defaults():
    args = parse_args(default_args("outputs"))

    assert args.images == ["input.png"]
    assert args.checkpoint == "weights/sam_vit_b.pth"
    assert args.output_dir == "outputs"
    assert args.multimask_output is True
    assert args.crop is False
    assert args.min_area is None
    assert args.output_format == "png"
    assert args.quiet is False


def test_main_passes_multimask_output_false(tmp_path):
    called = {}

    main(default_args(str(tmp_path)) + ["--no-multimask-output"], runner=make_fake_runner(tmp_path, called))

    assert called["multimask_output"] is False


def test_main_passes_crop_true(tmp_path):
    called = {}

    main(default_args(str(tmp_path)) + ["--crop"], runner=make_fake_runner(tmp_path, called))

    assert called["crop"] is True


def test_parse_args_accepts_optional_flags():
    args = parse_args(
        default_args("outputs") +
        [
            "--crop",
            "--min-area",
            "5",
            "--output-format",
            "jpg",
            "--quiet"
        ]
    )

    assert args.crop is True
    assert args.min_area == 5
    assert args.output_format == "jpg"
    assert args.quiet is True


def test_main_passes_min_area(tmp_path):
    called = {}

    main(default_args(str(tmp_path)) + ["--min-area", "5"], runner=make_fake_runner(tmp_path, called))

    assert called["min_area"] == 5


def test_run_segmentation_cli_logs_start_and_finish(
    monkeypatch,
    caplog,
    tmp_path,
    image_a,
):
    image_path = tmp_path / "input.png"
    checkpoint_path = tmp_path / "sam.pth"
    output_dir = tmp_path / "outputs"

    image_a.save(image_path)
    checkpoint_path.write_text("fake checkpoint")

    patch_segmentation(
        monkeypatch,
        [[output_dir / "image_0" / "mask_0.png"]],
    )

    with caplog.at_level("INFO", logger="src.segmentation.sam_cli"):
        run_segmentation_cli(
            image_paths=[image_path],
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
        )

    assert "Starting segmentation for 1 image(s)" in caplog.text
    assert "Saved 1 mask image(s)" in caplog.text


def test_configure_logging_uses_error_level_when_quiet(monkeypatch):
    called = {}

    def fake_basic_config(level, format):
        called["level"] = level
        called["format"] = format

    monkeypatch.setattr("src.segmentation.sam_cli.logging.basicConfig", fake_basic_config)

    configure_logging(quiet=True)

    assert called["level"] == logging.ERROR


def test_configure_logging_uses_info_level_by_default(monkeypatch):
    called = {}

    def fake_basic_config(level, format):
        called["level"] = level
        called["format"] = format

    monkeypatch.setattr("src.segmentation.sam_cli.logging.basicConfig", fake_basic_config)

    configure_logging(quiet=False)

    assert called["level"] == logging.INFO


def test_run_from_args_calls_runner(tmp_path):
    args = parse_args(
        default_args(str(tmp_path)) +
        [
            "--output-format",
            "jpg",
            "--crop",
            "--min-area",
            "5",
            "--no-multimask-output"
        ]
    )

    called = {}

    fake_runner = make_fake_runner(tmp_path, called)

    result = run_from_args(args, runner=fake_runner)

    assert result == [[tmp_path / "image_0" / "mask_0.jpg"]]
    assert called["output_format"] == "jpg"
    assert called["crop"] is True
    assert called["min_area"] == 5
    assert called["multimask_output"] is False


def test_run_segmentation_cli_passes_options(
        monkeypatch,
        tmp_path,
        image_a,
):
    image_path = tmp_path / "input.png"
    checkpoint_path = tmp_path / "sam.pth"
    output_dir = tmp_path / "outputs"

    image_a.save(image_path)
    checkpoint_path.write_text("fake checkpoint")

    called = {}

    FakePredictor = patch_segmentation(
        monkeypatch,
        [[output_dir / "image_0" / "mask_0.jpg"]],
        called=called,
    )

    result = run_segmentation_cli(
        image_paths=[image_path],
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        multimask_output=False,
        crop=True,
        min_area=5,
        output_format="jpg",
    )

    assert result == [[output_dir / "image_0" / "mask_0.jpg"]]

    assert len(called["images"]) == 1
    assert called["predictor"].__class__ is FakePredictor
    assert called["output_dir"] == output_dir
    assert called["multimask_output"] is False
    assert called["crop"] is True
    assert called["min_area"] == 5
    assert called["output_format"] == "jpg"
