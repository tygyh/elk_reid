import pytest

from src.segmentation.sam_cli import *


def test_parse_args_requires_image_checkpoint_and_output_dir():
    args = parse_args(
        [
            "--image",
            "input.png",
            "--checkpoint",
            "weights/sam_vit_h.pth",
            "--output-dir",
            "outputs",
        ]
    )

    assert args.image == "input.png"
    assert args.checkpoint == "weights/sam_vit_h.pth"
    assert args.output_dir == "outputs"


def test_cli_calls_pipeline_with_fake_runner(tmp_path):
    called_with = {}

    def fake_runner(image_path, checkpoint_path, output_dir, model_type, device):
        called_with["image_path"] = image_path
        called_with["checkpoint_path"] = checkpoint_path
        called_with["output_dir"] = output_dir
        called_with["model_type"] = model_type
        called_with["device"] = device
        return [tmp_path / "mask_0.png"]

    result = main(
        [
            "--image",
            "input.png",
            "--checkpoint",
            "weights/sam_vit_h.pth",
            "--output-dir",
            str(tmp_path),
        ],
        runner=fake_runner,
    )

    assert result == [tmp_path / "mask_0.png"]
    assert called_with == {
        "image_path": Path("input.png"),
        "checkpoint_path": Path("weights/sam_vit_h.pth"),
        "output_dir": tmp_path,
        "model_type": "vit_b",
        "device": "cuda_if_available",
    }


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

    class FakePredictor:
        pass

    def fake_create_sam_predictor(**kwargs):
        called["checkpoint_path"] = kwargs["checkpoint_path"]
        called["model_type"] = kwargs["model_type"]
        called["device"] = kwargs["device"]
        return FakePredictor()

    def fake_segment_and_save_masks(image, predictor, output_dir):
        called["image"] = image
        called["predictor"] = predictor
        called["output_dir"] = output_dir
        return [output_dir / "mask_0.png"]

    monkeypatch.setattr(
        "src.segmentation.sam_cli.create_sam_predictor",
        fake_create_sam_predictor,
    )
    monkeypatch.setattr(
        "src.segmentation.sam_cli.segment_and_save_masks",
        fake_segment_and_save_masks,
    )

    result = run_segmentation_cli(
        image_path=image_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
    )

    assert result == [output_dir / "mask_0.png"]
    assert called["checkpoint_path"] == checkpoint_path
    assert called["model_type"] == "vit_b"
    assert called["device"] == "cuda_if_available"
    assert called["predictor"].__class__ is FakePredictor
    assert called["output_dir"] == output_dir


def test_parse_args_accepts_model_type_and_device():
    args = parse_args(
        [
            "--image",
            "input.png",
            "--checkpoint",
            "weights/sam_vit_h.pth",
            "--output-dir",
            "outputs",
            "--model-type",
            "vit_h",
            "--device",
            "cpu",
        ]
    )

    assert args.model_type == "vit_h"
    assert args.device == "cpu"


def test_cli_calls_runner(tmp_path):
    called = {}

    def fake_runner(
            image_path,
            checkpoint_path,
            output_dir,
            model_type,
            device,
    ):
        called["image_path"] = image_path
        called["checkpoint_path"] = checkpoint_path
        called["output_dir"] = output_dir
        called["model_type"] = model_type
        called["device"] = device
        return [tmp_path / "mask_0.png"]

    result = main(
        [
            "--image",
            "input.png",
            "--checkpoint",
            "models/sam_vit_b_01ec64.pth",
            "--output-dir",
            str(tmp_path),
        ],
        runner=fake_runner,
    )

    assert result == [tmp_path / "mask_0.png"]
    assert called["model_type"] == "vit_b"
    assert called["device"] == "cuda_if_available"


def test_run_segmentation_cli_rejects_missing_image(tmp_path):
    image_path = tmp_path / "missing.png"
    checkpoint_path = tmp_path / "sam.pth"
    output_dir = tmp_path / "outputs"

    checkpoint_path.write_text("fake checkpoint")

    with pytest.raises(ValueError, match="image is missing"):
        run_segmentation_cli(
            image_path=image_path,
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
            image_path=image_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
        )

def test_run_segmentation_cli_rejects_image_that_cannot_convert_to_rgb(
    monkeypatch,
    tmp_path,
):
    image_path = tmp_path / "image.png"
    checkpoint_path = tmp_path / "sam.pth"
    output_dir = tmp_path / "outputs"

    image_path.write_text("fake image bytes")
    checkpoint_path.write_text("fake checkpoint")

    class FakeImage:
        def convert(self, mode):
            raise ValueError("cannot convert")

    monkeypatch.setattr(
        "src.segmentation.sam_cli.Image.open",
        lambda path: FakeImage(),
    )

    with pytest.raises(ValueError, match="invalid image file"):
        run_segmentation_cli(
            image_path=image_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
        )