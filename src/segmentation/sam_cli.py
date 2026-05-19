import argparse
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.segmentation.sam_segmentation import create_sam_predictor, segment_and_save_masks


def parse_args(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-type", default="vit_b")
    parser.add_argument("--device", default="cuda_if_available")

    return parser.parse_args(argv)


def run_segmentation_cli(
    image_path,
    checkpoint_path,
    output_dir,
    model_type="vit_b",
    device="cuda_if_available",
):
    if not image_path.exists():
        raise ValueError("image is missing")

    try:
        image = Image.open(image_path).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("invalid image file") from exc

    predictor = create_sam_predictor(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        device=device,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    return segment_and_save_masks(
        image=image,
        predictor=predictor,
        output_dir=output_dir,
    )


def main(argv=None, runner=run_segmentation_cli):
    args = parse_args(argv)

    return runner(
        image_path=Path(args.image),
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        model_type=args.model_type,
        device=args.device,
    )
