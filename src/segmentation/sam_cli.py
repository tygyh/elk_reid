import argparse
import logging
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.segmentation.sam_segmentation import create_sam_predictor, segment_and_save_many_images

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", dest="images", action="append", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-type", default="vit_b")
    parser.add_argument("--device", default="cuda_if_available")
    parser.add_argument("--no-multimask-output", dest="multimask_output", action="store_false")
    parser.add_argument("--crop", action="store_true")
    parser.set_defaults(multimask_output=True)
    parser.add_argument("--min-area", type=int, default=None)
    parser.add_argument("--output-format", choices=["png", "jpg"], default="png")
    parser.add_argument("--quiet", action="store_true")

    return parser.parse_args(argv)


def load_rgb_image(image_path):
    if not image_path.exists():
        raise ValueError("image is missing")

    try:
        return Image.open(image_path).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("invalid image file") from exc


def print_saved_paths(paths):
    for image_paths in paths:
        for path in image_paths:
            print(path)


def run_segmentation_cli(
        image_paths,
        checkpoint_path,
        output_dir,
        model_type="vit_b",
        device="cuda_if_available",
        multimask_output=True,
        crop=False,
        min_area=None,
        output_format="png",
):
    logger.info("Starting segmentation for %s image(s)", len(image_paths))

    images = [load_rgb_image(image_path) for image_path in image_paths]

    predictor = create_sam_predictor(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        device=device,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = segment_and_save_many_images(
        images=images,
        predictor=predictor,
        output_dir=output_dir,
        multimask_output=multimask_output,
        crop=crop,
        min_area=min_area,
        output_format=output_format,
    )

    saved_count = sum(len(paths) for paths in saved_paths)
    logger.info("Saved %s mask image(s)", saved_count)

    return saved_paths


def main(argv=None, runner=run_segmentation_cli):
    args = parse_args(argv)

    return run_from_args(args, runner=runner)


def configure_logging(quiet=False):
    if quiet:
        logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
        return

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def run_from_args(args, runner=run_segmentation_cli):
    saved_paths = runner(
        image_paths=[Path(image) for image in args.images],
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        model_type=args.model_type,
        device=args.device,
        multimask_output=args.multimask_output,
        crop=args.crop,
        min_area=args.min_area,
        output_format=args.output_format,
    )

    print_saved_paths(saved_paths)

    return saved_paths


def cli_entrypoint(argv=None, main_func=run_from_args):
    args = parse_args(argv)
    configure_logging(quiet=args.quiet)

    try:
        main_func(args)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    return 0
