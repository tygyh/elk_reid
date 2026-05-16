import numpy as np
from PIL import Image


def segment_image(image, predictor=None, box=None, multimask_output=True):
    validate_image(image)

    if predictor is None:
        return [
            np.array(
                [
                    [True, False, True],
                    [False, True, False],
                    [True, False, True],
                ],
                dtype=bool,
            )
        ]

    image_np = np.array(image)

    predictor.set_image(image_np)

    masks, scores, logits = predictor.predict(
        box=box,
        multimask_output=multimask_output,
    )

    return format_sam_output(masks, scores, logits)


def validate_image(image):
    if image is None:
        raise ValueError("invalid image")

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image")

    if image.size[0] == 0 or image.size[1] == 0:
        raise ValueError("empty image")

    if image.mode != "RGB":
        raise ValueError("image must be RGB")


def validate_mask(image, mask):
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a numpy array")

    if mask.dtype != bool:
        raise ValueError("mask must be boolean")

    image_width, image_height = image.size
    expected_shape = (image_height, image_width)

    if mask.shape != expected_shape:
        raise ValueError("mask shape must match image")


def apply_mask(image, mask):
    validate_image(image)
    validate_mask(image, mask)

    image_array = np.array(image).copy()
    image_array[~mask] = [0, 0, 0]

    return Image.fromarray(image_array, mode="RGB")


def apply_masks(image, masks):
    return [apply_mask(image, mask) for mask in masks]


def mask_area(mask):
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a numpy array")

    if mask.dtype != bool:
        raise ValueError("mask must be boolean")

    return int(np.sum(mask))


def filter_masks(masks, min_area):
    return [mask for mask in masks if mask_area(mask) >= min_area]


def sort_masks_by_area(masks):
    return sorted(masks, key=mask_area, reverse=True)


def format_sam_output(masks, scores, logits):
    sorted_indices = np.argsort(scores)[::-1]

    return [masks[index].astype(bool) for index in sorted_indices]


def crop_masked_region(image, mask):
    validate_image(image)
    validate_mask(image, mask)

    rows, cols = np.where(mask)

    top = rows.min()
    bottom = rows.max() + 1
    left = cols.min()
    right = cols.max() + 1

    return image.crop((left, top, right, bottom))


def save_masked_image(image, output_path):
    validate_image(image)

    image.save(output_path)


def segment_and_save_masks(image, predictor, output_dir, box=None):
    masks = segment_image(image, predictor=predictor, box=box)
    masked_images = apply_masks(image, masks)

    paths = []
    for index, masked_image in enumerate(masked_images):
        output_path = output_dir / f"mask_{index}.png"
        save_masked_image(masked_image, output_path)
        paths.append(output_path)

    return paths
