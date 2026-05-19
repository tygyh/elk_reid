import pytest

from src.segmentation.sam_segmentation import *


class FakePredictor:
    def __init__(self, masks):
        self.masks = np.array(masks)
        self.set_image_called_with = None
        self.predict_called_with = None

    def set_image(self, img_np):
        self.set_image_called_with = img_np

    def predict(self, box, multimask_output):
        self.predict_called_with = {
            "box": box,
            "multimask_output": multimask_output,
        }
        scores = np.ones(len(self.masks))
        logits = self.masks.astype(float)
        return self.masks, scores, logits


@pytest.fixture
def sam_predictor():
    return create_sam_predictor(
        checkpoint_path="weights/sam_vit_h.pth",
        model_type="vit_h",
        device="cuda",
    )


def test_segment_image_rejects_invalid_image():
    with pytest.raises(ValueError, match="invalid image"):
        segment_image(None)


def test_segment_image_rejects_empty_image():
    image = Image.new("RGB", (0, 0))

    with pytest.raises(ValueError, match="empty image"):
        segment_image(image)


def test_segment_image_rejects_non_pil_image():
    with pytest.raises(TypeError, match="image must be a PIL Image"):
        segment_image([])


def test_segment_image_rejects_grayscale_image():
    image = Image.new("L", (3, 3))

    with pytest.raises(ValueError, match="image must be RGB"):
        segment_image(image)


def test_segment_image_rejects_wrong_channel_count():
    image = Image.new("RGBA", (3, 3))

    with pytest.raises(ValueError, match="image must be RGB"):
        segment_image(image)


def test_segment_image_returns_masks(image_a, x_mask):
    result = segment_image(image_a)

    assert len(result) == 1
    np.testing.assert_array_equal(result[0], x_mask)


def test_apply_mask_keeps_only_masked_pixels(image_a, x_mask):
    result = apply_mask(image_a, x_mask)

    expected = np.array(
        [
            [[10, 20, 30], [0, 0, 0], [70, 80, 90]],
            [[0, 0, 0], [45, 55, 65], [0, 0, 0]],
            [[20, 30, 40], [0, 0, 0], [80, 90, 100]],
        ],
        dtype=np.uint8,
    )

    np.testing.assert_array_equal(np.array(result), expected)


def test_apply_mask_rejects_non_boolean_mask(image_a):
    bad_mask = np.ones((3, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="mask must be boolean"):
        apply_mask(image_a, bad_mask)


def test_apply_mask_rejects_mask_with_wrong_shape(image_a):
    bad_mask = np.ones((2, 2), dtype=bool)

    with pytest.raises(ValueError, match="mask shape must match image"):
        apply_mask(image_a, bad_mask)


def test_apply_mask_rejects_non_numpy_mask(image_a):
    bad_mask = [
        [True, False, True],
        [False, True, False],
        [True, False, True],
    ]

    with pytest.raises(TypeError, match="mask must be a numpy array"):
        apply_mask(image_a, bad_mask)


def test_apply_masks_applies_multiple_masks(image_a, x_mask, y_mask):
    result = apply_masks(image_a, [x_mask, y_mask])

    assert len(result) == 2


def test_mask_area_rejects_non_numpy_mask():
    mask = [
        [True, False, True],
        [False, True, False],
        [True, False, True],
    ]

    with pytest.raises(TypeError, match="mask must be a numpy array"):
        mask_area(mask)


def test_mask_area_rejects_non_boolean_mask():
    mask = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ],
        dtype=np.uint8,
    )

    with pytest.raises(ValueError, match="mask must be boolean"):
        mask_area(mask)


def test_filter_masks_removes_small_masks(x_mask, tiny_mask):
    result = filter_masks([x_mask, tiny_mask], min_area=2)

    assert len(result) == 1
    np.testing.assert_array_equal(result[0], x_mask)


def test_sort_masks_by_area_largest_first(x_mask, tiny_mask):
    result = sort_masks_by_area([tiny_mask, x_mask])

    np.testing.assert_array_equal(result[0], x_mask)
    np.testing.assert_array_equal(result[1], tiny_mask)


def test_segment_image_uses_predictor(image_a, x_mask):
    result = segment_image(image_a, predictor=FakePredictor([x_mask]))

    assert len(result) == 1
    np.testing.assert_array_equal(result[0], x_mask)


def test_segment_image_uses_sam_predictor(image_a, x_mask):
    predictor = FakePredictor([x_mask])
    box = np.array([0, 0, 3, 3])

    result = segment_image(image_a, predictor=predictor, box=box)

    np.testing.assert_array_equal(
        predictor.set_image_called_with,
        np.array(image_a),
    )

    np.testing.assert_array_equal(
        predictor.predict_called_with["box"],
        box,
    )
    assert predictor.predict_called_with["multimask_output"] is True

    assert len(result) == 1
    np.testing.assert_array_equal(result[0], x_mask)


def test_format_sam_output_returns_masks_only(x_mask):
    masks = np.array([x_mask])
    scores = np.array([1.0])
    logits = masks.astype(float)

    result = format_sam_output(masks, scores, logits)

    assert len(result) == 1
    np.testing.assert_array_equal(result[0], x_mask)


def test_format_sam_output_sorts_masks_by_score_descending(x_mask, tiny_mask):
    low_score_mask = tiny_mask
    high_score_mask = x_mask

    masks = np.array([low_score_mask, high_score_mask])
    scores = np.array([0.2, 0.9])
    logits = masks.astype(float)

    result = format_sam_output(masks, scores, logits)

    np.testing.assert_array_equal(result[0], high_score_mask)
    np.testing.assert_array_equal(result[1], low_score_mask)


def test_format_sam_output_returns_boolean_masks(x_mask):
    masks = np.array([x_mask.astype(float)])
    scores = np.array([1.0])
    logits = masks

    result = format_sam_output(masks, scores, logits)

    assert result[0].dtype == bool


def test_crop_masked_region_crops_to_mask_bounds(image_a, small_mask):
    result = crop_masked_region(image_a, small_mask)

    expected = np.array(
        [
            [[45, 55, 65], [75, 85, 95]],
            [[50, 60, 70], [80, 90, 100]],
        ],
        dtype=np.uint8,
    )

    np.testing.assert_array_equal(np.array(result), expected)


def test_save_masked_image_saves_file(tmp_path, image_a, x_mask):
    masked_image = apply_mask(image_a, x_mask)
    output_path = tmp_path / "mask.png"

    save_masked_image(masked_image, output_path)

    assert output_path.exists()

    saved = Image.open(output_path)
    np.testing.assert_array_equal(np.array(saved), np.array(masked_image))


def test_segment_and_save_masks_saves_masked_images(
        tmp_path,
        image_a,
        x_mask,
):
    predictor = FakePredictor([x_mask])
    box = np.array([0, 0, 3, 3])

    paths = segment_and_save_masks(
        image_a,
        predictor=predictor,
        output_dir=tmp_path,
        box=box,
    )

    assert len(paths) == 1
    assert paths[0].exists()


@pytest.mark.integration
def test_create_sam_predictor(sam_predictor):
    assert sam_predictor is not None


@pytest.mark.integration
def test_segment_image_with_real_sam(image_a, sam_predictor):
    box = np.array([0, 0, 3, 3])

    masks = segment_image(
        image_a,
        predictor=sam_predictor,
        box=box,
    )

    assert len(masks) > 0
    assert masks[0].shape == (3, 3)
    assert masks[0].dtype == bool


@pytest.mark.integration
def test_segment_and_save_masks_with_real_sam(tmp_path, image_a, sam_predictor):
    box = np.array([0, 0, 3, 3])

    paths = segment_and_save_masks(
        image_a,
        predictor=sam_predictor,
        output_dir=tmp_path,
        box=box,
    )

    assert len(paths) > 0
    assert paths[0].exists()


def test_create_sam_predictor_loads_model(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "sam.pth"
    checkpoint_path.write_text("fake checkpoint")

    called = {}

    class FakeSam:
        def to(self, **kwargs):
            called["device"] = kwargs["device"]

    class FakeSamPredictor:
        def __init__(self, sam_model):
            called["sam"] = sam_model

    def fake_model_factory(**kwargs):
        called["checkpoint"] = kwargs["checkpoint"]
        return FakeSam()

    monkeypatch.setitem(
        sam_model_registry,
        "vit_b",
        fake_model_factory,
    )
    monkeypatch.setattr(
        "src.segmentation.sam_segmentation.SamPredictor",
        FakeSamPredictor,
    )

    result = create_sam_predictor(
        checkpoint_path=checkpoint_path,
        model_type="vit_b",
        device="cpu",
    )

    assert isinstance(result, FakeSamPredictor)
    assert called["checkpoint"] == str(checkpoint_path)
    assert called["device"] == torch.device("cpu")


def test_select_device_returns_cuda_when_requested_and_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    result = select_device("cuda")

    assert result == torch.device("cuda")


def test_select_device_rejects_cuda_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="cuda is not available"):
        select_device("cuda")


def test_select_device_returns_cpu():
    result = select_device("cpu")

    assert result == torch.device("cpu")


def test_select_device_returns_cpu_when_cuda_with_fallback_requested(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    result = select_device("cuda_if_available")

    assert result == torch.device("cpu")


def test_select_device_returns_cuda_when_cuda_with_fallback_requested(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    result = select_device("cuda_if_available")

    assert result == torch.device("cuda")


def test_create_full_image_box(image_a):
    result = create_full_image_box(image_a)

    np.testing.assert_array_equal(
        result,
        np.array([0, 0, 3, 3]),
    )


def test_segment_image_uses_full_image_box_by_default(image_a, x_mask):
    predictor = FakePredictor([x_mask])

    segment_image(image_a, predictor=predictor)

    np.testing.assert_array_equal(
        predictor.predict_called_with["box"],
        np.array([0, 0, 3, 3]),
    )


def test_segment_and_save_many_images(tmp_path, image_a, image_b, x_mask):
    predictor = FakePredictor([x_mask])

    paths = segment_and_save_many_images(
        images=[image_a, image_b],
        predictor=predictor,
        output_dir=tmp_path,
    )

    assert len(paths) == 2
    assert len(paths[0]) == 1
    assert len(paths[1]) == 1

    assert paths[0][0].exists()
    assert paths[1][0].exists()


def test_create_sam_predictor_rejects_unknown_model_type(tmp_path):
    checkpoint_path = tmp_path / "sam.pth"
    checkpoint_path.write_text("fake checkpoint")

    with pytest.raises(ValueError, match="unknown model type"):
        create_sam_predictor(
            checkpoint_path=checkpoint_path,
            model_type="vit_x",
            device="cpu",
        )


def test_create_sam_predictor_rejects_missing_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "missing.pth"

    with pytest.raises(ValueError, match="checkpoint is missing"):
        create_sam_predictor(
            checkpoint_path=checkpoint_path,
            model_type="vit_b",
            device="cpu",
        )
