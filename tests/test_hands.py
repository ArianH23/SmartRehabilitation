import numpy as np
import pytest

from smartrehab.hands import hand_bbox, landmark_features, phantom_object_center, to_frame_pixels


def thesis_bbox(xs, ys, kind, image_width, image_height, limit):
    """Bounding box loop as written in the thesis' 1_mp_BBoxes.py (EK / Yale variants)."""
    max_x = max_y = 0
    min_x = min_y = limit
    for x_val, y_val in zip(xs, ys):
        if kind == "crop":
            px = x_val * 1080 + 420
        else:
            px = x_val * image_width
        if kind == "pad":
            py = y_val * 1920 - 420
        else:
            py = y_val * image_height
        max_x, min_x = max(max_x, px), min(min_x, px)
        max_y, min_y = max(max_y, py), min(min_y, py)

    min_x = image_width - min_x
    max_x = image_width - max_x
    min_x = int(np.round(min_x))
    max_x = int(np.round(max_x))
    min_x, max_x = max_x, min_x
    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(image_width, max_x)
    max_y = min(image_height, max_y)
    width = max_x - min_x
    height = int(np.round(max_y)) - int(np.round(min_y))
    return [min_x, int(np.round(min_y)), width, height]


@pytest.mark.parametrize("kind", ["plain", "pad", "crop"])
def test_hand_bbox_matches_thesis_loop(kind):
    rng = np.random.default_rng(7)
    for _ in range(200):
        xs, ys = rng.random(21), rng.random(21)
        assert hand_bbox(xs, ys, kind, 1920, 1080) == thesis_bbox(xs, ys, kind, 1920, 1080, 1920)


def test_hand_bbox_plain_yale_size():
    rng = np.random.default_rng(3)
    for _ in range(100):
        xs, ys = rng.random(21), rng.random(21)
        assert hand_bbox(xs, ys, "plain", 640, 480) == thesis_bbox(xs, ys, "plain", 640, 480, 640)


def test_hand_bbox_undoes_mirroring():
    # A hand on the left of the mirrored image is on the right of the real frame.
    xs, ys = np.array([0.1, 0.2]), np.array([0.5, 0.6])
    min_x, _, width, _ = hand_bbox(xs, ys, "plain", 1000, 500)
    assert min_x == 800 and width == 100


def test_pad_and_crop_offsets_follow_frame_size():
    # 1920x1080: 420 px of padding / cropping on each side; 640x480 would be 80.
    px, py = to_frame_pixels([0.5], [0.5], "crop", 1920, 1080)
    assert px[0] == 0.5 * 1080 + 420
    px, py = to_frame_pixels([0.5], [0.5], "pad", 1920, 1080)
    assert py[0] == 0.5 * 1920 - 420
    px, py = to_frame_pixels([0.5], [0.5], "pad", 640, 480)
    assert py[0] == 0.5 * 640 - 80


def test_landmark_features_layout():
    xs, ys, zs = np.full(21, 0.25), np.full(21, 0.5), np.linspace(-0.1, 0.1, 21)
    features = landmark_features(xs, ys, zs, "plain", 1920, 1080, object_center=(1000, 600))
    assert len(features) == 63
    # mirrored x = 1920 - 0.25 * 1920 = 1440, y = 540
    assert features[0] == 1000 - 1440
    assert features[1] == 600 - 540
    assert features[2] == zs[0]


@pytest.mark.parametrize(
    "hand_box, expected",
    [
        ((100, 100, 200, 200), (1920, 1080)),    # hand top-left  -> corner bottom-right
        ((1600, 800, 200, 200), (0, 0)),         # hand bottom-right -> corner top-left
        ((1600, 100, 200, 200), (0, 1080)),
    ],
)
def test_phantom_object_is_opposite_corner(hand_box, expected):
    assert phantom_object_center(hand_box, 1920, 1080) == expected
