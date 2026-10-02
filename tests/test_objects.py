import numpy as np

from smartrehab.objects import MIN_OVERLAP, select_object, yolo_to_pixel_box


def thesis_selection(hand, detections, width, height):
    """Selection loop as written in the thesis' 2_yolo_test_mod.py."""
    hand_min_x, hand_min_y = hand[0], hand[1]
    hand_max_x, hand_max_y = hand_min_x + hand[2], hand_min_y + hand[3]
    hand_center_x, hand_center_y = hand[0] + hand[2] // 2, hand[1] + hand[3] // 2

    best_iou, closest_distance = 5000, width * height
    chosen, closest = [0, 0, 0, 0], [0, 0, 0, 0]
    for min_x, min_y, w, h in detections:
        max_x, max_y = min_x + w, min_y + h
        center_x, center_y = (min_x + max_x) // 2, (min_y + max_y) // 2
        inter = max(0, min(hand_max_x, max_x) - max(hand_min_x, min_x) + 1) * max(
            0, min(hand_max_y, max_y) - max(hand_min_y, min_y) + 1)
        distance = np.sqrt(np.power(center_x - hand_center_x, 2) + np.power(center_y - hand_center_y, 2))
        if inter > best_iou:
            best_iou, chosen = inter, [min_x, min_y, w, h]
        if distance < closest_distance:
            closest_distance, closest = distance, [min_x, min_y, w, h]
    return closest if best_iou == 5000 else chosen


def test_no_detection_gives_empty_box():
    assert select_object((10, 10, 50, 50), [], 1920, 1080) == [0, 0, 0, 0]


def test_largest_overlap_wins_over_nearest():
    hand = (500, 500, 400, 400)
    overlapping = (450, 450, 300, 300)          # large overlap, centre a bit off
    nearest_but_tiny = (690, 690, 20, 20)        # centre exactly on the hand centre, overlap < threshold
    assert select_object(hand, [nearest_but_tiny, overlapping], 1920, 1080) == list(overlapping)


def test_falls_back_to_nearest_when_overlap_is_small():
    hand = (100, 100, 50, 50)
    near, far = (200, 100, 40, 40), (1500, 800, 40, 40)
    assert select_object(hand, [far, near], 1920, 1080) == list(near)


def test_matches_thesis_loop_on_random_scenes():
    rng = np.random.default_rng(11)
    for _ in range(300):
        hand = (int(rng.integers(0, 1500)), int(rng.integers(0, 800)), int(rng.integers(20, 400)), int(rng.integers(20, 400)))
        detections = [
            (int(rng.integers(0, 1800)), int(rng.integers(0, 1000)), int(rng.integers(10, 500)), int(rng.integers(10, 500)))
            for _ in range(int(rng.integers(0, 5)))
        ]
        assert select_object(hand, detections, 1920, 1080) == thesis_selection(hand, detections, 1920, 1080)


def test_overlap_threshold_constant():
    assert MIN_OVERLAP == 5000


def test_yolo_box_conversion_and_clipping():
    # centre box 0.5/0.5 of size 0.2 x 0.4 on a 1000x500 frame
    assert yolo_to_pixel_box(0.5, 0.5, 0.2, 0.4, 1000, 500) == (400, 150, 200, 200)
    # a box overhanging the left edge is clipped, which also shrinks its width
    min_x, _, w, _ = yolo_to_pixel_box(0.05, 0.5, 0.2, 0.4, 1000, 500)
    assert min_x == 0 and w == 150
