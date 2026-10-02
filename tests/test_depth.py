import struct

import numpy as np

from smartrehab.depth import depth_difference, read_pfm


def write_pfm(path, array, scale=-1.0):
    """Write a grayscale PFM (little endian for negative scale). Rows are stored bottom to top."""
    height, width = array.shape
    with open(path, "wb") as f:
        f.write(b"Pf\n" + f"{width} {height}\n".encode() + f"{scale}\n".encode())
        f.write(struct.pack(f"<{width * height}f", *np.flipud(array).ravel()))


def test_read_pfm_round_trip(tmp_path):
    depth = np.arange(12, dtype=np.float32).reshape(3, 4)
    write_pfm(tmp_path / "d.pfm", depth)
    np.testing.assert_allclose(read_pfm(tmp_path / "d.pfm"), depth)


def test_depth_difference_is_object_minus_hand_median():
    depth = np.zeros((100, 200))
    depth[10:30, 10:30] = 5.0
    depth[50:70, 100:120] = 8.0
    assert depth_difference(depth, (10, 10, 20, 20), (100, 50, 20, 20), 200, 100) == 3.0


def test_phantom_object_uses_corner_depth():
    depth = np.ones((100, 200))
    depth[99, 199] = 7.0
    depth[10:30, 10:30] = 2.0
    # phantom object at the bottom-right corner (200, 100) -> moved one pixel inside the frame
    assert depth_difference(depth, (10, 10, 20, 20), (200, 100, 0, 0), 200, 100) == 5.0


def test_empty_hand_box_gives_nan():
    depth = np.ones((10, 10))
    assert np.isnan(depth_difference(depth, (5, 5, 0, 0), (1, 1, 2, 2), 10, 10))
