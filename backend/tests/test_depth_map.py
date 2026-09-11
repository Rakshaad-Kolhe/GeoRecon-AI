import numpy as np

from georecon.util.depth_map import read_array, write_array


def test_depth_map_round_trip_single_channel(tmp_path):
    rng = np.random.default_rng(0)
    orig = rng.uniform(0, 50, size=(37, 53)).astype(np.float32)   # (h, w)
    p = tmp_path / "d.geometric.bin"
    write_array(p, orig)
    back = read_array(p)
    assert back.shape == orig.shape
    assert np.allclose(orig, back)


def test_depth_map_round_trip_multi_channel(tmp_path):
    rng = np.random.default_rng(1)
    orig = rng.uniform(-1, 1, size=(20, 30, 3)).astype(np.float32)  # normal map
    p = tmp_path / "n.geometric.bin"
    write_array(p, orig)
    back = read_array(p)
    assert back.shape == orig.shape
    assert np.allclose(orig, back)


def test_depth_map_parses_hand_built_header(tmp_path):
    """Build the file by hand (not via write_array) to pin down COLMAP's exact
    on-disk layout: ASCII 'w&h&c&' then float32 data in column-major (w,h,c)
    order, independent of our own writer."""
    w, h, c = 4, 3, 1
    # column-major (Fortran) flatten of a simple (w,h) grid so each pixel's
    # value equals its (row, col) grid position for an unambiguous check.
    grid_whc = np.zeros((w, h, c), np.float32)
    for x in range(w):
        for y in range(h):
            grid_whc[x, y, 0] = y * 10 + x     # row*10 + col
    flat = grid_whc.reshape(-1, order="F")
    header = f"{w}&{h}&{c}&".encode("ascii")
    p = tmp_path / "hand.bin"
    p.write_bytes(header + flat.astype(np.float32).tobytes())

    arr = read_array(p)
    assert arr.shape == (h, w)
    for y in range(h):
        for x in range(w):
            assert arr[y, x] == y * 10 + x


def test_depth_map_truncated_raises(tmp_path):
    p = tmp_path / "bad.bin"
    p.write_bytes(b"4&4&1&" + np.zeros(3, np.float32).tobytes())   # needs 16
    try:
        read_array(p)
        assert False, "expected ValueError"
    except ValueError:
        pass
