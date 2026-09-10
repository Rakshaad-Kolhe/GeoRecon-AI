import numpy as np

from georecon.util.ply import read_ply, write_ply


def test_ply_roundtrip_with_extra_props(tmp_path):
    rng = np.random.default_rng(0)
    n = 50
    xyz = rng.normal(size=(n, 3)) * 10.0
    rgb = rng.integers(0, 256, (n, 3), dtype=np.uint8)
    err = rng.random(n).astype(np.float32)
    track = rng.integers(2, 40, n).astype(np.float32)

    path = tmp_path / "cloud.ply"
    write_ply(path, xyz, {"red": rgb[:, 0], "green": rgb[:, 1], "blue": rgb[:, 2],
                          "error": err, "track_len": track})

    got = read_ply(path)
    assert set(got) == {"x", "y", "z", "red", "green", "blue", "error", "track_len"}
    assert np.allclose(got["x"], xyz[:, 0].astype(np.float32))
    assert np.allclose(got["z"], xyz[:, 2].astype(np.float32))
    assert np.array_equal(got["green"], rgb[:, 1])
    assert got["red"].dtype == np.uint8
    assert np.allclose(got["error"], err)
    assert np.array_equal(got["track_len"], track)


def test_ply_empty(tmp_path):
    path = tmp_path / "empty.ply"
    write_ply(path, np.zeros((0, 3)), {"error": np.zeros((0,), np.float32)})
    got = read_ply(path)
    assert got["x"].shape == (0,) and got["error"].shape == (0,)
