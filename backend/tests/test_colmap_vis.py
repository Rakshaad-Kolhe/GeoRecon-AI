from georecon.util.colmap_vis import read_vis, write_vis


def test_vis_roundtrip(tmp_path):
    p = tmp_path / "fused.ply.vis"
    write_vis(p, [[0, 1, 2], [3], [], [1, 4, 5, 6]])
    counts, status = read_vis(p)
    assert status == "ok"
    assert list(counts) == [3, 1, 0, 4]


def test_vis_truncated_ids(tmp_path):
    p = tmp_path / "fused.ply.vis"
    write_vis(p, [[0, 1], [2, 3, 4]])
    p.write_bytes(p.read_bytes()[:-4])          # drop one id uint32
    counts, status = read_vis(p)
    assert counts is None and "truncat" in status.lower()


def test_vis_trailing_bytes(tmp_path):
    p = tmp_path / "fused.ply.vis"
    write_vis(p, [[9]])
    p.write_bytes(p.read_bytes() + b"\x00\x00\x00")
    counts, status = read_vis(p)
    assert counts is None and "trailing" in status


def test_vis_too_small(tmp_path):
    p = tmp_path / "fused.ply.vis"
    p.write_bytes(b"\x01\x02")
    counts, status = read_vis(p)
    assert counts is None
