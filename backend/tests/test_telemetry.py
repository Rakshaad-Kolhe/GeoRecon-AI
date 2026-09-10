import numpy as np
import pytest

from georecon.ingest.telemetry import interpolate, load_telemetry, telemetry_stats

SRT_A = """1
00:00:00,000 --> 00:00:01,000
[latitude: 18.520430] [longitude: 73.856743] [rel_alt: 50.000 abs_alt: 612.300]

2
00:00:01,000 --> 00:00:02,000
[latitude: 18.520500] [longitude: 73.856800] [rel_alt: 51.000 abs_alt: 613.300]
"""

SRT_A_FONT = """1
00:00:00,000 --> 00:00:01,000
<font size="28">SrtCnt : 1, DiffTime : 33ms
[latitude : 18.520430] [longitude : 73.856743] [rel_alt : 50.000 abs_alt : 612.300]</font>

2
00:00:02,000 --> 00:00:03,000
<font size="28">[latitude:18.520900] [longitude:73.857000] [altitude:55.0]</font>
"""

SRT_B = """1
00:00:00,000 --> 00:00:01,000
HOME(73.8560,18.5200) GPS(73.8567,18.5204,19) BAROMETER:50.2

2
00:00:01,000 --> 00:00:02,000
GPS(73.8570,18.5210,19) BAROMETER:51.0
"""


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_srt_format_a(tmp_path):
    df = load_telemetry(_write(tmp_path, "a.srt", SRT_A))
    assert list(df["t"]) == [0.0, 1.0]
    assert df["lat"].iloc[0] == pytest.approx(18.520430)
    assert df["lon"].iloc[0] == pytest.approx(73.856743)
    # abs_alt wins over rel_alt
    assert df["alt"].iloc[0] == pytest.approx(612.300)


def test_srt_format_b(tmp_path):
    df = load_telemetry(_write(tmp_path, "b.srt", SRT_B))
    assert df["lon"].iloc[0] == pytest.approx(73.8567)
    assert df["lat"].iloc[0] == pytest.approx(18.5204)
    assert df["alt"].iloc[0] == pytest.approx(19.0)
    assert df["rel_alt"].iloc[0] == pytest.approx(50.2)


def test_srt_font_wrapped_a(tmp_path):
    df = load_telemetry(_write(tmp_path, "font.srt", SRT_A_FONT))
    assert len(df) == 2
    assert df["lat"].iloc[0] == pytest.approx(18.520430)
    assert df["alt"].iloc[0] == pytest.approx(612.300)
    # second block falls back to plain [altitude: x]
    assert df["alt"].iloc[1] == pytest.approx(55.0)


def test_csv_plain_headers(tmp_path):
    csv = "time,latitude,longitude,altitude\n0,18.52,73.85,600\n1,18.53,73.86,601\n"
    df = load_telemetry(_write(tmp_path, "p.csv", csv))
    assert list(df["t"]) == [0.0, 1.0]
    assert df["alt"].iloc[1] == pytest.approx(601.0)


def test_csv_millisecond_and_feet(tmp_path):
    csv = (
        "time(millisecond),lat,lng,altitude_above_sealevel(feet)\n"
        "0,18.52,73.85,3280.84\n"
        "500,18.53,73.86,3300.0\n"
    )
    df = load_telemetry(_write(tmp_path, "ms.csv", csv))
    assert list(df["t"]) == [0.0, 0.5]
    assert df["alt"].iloc[0] == pytest.approx(3280.84 * 0.3048)


def test_bad_file_raises_valueerror(tmp_path):
    with pytest.raises(ValueError):
        load_telemetry(_write(tmp_path, "junk.srt", "not a subtitle file at all"))
    with pytest.raises(ValueError):
        load_telemetry(_write(tmp_path, "junk.txt", "whatever"))


def test_interpolate_edge_clamp_and_mask(tmp_path):
    df = load_telemetry(_write(tmp_path, "a.srt", SRT_A))
    q = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    lat, lon, alt, oob = interpolate(df, q)
    assert list(oob) == [True, False, False, False, True]
    # clamped to first sample before t=0
    assert lat[0] == pytest.approx(df["lat"].iloc[0])
    # linear midpoint
    assert lat[2] == pytest.approx((df["lat"].iloc[0] + df["lat"].iloc[1]) / 2)
    assert alt[0] == pytest.approx(df["alt"].iloc[0])


def test_telemetry_stats(tmp_path):
    df = load_telemetry(_write(tmp_path, "a.srt", SRT_A))
    st = telemetry_stats(df)
    assert st == {"rows": 2, "hz": pytest.approx(1.0), "t_start": 0.0, "t_end": 1.0}
