import pycolmap

from georecon.config import settings
from georecon.util import colmap_cli
from georecon.util.colmap_cli import build_args, parse_help


def test_build_args_types_and_spaces():
    assert build_args({
        "SiftExtraction.max_num_features": 8192,
        "flag_on": True, "flag_off": False, "skip": None,
        "path": "C:/a b/c.db",
    }) == [
        "--SiftExtraction.max_num_features", "8192",
        "--flag_on", "1", "--flag_off", "0",
        "--path", "C:/a b/c.db",
    ]


def test_parse_help_with_cuda():
    txt = ("COLMAP 3.11.1 -- Structure-from-Motion and Multi-View Stereo\n"
           "(Commit ... compiled with CUDA)\nOptions:")
    assert parse_help(txt) == {"version": "3.11.1", "cuda": True}


def test_parse_help_cpu_only():
    assert parse_help("COLMAP 3.9 -- SfM\nOptions:") == {"version": "3.9", "cuda": False}


def test_parse_help_no_version():
    p = parse_help("random text mentioning with CUDA support")
    assert p == {"version": None, "cuda": True}


def test_probe_unset_bin(monkeypatch):
    monkeypatch.setattr(settings, "colmap_bin", "")
    colmap_cli.reset_probe_cache()
    assert colmap_cli.probe() == {"available": False, "version": None, "cuda": False}


class _Log:
    def __init__(self):
        self.msgs = []

    def warning(self, msg, *a):
        self.msgs.append(msg % a if a else msg)


def test_check_version_compat_warns_on_mismatch(monkeypatch):
    monkeypatch.setattr(colmap_cli, "probe",
                        lambda: {"available": True, "version": "3.8.0", "cuda": True})
    log = _Log()
    colmap_cli.check_version_compat(log)
    assert log.msgs and "incompatible" in log.msgs[0]


def test_check_version_compat_silent_on_match(monkeypatch):
    monkeypatch.setattr(colmap_cli, "probe",
                        lambda: {"available": True, "version": pycolmap.__version__, "cuda": True})
    log = _Log()
    colmap_cli.check_version_compat(log)
    assert not log.msgs
