import pytest

from postproc_common.discovery import (
    normalize_lo,
    parse_slice_dir_name,
    pick_optional,
    discover_slices,
)


def test_normalize_lo():
    assert normalize_lo(None) is None
    assert normalize_lo("") is None
    assert normalize_lo("A") == "LoA"
    assert normalize_lo("LoA") == "LoA"
    assert normalize_lo("b") == "LoB"


def test_parse_slice_dir_name():
    assert parse_slice_dir_name("LoA.C0928") == ("LoA", 928)
    assert parse_slice_dir_name("LoB.C0352") == ("LoB", 352)
    assert parse_slice_dir_name("junk") is None
    assert parse_slice_dir_name("LoA") is None
    assert parse_slice_dir_name("LoAA.C0928") is None
    assert parse_slice_dir_name("LoA.X0928") is None
    assert parse_slice_dir_name("LoA.Cabcd") is None


def test_pick_optional(tmp_path):
    d = tmp_path / "x"
    d.mkdir()

    assert pick_optional(d, "*.fil") is None

    f1 = d / "a.fil"
    f1.write_bytes(b"abc")
    assert pick_optional(d, "*.fil") == f1

    f2 = d / "b.fil"
    f2.write_bytes(b"def")

    with pytest.raises(ValueError):
        pick_optional(d, "*.fil")


def test_discover_slices(tmp_path):
    obs = tmp_path / "obs"
    obs.mkdir()

    a1 = obs / "LoA.C0352"
    a2 = obs / "LoA.C0544"
    b1 = obs / "LoB.C0352"
    junk = obs / "random_dir"

    a1.mkdir()
    a2.mkdir()
    b1.mkdir()
    junk.mkdir()

    (a1 / "one.fil").write_bytes(b"x")
    (a1 / "one.kurtosismask.bin").write_bytes(b"y")
    (a1 / "status_dump.json").write_text("{}")

    (a2 / "two.fil").write_bytes(b"x")
    (b1 / "three.fil").write_bytes(b"x")

    slices = discover_slices(obs)

    assert len(slices) == 3
    assert slices[0]["lo"] == "LoA"
    assert slices[0]["schan"] == 352
    assert slices[1]["lo"] == "LoA"
    assert slices[1]["schan"] == 544
    assert slices[2]["lo"] == "LoB"
    assert slices[2]["schan"] == 352

    slices_a = discover_slices(obs, lo="A")

    assert len(slices_a) == 2
    assert slices_a[0]["fil_path"].name == "one.fil"
    assert slices_a[0]["kurt_path"].name == "one.kurtosismask.bin"
    assert slices_a[0]["status_path"].name == "status_dump.json"
    assert slices_a[1]["fil_path"].name == "two.fil"
    assert slices_a[1]["kurt_path"] is None
    assert slices_a[1]["status_path"] is None
