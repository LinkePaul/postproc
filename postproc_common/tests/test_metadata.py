import json

import pytest

from postproc_common.metadata import (
    read_status,
    load_slice_metadata,
    load_all_metadata,
    check_metadata_compatible,
)


def make_status(path, **overrides):
    data = {
        "SCHAN": 352,
        "SUBBAND": "C0352",
        "NANTS": 28,
        "NCHAN": 192,
        "NPOL": 2,
        "PIPERBLK": 8192,
        "CHAN_BW": 0.5,
        "TBIN": 2e-06,
        "SOURCE": "J2022+5154",
        "OBSBW": 96,
        "OBSNCHAN": 5376,
    }
    data.update(overrides)
    path.write_text(json.dumps(data))
    return data


def test_read_status(tmp_path):
    path = tmp_path / "status_dump.json"
    raw = make_status(path)
    assert read_status(path) == raw


def test_load_slice_metadata(tmp_path):
    status_path = tmp_path / "status_dump.json"
    make_status(status_path)

    slice_info = {
        "path": tmp_path,
        "lo": "LoA",
        "schan": 352,
        "status_path": status_path,
    }

    meta = load_slice_metadata(slice_info)

    assert meta["lo"] == "LoA"
    assert meta["schan"] == 352
    assert meta["subband"] == "C0352"
    assert meta["nants"] == 28
    assert meta["nchan"] == 192
    assert meta["npol"] == 2
    assert meta["piperblk"] == 8192
    assert meta["chan_bw"] == 0.5
    assert meta["tbin"] == 2e-06
    assert meta["source"] == "J2022+5154"
    assert meta["obsbw"] == 96.0
    assert meta["obsnchan"] == 5376


def test_load_slice_metadata_missing_status(tmp_path):
    slice_info = {
        "path": tmp_path,
        "lo": "LoA",
        "schan": 352,
        "status_path": None,
    }

    with pytest.raises(ValueError):
        load_slice_metadata(slice_info)


def test_load_all_metadata(tmp_path):
    p1 = tmp_path / "s1.json"
    p2 = tmp_path / "s2.json"

    make_status(p1, SCHAN=352, SUBBAND="C0352")
    make_status(p2, SCHAN=544, SUBBAND="C0544")

    slice_list = [
        {"path": tmp_path / "LoA.C0352", "lo": "LoA", "schan": 352, "status_path": p1},
        {"path": tmp_path / "LoA.C0544", "lo": "LoA", "schan": 544, "status_path": p2},
    ]

    meta_list = load_all_metadata(slice_list)

    assert len(meta_list) == 2
    assert meta_list[0]["schan"] == 352
    assert meta_list[1]["schan"] == 544


def test_check_metadata_compatible_ok(tmp_path):
    meta_list = [
        {
            "path": tmp_path / "a",
            "lo": "LoA",
            "schan": 352,
            "nants": 28,
            "nchan": 192,
            "npol": 2,
            "piperblk": 8192,
            "chan_bw": 0.5,
            "tbin": 2e-06,
        },
        {
            "path": tmp_path / "b",
            "lo": "LoA",
            "schan": 544,
            "nants": 28,
            "nchan": 192,
            "npol": 2,
            "piperblk": 8192,
            "chan_bw": 0.5,
            "tbin": 2e-06,
        },
    ]

    assert check_metadata_compatible(meta_list) is True


def test_check_metadata_compatible_mismatch(tmp_path):
    meta_list = [
        {
            "path": tmp_path / "a",
            "lo": "LoA",
            "schan": 352,
            "nants": 28,
            "nchan": 192,
            "npol": 2,
            "piperblk": 8192,
            "chan_bw": 0.5,
            "tbin": 2e-06,
        },
        {
            "path": tmp_path / "b",
            "lo": "LoA",
            "schan": 544,
            "nants": 27,
            "nchan": 192,
            "npol": 2,
            "piperblk": 8192,
            "chan_bw": 0.5,
            "tbin": 2e-06,
        },
    ]

    with pytest.raises(ValueError):
        check_metadata_compatible(meta_list)


def test_check_metadata_compatible_noncontiguous(tmp_path):
    meta_list = [
        {
            "path": tmp_path / "a",
            "lo": "LoA",
            "schan": 352,
            "nants": 28,
            "nchan": 192,
            "npol": 2,
            "piperblk": 8192,
            "chan_bw": 0.5,
            "tbin": 2e-06,
        },
        {
            "path": tmp_path / "b",
            "lo": "LoA",
            "schan": 736,
            "nants": 28,
            "nchan": 192,
            "npol": 2,
            "piperblk": 8192,
            "chan_bw": 0.5,
            "tbin": 2e-06,
        },
    ]

    with pytest.raises(ValueError):
        check_metadata_compatible(meta_list)