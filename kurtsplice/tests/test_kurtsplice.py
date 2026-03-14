import json
import sys
from pathlib import Path

import numpy as np

from postproc_common.kurtio import layout_from_metadata, write_mask, read_mask
from kurtsplice.cli import main


def make_status(path, schan):
    data = {
        "SCHAN": schan,
        "SUBBAND": f"C{schan:04d}",
        "NANTS": 2,
        "NCHAN": 192,
        "NPOL": 2,
        "PIPERBLK": 8192,
        "CHAN_BW": 0.5,
        "TBIN": 2e-06,
        "SOURCE": "J2022+5154",
        "OBSBW": 96,
        "OBSNCHAN": 384,
    }
    path.write_text(json.dumps(data))


def make_meta(schan, nchan=192):
    return {
        "path": "dummy",
        "lo": "LoA",
        "schan": schan,
        "nants": 2,
        "nchan": nchan,
        "npol": 2,
        "piperblk": 8192,
        "chan_bw": 0.5,
        "tbin": 2e-06,
    }


def test_kurtsplice_default_output(tmp_path, monkeypatch, capsys):
    print("\nchecking kurtsplice default output path", file=sys.stderr)

    obs = tmp_path / "obs"
    obs.mkdir()

    d1 = obs / "LoA.C0352"
    d2 = obs / "LoA.C0544"
    d1.mkdir()
    d2.mkdir()

    make_status(d1 / "status_dump.json", 352)
    make_status(d2 / "status_dump.json", 544)

    meta1 = make_meta(352)
    meta2 = make_meta(544)

    layout1 = layout_from_metadata(meta1, kbsize=256)
    layout2 = layout_from_metadata(meta2, kbsize=256)

    mask1 = np.zeros((2, 192, 64, 2), dtype=np.uint8)
    mask2 = np.ones((2, 192, 64, 2), dtype=np.uint8)

    write_mask(d1 / "one.kurtosismask.bin", mask1, layout1)
    write_mask(d2 / "two.kurtosismask.bin", mask2, layout2)

    monkeypatch.setattr(
        "sys.argv",
        [
            "kurtsplice",
            str(obs),
            "--lo",
            "A",
        ],
    )

    main()

    captured = capsys.readouterr()
    print(captured.out, file=sys.stderr)

    out_path = Path(captured.out.strip())
    assert out_path == obs / "LoA_spliced.kurtosismask.bin"
    assert out_path.exists()

    out_meta = make_meta(352, nchan=384)
    out_layout = layout_from_metadata(out_meta, kbsize=256)
    out_mask = read_mask(out_path, out_layout)

    assert out_mask.shape == (2, 384, 64, 2)
    assert np.all(out_mask[:, :192, :, :] == 0)
    assert np.all(out_mask[:, 192:, :, :] == 1)


def test_kurtsplice_custom_output(tmp_path, monkeypatch, capsys):
    print("\nchecking kurtsplice custom output path", file=sys.stderr)

    obs = tmp_path / "obs"
    obs.mkdir()

    d1 = obs / "LoA.C0352"
    d2 = obs / "LoA.C0544"
    d1.mkdir()
    d2.mkdir()

    make_status(d1 / "status_dump.json", 352)
    make_status(d2 / "status_dump.json", 544)

    meta1 = make_meta(352)
    meta2 = make_meta(544)

    layout1 = layout_from_metadata(meta1, kbsize=256)
    layout2 = layout_from_metadata(meta2, kbsize=256)

    mask1 = np.zeros((2, 192, 64, 2), dtype=np.uint8)
    mask2 = np.ones((2, 192, 64, 2), dtype=np.uint8)

    write_mask(d1 / "one.kurtosismask.bin", mask1, layout1)
    write_mask(d2 / "two.kurtosismask.bin", mask2, layout2)

    custom_out = tmp_path / "custom_out.bin"

    monkeypatch.setattr(
        "sys.argv",
        [
            "kurtsplice",
            str(obs),
            "--lo",
            "A",
            "--out",
            str(custom_out),
        ],
    )

    main()

    captured = capsys.readouterr()
    print(captured.out, file=sys.stderr)

    out_path = Path(captured.out.strip())
    assert out_path == custom_out
    assert out_path.exists()


def test_kurtsplice_missing_mask_fails(tmp_path, monkeypatch):
    print("\nchecking kurtsplice missing mask handling", file=sys.stderr)

    obs = tmp_path / "obs"
    obs.mkdir()

    d1 = obs / "LoA.C0352"
    d2 = obs / "LoA.C0544"
    d1.mkdir()
    d2.mkdir()

    make_status(d1 / "status_dump.json", 352)
    make_status(d2 / "status_dump.json", 544)

    meta1 = make_meta(352)
    layout1 = layout_from_metadata(meta1, kbsize=256)
    mask1 = np.zeros((2, 192, 64, 2), dtype=np.uint8)

    write_mask(d1 / "one.kurtosismask.bin", mask1, layout1)
    # d2 intentionally missing mask

    monkeypatch.setattr(
        "sys.argv",
        [
            "kurtsplice",
            str(obs),
            "--lo",
            "A",
        ],
    )

    try:
        main()
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        assert "Missing kurtosismask.bin" in str(e)
    else:
        raise AssertionError("Expected SystemExit for missing mask file")


def test_kurtsplice_missing_lo_fails(tmp_path, monkeypatch):
    print("\nchecking kurtsplice bad LO handling", file=sys.stderr)

    obs = tmp_path / "obs"
    obs.mkdir()

    monkeypatch.setattr(
        "sys.argv",
        [
            "kurtsplice",
            str(obs),
            "--lo",
            "A",
        ],
    )

    try:
        main()
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        assert "No slice directories found" in str(e)
    else:
        raise AssertionError("Expected SystemExit for missing slice directories")