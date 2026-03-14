import json
import sys
from pathlib import Path

import numpy as np

from postproc_common.kurtio import layout_from_metadata, write_mask
from rfiperf.cli import main


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


def test_rfiperf_obs_dir_json_summary(tmp_path, monkeypatch, capsys):
    print("\nchecking CLI obs-directory mode with JSON summary", file=sys.stderr)

    obs = tmp_path / "obs"
    obs.mkdir()

    d1 = obs / "LoA.C0352"
    d2 = obs / "LoA.C0544"
    d1.mkdir()
    d2.mkdir()

    make_status(d1 / "status_dump.json", 352)
    make_status(d2 / "status_dump.json", 544)

    meta = {
        "schan": 352,
        "nants": 2,
        "nchan": 384,
        "npol": 2,
        "piperblk": 8192,
    }
    layout = layout_from_metadata(meta, kbsize=256)

    mask = np.zeros((2, 384, 64, 2), dtype=np.uint8)
    mask[0, 0, 0, 0] = 1
    mask[0, 1, 0, 0] = 1
    mask[1, 2, 0, 0] = 1

    mask_path = obs / "LoA_spliced.kurtosismask.bin"
    write_mask(mask_path, mask, layout)

    monkeypatch.setattr(
        "sys.argv",
        [
            "rfiperf",
            "kurtosis",
            str(obs),
            "--lo",
            "A",
            "--pol",
            "x",
            "--json",
            "summary",
        ],
    )

    main()

    captured = capsys.readouterr()
    print(captured.out, file=sys.stderr)

    out = json.loads(captured.out)

    assert out["lo"] == "LoA"
    assert out["pol"] == "x"
    assert out["ant"] is None
    assert out["nants_used"] == 2
    assert out["nchans"] == 384
    assert out["ntime"] == 64
    assert out["zapped_cells"] == 3
    assert out["total_cells"] == 2 * 384 * 64
    assert out["zap_fraction"] == 3 / (2 * 384 * 64)


def test_rfiperf_single_file_json_summary(tmp_path, monkeypatch, capsys):
    print("\nchecking CLI single-file mode with JSON summary", file=sys.stderr)

    status_path = tmp_path / "status_dump.json"
    make_status(status_path, 352)

    meta = {
        "schan": 352,
        "nants": 2,
        "nchan": 384,
        "npol": 2,
        "piperblk": 8192,
    }
    layout = layout_from_metadata(meta, kbsize=256)

    mask = np.zeros((2, 384, 64, 2), dtype=np.uint8)
    mask[1, 10, 5, 0] = 1

    mask_path = tmp_path / "LoA_spliced.kurtosismask.bin"
    write_mask(mask_path, mask, layout)

    monkeypatch.setattr(
        "sys.argv",
        [
            "rfiperf",
            "kurtosis",
            str(mask_path),
            "--status",
            str(status_path),
            "--nchan",
            "384",
            "--pol",
            "x",
            "--json",
            "summary",
        ],
    )

    main()

    captured = capsys.readouterr()
    print(captured.out, file=sys.stderr)

    out = json.loads(captured.out)

    assert out["lo"] == "LoA"
    assert out["pol"] == "x"
    assert out["zapped_cells"] == 1
    assert out["total_cells"] == 2 * 384 * 64
    assert out["zap_fraction"] == 1 / (2 * 384 * 64)


def test_rfiperf_single_file_plot_output(tmp_path, monkeypatch, capsys):
    print("\nchecking CLI single-file mode with saved plot", file=sys.stderr)

    status_path = tmp_path / "status_dump.json"
    make_status(status_path, 352)

    meta = {
        "schan": 352,
        "nants": 2,
        "nchan": 384,
        "npol": 2,
        "piperblk": 8192,
    }
    layout = layout_from_metadata(meta, kbsize=256)

    mask = np.zeros((2, 384, 64, 2), dtype=np.uint8)
    mask[0, 0, 0, 0] = 1

    mask_path = tmp_path / "LoA_spliced.kurtosismask.bin"
    write_mask(mask_path, mask, layout)

    outdir = tmp_path / "plots"

    monkeypatch.setattr(
        "sys.argv",
        [
            "rfiperf",
            "kurtosis",
            str(mask_path),
            "--status",
            str(status_path),
            "--nchan",
            "384",
            "--pol",
            "x",
            "--plot",
            "freq",
            "--outdir",
            str(outdir),
        ],
    )

    main()

    captured = capsys.readouterr()
    print(captured.out, file=sys.stderr)

    plot_path = Path(captured.out.strip())
    assert plot_path.exists()
    assert plot_path.name == "LoA_polx_freq.png"


def test_rfiperf_bad_ant_index(tmp_path, monkeypatch):
    print("\nchecking CLI bad antenna index handling", file=sys.stderr)

    status_path = tmp_path / "status_dump.json"
    make_status(status_path, 352)

    meta = {
        "schan": 352,
        "nants": 2,
        "nchan": 384,
        "npol": 2,
        "piperblk": 8192,
    }
    layout = layout_from_metadata(meta, kbsize=256)

    mask = np.zeros((2, 384, 64, 2), dtype=np.uint8)
    mask_path = tmp_path / "LoA_spliced.kurtosismask.bin"
    write_mask(mask_path, mask, layout)

    monkeypatch.setattr(
        "sys.argv",
        [
            "rfiperf",
            "kurtosis",
            str(mask_path),
            "--status",
            str(status_path),
            "--nchan",
            "384",
            "--pol",
            "x",
            "--json",
            "summary",
            "--ant",
            "2",
        ],
    )

    try:
        main()
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        assert "Invalid antenna index" in str(e)
    else:
        raise AssertionError("Expected SystemExit for invalid antenna index")