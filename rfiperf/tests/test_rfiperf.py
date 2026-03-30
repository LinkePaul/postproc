import json
import sys
from pathlib import Path

import numpy as np
import pytest

from postproc_common.kurtio import layout_from_metadata, write_mask

from rfiperf.cli import main


def make_status(path: Path, schan: int, *, obsfreq: float) -> None:
    data = {
        "SCHAN": schan,
        "SUBBAND": f"C{schan:04d}",
        "NANTS": 28,
        "NCHAN": 192,
        "PKTNCHAN": 96,
        "NPOL": 2,
        "PIPERBLK": 8192,
        "CHAN_BW": 0.5,
        "TBIN": 2e-06,
        "PKTSTART": 0,
        "PKTSTOP": 150_000_000,  # 300 s
        "SOURCE": "J2022+5154",
        "OBSBW": 96,
        "OBSNCHAN": 5376,
        "OBSFREQ": obsfreq,
        "ANTNMS00": "1bA,1cA,1dA,1eA,1fA,1gA,1hA,4jA,1kA,2aA,2bA,2cA,2dA,2eA,2fA,4gA,2hA",
        "ANTNMS01": "2jA,2kA,2lA,2mA,3cA,3dA,5bA,5cA,3lA,4eA,5eA",
    }
    path.write_text(json.dumps(data))


def make_obs(tmp_path: Path):
    obs = tmp_path / "obs"
    obs.mkdir()

    d1 = obs / "LoA.C0352"
    d2 = obs / "LoA.C0544"
    d1.mkdir()
    d2.mkdir()

    make_status(d1 / "status_dump.json", 352, obsfreq=1211.75)
    make_status(d2 / "status_dump.json", 544, obsfreq=1307.75)

    meta = {
        "schan": 0,
        "nants": 28,
        "nchan": 384,
        "npol": 2,
        "piperblk": 8192,
    }
    layout = layout_from_metadata(meta, kbsize=256)
    mask = np.zeros((28, 384, 64, 2), dtype=np.uint8)

    mask[0, 0, 0, 0] = 1
    mask[0, 1, 0, 0] = 1
    mask[1, 2, 0, 0] = 1
    mask[15, 10:20, 5:8, 0] = 1
    mask[15, 200:210, 40:43, 0] = 1
    mask[1, 50:55, 10:12, 1] = 1

    mask_path = obs / "LoA_spliced.kurtosismask.bin"
    write_mask(mask_path, mask, layout)

    return obs, mask_path


def test_rfiperf_obs_dir_json_summary(tmp_path, monkeypatch, capsys):
    print("\nchecking CLI obs-directory mode with JSON summary", file=sys.stderr)
    obs, _ = make_obs(tmp_path)

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
    assert out["nants_used"] == 28
    assert out["nchans"] == 384
    assert out["ntime"] == 64
    assert out["zapped_cells"] == int(2 + 1 + 10 * 3 + 10 * 3)
    assert out["total_cells"] == 28 * 384 * 64


def test_rfiperf_spliced_file_json_summary_autodiscovery(tmp_path, monkeypatch, capsys):
    print("\nchecking CLI spliced-file autodiscovery with JSON summary", file=sys.stderr)
    _, mask_path = make_obs(tmp_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "rfiperf",
            "kurtosis",
            str(mask_path),
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
    assert out["nants_used"] == 28
    assert out["nchans"] == 384
    assert out["ntime"] == 64


def test_rfiperf_named_ant_json_summary_returns_name(tmp_path, monkeypatch, capsys):
    print("\nchecking CLI named antenna selector in JSON summary", file=sys.stderr)
    _, mask_path = make_obs(tmp_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "rfiperf",
            "kurtosis",
            str(mask_path),
            "--pol",
            "x",
            "--json",
            "summary",
            "--ant",
            "4g",
        ],
    )
    main()
    captured = capsys.readouterr()
    print(captured.out, file=sys.stderr)

    out = json.loads(captured.out)
    assert out["ant"] == "4g"
    assert out["nants_used"] == 1
    assert out["nchans"] == 384
    assert out["ntime"] == 64
    assert out["zapped_cells"] == int(10 * 3 + 10 * 3)


def test_rfiperf_freq_plot_output(tmp_path, monkeypatch, capsys):
    print("\nchecking CLI single-file mode with saved freq plot", file=sys.stderr)
    _, mask_path = make_obs(tmp_path)
    outdir = tmp_path / "plots"

    monkeypatch.setattr(
        "sys.argv",
        [
            "rfiperf",
            "kurtosis",
            str(mask_path),
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


def test_rfiperf_time_plot_named_ant_output(tmp_path, monkeypatch, capsys):
    print("\nchecking CLI named antenna selector for time plot", file=sys.stderr)
    _, mask_path = make_obs(tmp_path)
    outdir = tmp_path / "plots"

    monkeypatch.setattr(
        "sys.argv",
        [
            "rfiperf",
            "kurtosis",
            str(mask_path),
            "--pol",
            "x",
            "--plot",
            "time",
            "--ant",
            "4g",
            "--outdir",
            str(outdir),
        ],
    )
    main()
    captured = capsys.readouterr()
    print(captured.out, file=sys.stderr)

    plot_path = Path(captured.out.strip())
    assert plot_path.exists()
    assert plot_path.name == "LoA_polx_ant4g_time.png"


def test_rfiperf_bad_ant_name_raises(tmp_path, monkeypatch):
    print("\nchecking CLI bad antenna name handling", file=sys.stderr)
    _, mask_path = make_obs(tmp_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "rfiperf",
            "kurtosis",
            str(mask_path),
            "--pol",
            "x",
            "--json",
            "summary",
            "--ant",
            "9z",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert "Unknown antenna selector" in str(excinfo.value)


def test_rfiperf_bad_numeric_ant_index_raises(tmp_path, monkeypatch):
    print("\nchecking CLI bad numeric antenna index handling", file=sys.stderr)
    _, mask_path = make_obs(tmp_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "rfiperf",
            "kurtosis",
            str(mask_path),
            "--pol",
            "x",
            "--json",
            "summary",
            "--ant",
            "99",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert "Invalid antenna index" in str(excinfo.value)


def test_rfiperf_waterfall_named_antenna_selector_uses_ant_label_in_filename(
    tmp_path, monkeypatch, capsys
):
    print("\nchecking CLI named antenna selector for waterfall filename", file=sys.stderr)
    _, mask_path = make_obs(tmp_path)
    outdir = tmp_path / "plots"

    monkeypatch.setattr(
        "sys.argv",
        [
            "rfiperf",
            "kurtosis",
            str(mask_path),
            "--pol",
            "x",
            "--plot",
            "waterfall",
            "--ant",
            "4g",
            "--outdir",
            str(outdir),
        ],
    )
    main()
    captured = capsys.readouterr()
    print(captured.out, file=sys.stderr)

    plot_path = Path(captured.out.strip())
    assert plot_path.exists()
    assert plot_path.name == "LoA_polx_ant4g_waterfall.png"


def test_rfiperf_waterfall_grid_autodiscovery_from_spliced_mask(
    tmp_path, monkeypatch, capsys
):
    print("\nchecking CLI all-antenna waterfall grid from spliced mask autodiscovery", file=sys.stderr)
    _, mask_path = make_obs(tmp_path)
    outdir = tmp_path / "plots"

    monkeypatch.setattr(
        "sys.argv",
        [
            "rfiperf",
            "kurtosis",
            str(mask_path),
            "--pol",
            "x",
            "--plot",
            "waterfall",
            "--tend",
            "16",
            "--outdir",
            str(outdir),
        ],
    )
    main()
    captured = capsys.readouterr()
    print(captured.out, file=sys.stderr)

    plot_path = Path(captured.out.strip())
    assert plot_path.exists()
    assert plot_path.name == "LoA_polx_t0-16_waterfall.png"


def test_rfiperf_obs_dir_waterfall_named_ant(tmp_path, monkeypatch, capsys):
    print("\nchecking CLI obs-directory mode with named antenna waterfall", file=sys.stderr)
    obs, _ = make_obs(tmp_path)
    outdir = tmp_path / "plots"

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
            "--plot",
            "waterfall",
            "--ant",
            "4g",
            "--tend",
            "16",
            "--outdir",
            str(outdir),
        ],
    )
    main()
    captured = capsys.readouterr()
    print(captured.out, file=sys.stderr)

    plot_path = Path(captured.out.strip())
    assert plot_path.exists()
    assert plot_path.name == "LoA_polx_ant4g_t0-16_waterfall.png"
    