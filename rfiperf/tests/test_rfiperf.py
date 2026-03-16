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


def make_bestprof(path, candidate="test", dm=21.6, p_bary_ms=529.2172, chi=37.7, sigma=44.8):
    lines = [
        "# Input file       =  spliced_loa.fil",
        f"# Candidate        =  {candidate}",
        "# Telescope        =  Unknown",
        "# Epoch_topo       =  61021.742592592593",
        "# Epoch_bary (MJD) =  61021.742904164574",
        "# T_sample         =  6.4e-05",
        "# Data Folded      =  4608000",
        "# Data Avg         =  171610.592129413",
        "# Data StdDev      =  1198.88570955295",
        "# Profile Bins     =  8",
        "# Profile Avg      =  12350134591.7883",
        "# Profile StdDev   =  321694.793257615",
        f"# Reduced chi-sqr  =  {chi}",
        f"# Prob(Noise)      <  0   (~{sigma} sigma)",
        f"# Best DM          =  {dm}",
        "# P_topo (ms)      =  529.212564950708  +/- 0.00599",
        "# P'_topo (s/s)    =  1.42516507021293e-12 +/- 7.74e-08",
        "# P''_topo (s/s^2) =  0                 +/- 1.7e-09",
        f"# P_bary (ms)      =  {p_bary_ms}  +/- 0.00599",
        "# P'_bary (s/s)    =  -5.99298427815554e-19 +/- 7.74e-08",
        "# P''_bary (s/s^2) =  7.39166484884108e-15 +/- 1.7e-09",
        "######################################################",
        "   0  10",
        "   1  10",
        "   2  10",
        "   3  20",
        "   4  10",
        "   5  10",
        "   6  10",
        "   7  10",
    ]
    path.write_text("\n".join(lines) + "\n")


def test_rfiperf_snr_single_file_json_summary_e2e(tmp_path, monkeypatch, capsys):
    print("\nchecking CLI snr single-file JSON summary", file=sys.stderr)

    path = tmp_path / "a.bestprof"
    make_bestprof(path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "rfiperf",
            "snr",
            str(path),
            "--baseline",
            "median",
            "--json",
        ],
    )

    main()

    captured = capsys.readouterr()
    print(captured.out, file=sys.stderr)

    out = json.loads(captured.out)
    assert out["candidate"] == "test"
    assert out["best_dm"] == 21.6
    assert out["p_bary_ms"] == 529.2172
    assert out["baseline"] == "median"
    assert out["peak_bin"] == 3
    assert out["profile_snr"] > 0
    assert out["prob_noise"] == "<  0"
    assert out["presto_sigma"] == 44.8


def test_rfiperf_snr_compare_json_summary_e2e(tmp_path, monkeypatch, capsys):
    print("\nchecking CLI snr compare JSON summary", file=sys.stderr)

    p1 = tmp_path / "fil_61021_64160_4434448_J2022+5154_0001" / "spliced_loa_PSR_2022+5154.pfd.bestprof"
    p2 = tmp_path / "fil_61111_61234_189484436_J2022+5154_0001" / "spliced_loa_PSR_2022+5154.pfd.bestprof"
    p1.parent.mkdir(parents=True)
    p2.parent.mkdir(parents=True)

    make_bestprof(p1, candidate="PSR_2022+5154", chi=200.0, sigma=100.0)
    make_bestprof(p2, candidate="PSR_2022+5154", chi=50.0, sigma=40.0)

    monkeypatch.setattr(
        "sys.argv",
        [
            "rfiperf",
            "snr",
            str(p1),
            str(p2),
            "--baseline",
            "median",
            "--json",
        ],
    )

    main()

    captured = capsys.readouterr()
    print(captured.out, file=sys.stderr)

    out = json.loads(captured.out)
    assert out["mode"] == "compare"
    assert out["baseline"] == "median"
    assert out["n_files"] == 2
    assert len(out["files"]) == 2
    assert "profile_snr" in out
    assert "reduced_chi_sqr" in out
    assert "presto_sigma" in out


def test_rfiperf_snr_compare_overlay_plot_e2e(tmp_path, monkeypatch, capsys):
    print("\nchecking CLI snr compare overlay plot output", file=sys.stderr)

    p1 = tmp_path / "fil_61021_64160_4434448_J2022+5154_0001" / "spliced_loa_PSR_2022+5154.pfd.bestprof"
    p2 = tmp_path / "fil_61111_61234_189484436_J2022+5154_0001" / "spliced_loa_PSR_2022+5154.pfd.bestprof"
    p1.parent.mkdir(parents=True)
    p2.parent.mkdir(parents=True)

    make_bestprof(p1, candidate="PSR_2022+5154")
    make_bestprof(p2, candidate="PSR_2022+5154")

    monkeypatch.setattr(
        "sys.argv",
        [
            "rfiperf",
            "snr",
            str(p1),
            str(p2),
            "--baseline",
            "median",
            "--plot",
            "overlay",
            "--normalize",
        ],
    )

    main()

    captured = capsys.readouterr()
    print(captured.out, file=sys.stderr)

    lines = captured.out.strip().splitlines()
    plot_path = Path(lines[-1])

    assert plot_path.exists()
    assert plot_path.name == "compare_overlay.png"
    assert plot_path.parent.name == "rfiperf_compare"
