import json
from pathlib import Path

from rfiperf.cli import main


def make_bestprof(path, candidate="test", dm=21.6, p_bary_ms=529.2172, chi=37.7, sigma=44.8):
    lines = [
        "# Input file       =  spliced_loa.fil",
        f"# Candidate        =  {candidate}",
        "# Telescope        =  Unknown",
        "# Profile Bins     =  8",
        f"# Reduced chi-sqr  =  {chi}",
        f"# Prob(Noise)      <  0   (~{sigma} sigma)",
        f"# Best DM          =  {dm}",
        "# P_topo (ms)      =  529.212564950708  +/- 0.00599",
        f"# P_bary (ms)      =  {p_bary_ms}  +/- 0.00599",
        "######################################################",
        "   0  9",
        "   1  10",
        "   2  11",
        "   3  20",
        "   4  10",
        "   5  9",
        "   6  11",
        "   7  10",
    ]
    path.write_text("\n".join(lines) + "\n")


def test_rfiperf_snr_single_file_json_summary(tmp_path, monkeypatch, capsys):
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

    out = json.loads(capsys.readouterr().out)
    assert out["candidate"] == "test"
    assert out["best_dm"] == 21.6
    assert out["p_bary_ms"] == 529.2172
    assert out["baseline"] == "median"
    assert out["peak_bin"] == 3
    assert out["profile_snr"] > 0
    assert out["prob_noise"] == "<  0"
    assert out["presto_sigma"] == 44.8


def test_rfiperf_snr_compare_json_summary(tmp_path, monkeypatch, capsys):
    p1 = tmp_path / "a.bestprof"
    p2 = tmp_path / "b.bestprof"

    make_bestprof(p1, candidate="a", chi=10.0, sigma=10.0)
    make_bestprof(p2, candidate="b", chi=20.0, sigma=20.0)

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

    out = json.loads(capsys.readouterr().out)
    assert out["mode"] == "compare"
    assert out["baseline"] == "median"
    assert out["n_files"] == 2
    assert len(out["files"]) == 2
    assert set(item["candidate"] for item in out["files"]) == {"a", "b"}


def test_rfiperf_snr_compare_overlay_plot(tmp_path, monkeypatch, capsys):
    p1 = tmp_path / "a.bestprof"
    p2 = tmp_path / "b.bestprof"
    outdir = tmp_path / "plots"

    make_bestprof(p1, candidate="a")
    make_bestprof(p2, candidate="b")

    monkeypatch.setattr(
        "sys.argv",
        [
            "rfiperf",
            "snr",
            str(p1),
            str(p2),
            "--plot",
            "overlay",
            "--normalize",
            "--outdir",
            str(outdir),
        ],
    )

    main()

    out_lines = capsys.readouterr().out.strip().splitlines()
    plot_path = Path(out_lines[-1])
    assert plot_path.exists()
    assert plot_path.name == "compare_overlay.png"


def test_rfiperf_snr_multiple_inputs_prints_compare_table(tmp_path, monkeypatch, capsys):
    p1 = tmp_path / "a.bestprof"
    p2 = tmp_path / "b.bestprof"

    make_bestprof(p1, candidate="a", chi=10.0)
    make_bestprof(p2, candidate="b", chi=20.0)

    monkeypatch.setattr(
        "sys.argv",
        [
            "rfiperf",
            "snr",
            str(p1),
            str(p2),
            "--baseline",
            "median",
        ],
    )

    main()

    out = capsys.readouterr().out
    assert "file" in out
    assert "prof_snr" in out
    assert p1.parent.name in out
    assert p2.parent.name in out


def test_rfiperf_snr_compare_json_uses_alias_from_postproc_config(tmp_path, monkeypatch, capsys):
    data_root = tmp_path / "folded_rfi_pulsar"
    obs1 = data_root / "fil_61021_66884_4600708_J2022+5154_0001"
    obs2 = data_root / "fil_61111_61234_189484436_J2022+5154_0001"
    obs1.mkdir(parents=True)
    obs2.mkdir(parents=True)

    p1 = obs1 / "spliced_loa_529.22ms_Cand.pfd.bestprof"
    p2 = obs2 / "spliced_loa_529.22ms_Cand.pfd.bestprof"
    make_bestprof(p1, candidate="PSR_2022+5154", chi=100.0, sigma=50.0)
    make_bestprof(p2, candidate="PSR_2022+5154", chi=50.0, sigma=20.0)

    (data_root / "postproc_config.yaml").write_text(
        """
aliases:
  fil_61021_66884_4600708_J2022+5154_0001: "sigma = 5"
  fil_61111_61234_189484436_J2022+5154_0001: "no kurtosis"
"""
    )

    monkeypatch.chdir(data_root)

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

    out = json.loads(capsys.readouterr().out)
    labels = [item["file_label"] for item in out["files"]]
    assert "sigma = 5" in labels
    assert "no kurtosis" in labels


def test_rfiperf_snr_compare_table_uses_alias_from_postproc_config(tmp_path, monkeypatch, capsys):
    data_root = tmp_path / "folded_rfi_pulsar"
    obs1 = data_root / "fil_61021_66884_4600708_J2022+5154_0001"
    obs2 = data_root / "fil_61111_61234_189484436_J2022+5154_0001"
    obs1.mkdir(parents=True)
    obs2.mkdir(parents=True)

    p1 = obs1 / "spliced_loa_529.22ms_Cand.pfd.bestprof"
    p2 = obs2 / "spliced_loa_529.22ms_Cand.pfd.bestprof"
    make_bestprof(p1)
    make_bestprof(p2)

    (data_root / "postproc_config.yaml").write_text(
        """
aliases:
  fil_61021_66884_4600708_J2022+5154_0001: "sigma = 5"
  fil_61111_61234_189484436_J2022+5154_0001: "no kurtosis"
"""
    )

    monkeypatch.chdir(data_root)

    monkeypatch.setattr(
        "sys.argv",
        [
            "rfiperf",
            "snr",
            str(p1),
            str(p2),
            "--baseline",
            "median",
        ],
    )

    main()

    out = capsys.readouterr().out
    assert "sigma = 5" in out
    assert "no kurtosis" in out


def test_rfiperf_snr_overlay_uses_alias_from_postproc_config(tmp_path, monkeypatch, capsys):
    data_root = tmp_path / "folded_rfi_pulsar"
    obs1 = data_root / "fil_61021_66884_4600708_J2022+5154_0001"
    obs2 = data_root / "fil_61111_61234_189484436_J2022+5154_0001"
    obs1.mkdir(parents=True)
    obs2.mkdir(parents=True)

    p1 = obs1 / "spliced_loa_529.22ms_Cand.pfd.bestprof"
    p2 = obs2 / "spliced_loa_529.22ms_Cand.pfd.bestprof"
    make_bestprof(p1)
    make_bestprof(p2)

    (data_root / "postproc_config.yaml").write_text(
        """
aliases:
  fil_61021_66884_4600708_J2022+5154_0001: "sigma = 5"
  fil_61111_61234_189484436_J2022+5154_0001: "no kurtosis"
"""
    )

    monkeypatch.chdir(data_root)

    monkeypatch.setattr(
        "sys.argv",
        [
            "rfiperf",
            "snr",
            str(p1),
            str(p2),
            "--plot",
            "overlay",
            "--normalize",
        ],
    )

    main()

    out_lines = capsys.readouterr().out.strip().splitlines()
    plot_path = Path(out_lines[-1])
    assert plot_path.exists()
    assert plot_path.parent.name == "rfiperf_compare"
