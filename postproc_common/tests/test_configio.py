from pathlib import Path

from postproc_common.configio import (
    CONFIG_FILENAME,
    alias_for_key,
    alias_for_path,
    default_from_config,
    file_key_for_path,
    find_postproc_config,
    label_for_path,
    load_config_for_paths,
    load_postproc_config,
)


def test_load_postproc_config_aliases_and_defaults(tmp_path):
    config_path = tmp_path / CONFIG_FILENAME
    config_path.write_text(
        """
aliases:
  fil_61021_66884_4600708_J2022+5154_0001: "sigma = 5"
  fil_61111_61234_189484436_J2022+5154_0001: "no kurtosis"

defaults:
  rfiperf:
    snr:
      baseline: median
"""
    )

    cfg = load_postproc_config(config_path)

    assert cfg["aliases"]["fil_61021_66884_4600708_J2022+5154_0001"] == "sigma = 5"
    assert cfg["aliases"]["fil_61111_61234_189484436_J2022+5154_0001"] == "no kurtosis"
    assert cfg["defaults"]["rfiperf"]["snr"]["baseline"] == "median"
    assert cfg["_config_path"] == str(config_path)


def test_find_postproc_config_prefers_cwd_over_common_root(tmp_path):
    cwd = tmp_path / "cwd"
    data_root = tmp_path / "folded_rfi_pulsar"
    obs1 = data_root / "fil_61021_66884_4600708_J2022+5154_0001"
    obs2 = data_root / "fil_61111_61234_189484436_J2022+5154_0001"

    cwd.mkdir()
    obs1.mkdir(parents=True)
    obs2.mkdir(parents=True)

    cwd_config = cwd / CONFIG_FILENAME
    root_config = data_root / CONFIG_FILENAME

    cwd_config.write_text('aliases: {a: "cwd alias"}\n')
    root_config.write_text('aliases: {a: "root alias"}\n')

    p1 = obs1 / "spliced_loa_529.22ms_Cand.pfd.bestprof"
    p2 = obs2 / "spliced_loa_529.22ms_Cand.pfd.bestprof"
    p1.write_text("")
    p2.write_text("")

    found = find_postproc_config(paths=[p1, p2], cwd=cwd)
    assert found == cwd_config


def test_find_postproc_config_uses_common_root_when_no_cwd_config(tmp_path):
    cwd = tmp_path / "cwd"
    data_root = tmp_path / "folded_rfi_pulsar"
    obs1 = data_root / "fil_61021_66884_4600708_J2022+5154_0001"
    obs2 = data_root / "fil_61111_61234_189484436_J2022+5154_0001"

    cwd.mkdir()
    obs1.mkdir(parents=True)
    obs2.mkdir(parents=True)

    root_config = data_root / CONFIG_FILENAME
    root_config.write_text(
        """
aliases:
  fil_61021_66884_4600708_J2022+5154_0001: "sigma = 5"
"""
    )

    p1 = obs1 / "spliced_loa_529.22ms_Cand.pfd.bestprof"
    p2 = obs2 / "spliced_loa_529.22ms_Cand.pfd.bestprof"
    p1.write_text("")
    p2.write_text("")

    found = find_postproc_config(paths=[p1, p2], cwd=cwd)
    assert found == root_config


def test_load_config_for_paths_returns_empty_config_when_missing(tmp_path):
    cwd = tmp_path / "cwd"
    cwd.mkdir()

    cfg = load_config_for_paths(paths=None, cwd=cwd)

    assert cfg["aliases"] == {}
    assert cfg["defaults"] == {}
    assert cfg["_config_path"] is None


def test_file_key_alias_and_label_for_bestprof_path(tmp_path):
    data_root = tmp_path / "folded_rfi_pulsar"
    obs = data_root / "fil_61021_66884_4600708_J2022+5154_0001"
    obs.mkdir(parents=True)

    bestprof = obs / "spliced_loa_529.22ms_Cand.pfd.bestprof"
    bestprof.write_text("")

    cfg = {
        "aliases": {
            "fil_61021_66884_4600708_J2022+5154_0001": "sigma = 5",
        },
        "defaults": {},
    }

    assert file_key_for_path(bestprof) == "fil_61021_66884_4600708_J2022+5154_0001"
    assert alias_for_path(bestprof, cfg) == "sigma = 5"
    assert label_for_path(bestprof, cfg) == "sigma = 5"


def test_label_for_path_falls_back_to_file_key(tmp_path):
    data_root = tmp_path / "folded_rfi_pulsar"
    obs = data_root / "fil_61111_61234_189484436_J2022+5154_0001"
    obs.mkdir(parents=True)

    bestprof = obs / "spliced_loa_529.22ms_Cand.pfd.bestprof"
    bestprof.write_text("")

    cfg = {
        "aliases": {},
        "defaults": {},
    }

    key = "fil_61111_61234_189484436_J2022+5154_0001"
    assert file_key_for_path(bestprof) == key
    assert alias_for_key(key, cfg) is None
    assert label_for_path(bestprof, cfg) == key


def test_default_from_config_returns_nested_value_and_fallback():
    cfg = {
        "aliases": {},
        "defaults": {
            "rfiperf": {
                "snr": {
                    "baseline": "median",
                    "threshold_sigma": 1.5,
                }
            }
        },
    }

    assert default_from_config(cfg, "rfiperf", "snr", "baseline") == "median"
    assert default_from_config(cfg, "rfiperf", "snr", "threshold_sigma") == 1.5
    assert default_from_config(cfg, "rfiperf", "snr", "missing", fallback="x") == "x"
    assert default_from_config(cfg, "missing", fallback=None) is None
