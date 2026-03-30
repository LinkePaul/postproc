import json
from pathlib import Path

import numpy as np
import pytest

from postproc_common.kurtio import layout_from_metadata, write_mask

from rfiperf.kurtosis import (
    ant_label_for_index,
    build_single_layout_from_status,
    build_spliced_layout_from_statuses,
    build_waterfall_axis_info,
    discover_lo_statuses,
    infer_lo_from_mask_path,
    infer_nchan_from_status,
    load_spliced_mask_from_file,
    load_spliced_mask_from_obs_dir,
    normalize_lo,
    resolve_ant_index,
    resolve_kurtosis_input,
    select_pol,
    stream_extract_waterfall,
    stream_extract_waterfalls,
    summary_stats,
    zap_fraction_over_ant,
    zap_fraction_over_freq,
    zap_fraction_over_time,
)


def make_status(path: Path, schan: int, *, obsfreq: float) -> dict:
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
        "PKTSTOP": 150_000_000,  # 300 s with TBIN = 2e-06
        "SOURCE": "J2022+5154",
        "OBSBW": 96,
        "OBSNCHAN": 5376,
        "OBSFREQ": obsfreq,
        "ANTNMS00": "1bA,1cA,1dA,1eA,1fA,1gA,1hA,4jA,1kA,2aA,2bA,2cA,2dA,2eA,2fA,4gA,2hA",
        "ANTNMS01": "2jA,2kA,2lA,2mA,3cA,3dA,5bA,5cA,3lA,4eA,5eA",
    }
    path.write_text(json.dumps(data))
    return data


def make_obs(tmp_path: Path):
    obs = tmp_path / "obs"
    obs.mkdir()

    d1 = obs / "LoA.C0352"
    d2 = obs / "LoA.C0544"
    d1.mkdir()
    d2.mkdir()

    s1 = make_status(d1 / "status_dump.json", 352, obsfreq=1211.75)
    s2 = make_status(d2 / "status_dump.json", 544, obsfreq=1307.75)

    meta = {
        "schan": 0,
        "nants": 28,
        "nchan": 384,
        "npol": 2,
        "piperblk": 8192,
    }
    layout = layout_from_metadata(meta, kbsize=256)

    mask = np.zeros((28, 384, 64, 2), dtype=np.uint8)

    # make antenna 15 / 4g distinctive
    mask[15, 10:20, 5:8, 0] = 1
    mask[15, 200:210, 40:43, 0] = 1

    # make another antenna distinctive too
    mask[0, 0:4, 0:2, 0] = 1
    mask[1, 50:55, 10:12, 1] = 1

    mask_path = obs / "LoA_spliced.kurtosismask.bin"
    write_mask(mask_path, mask, layout)

    return {
        "obs": obs,
        "mask_path": mask_path,
        "mask": mask,
        "layout": layout,
        "statuses": [s1, s2],
    }


def test_normalize_lo():
    assert normalize_lo("A") == "LoA"
    assert normalize_lo("LoA") == "LoA"
    assert normalize_lo("b") == "LoB"


def test_infer_lo_from_mask_path():
    assert infer_lo_from_mask_path(Path("LoA_spliced.kurtosismask.bin")) == "LoA"
    assert infer_lo_from_mask_path(Path("LoB_spliced.kurtosismask.bin")) == "LoB"
    assert infer_lo_from_mask_path(Path("weird.bin")) is None


def test_infer_nchan_from_status_prefers_present_key(tmp_path):
    status = make_status(tmp_path / "status_dump.json", 352, obsfreq=1211.75)
    assert infer_nchan_from_status(status) == 192

    del status["NCHAN"]
    assert infer_nchan_from_status(status) == 96

    del status["PKTNCHAN"]
    assert infer_nchan_from_status(status) == 5376


def test_discover_lo_statuses_returns_sorted_statuses(tmp_path):
    data = make_obs(tmp_path)
    obs = data["obs"]

    lo, statuses = discover_lo_statuses(obs, lo="A")

    assert lo == "LoA"
    assert len(statuses) == 2
    assert [int(s["SCHAN"]) for s in statuses] == [352, 544]


def test_build_spliced_layout_from_statuses_has_ant_names_and_freq_metadata(tmp_path):
    data = make_obs(tmp_path)

    layout = build_spliced_layout_from_statuses(data["statuses"], kbsize=256)

    assert layout["nants"] == 28
    assert layout["nchan"] == 384
    assert layout["ant_names"][0] == "1b"
    assert layout["ant_names"][15] == "4g"
    assert layout["ant_names"][-1] == "5e"
    assert layout["foff_mhz"] == 0.5
    assert layout["fch1_mhz"] is not None


def test_build_single_layout_from_status_has_ant_names(tmp_path):
    status = make_status(tmp_path / "status_dump.json", 352, obsfreq=1211.75)

    layout = build_single_layout_from_status(status, kbsize=256)

    assert layout["nants"] == 28
    assert layout["nchan"] == 192
    assert layout["ant_names"][0] == "1b"
    assert layout["ant_names"][15] == "4g"
    assert layout["schan"] == 352


def test_resolve_kurtosis_input_from_obs_dir_autodiscovers_everything(tmp_path):
    data = make_obs(tmp_path)

    mask_path, lo, layout = resolve_kurtosis_input(data["obs"])

    assert mask_path == data["mask_path"]
    assert lo == "LoA"
    assert layout["nchan"] == 384
    assert layout["nants"] == 28
    assert layout["ant_names"][15] == "4g"


def test_resolve_kurtosis_input_from_spliced_file_autodiscovers_statuses(tmp_path):
    data = make_obs(tmp_path)

    mask_path, lo, layout = resolve_kurtosis_input(data["mask_path"])

    assert mask_path == data["mask_path"]
    assert lo == "LoA"
    assert layout["nchan"] == 384
    assert layout["ant_names"][15] == "4g"


def test_load_spliced_mask_from_obs_dir_roundtrip(tmp_path):
    data = make_obs(tmp_path)

    mask, lo, layout = load_spliced_mask_from_obs_dir(data["obs"], lo="A")

    assert lo == "LoA"
    assert mask.shape == (28, 384, 64, 2)
    assert layout["ant_names"][15] == "4g"
    assert int(mask[15, 10:20, 5:8, 0].sum()) > 0


def test_load_spliced_mask_from_file_roundtrip(tmp_path):
    data = make_obs(tmp_path)

    mask, lo, layout = load_spliced_mask_from_file(data["mask_path"])

    assert lo == "LoA"
    assert mask.shape == (28, 384, 64, 2)
    assert layout["ant_names"][0] == "1b"


def test_resolve_ant_index_and_ant_label(tmp_path):
    data = make_obs(tmp_path)
    _, _, layout = resolve_kurtosis_input(data["mask_path"])

    assert resolve_ant_index(layout, "4g") == 15
    assert resolve_ant_index(layout, "1b") == 0
    assert resolve_ant_index(layout, "15") == 15
    assert resolve_ant_index(layout, 15) == 15
    assert ant_label_for_index(layout, 15) == "4g"
    assert ant_label_for_index(layout, 0) == "1b"

    with pytest.raises(ValueError):
        resolve_ant_index(layout, "9z")


def test_select_pol_and_summary_reducers(tmp_path):
    data = make_obs(tmp_path)
    mask, _, _ = load_spliced_mask_from_file(data["mask_path"])

    mask_x = select_pol(mask, "x")
    mask_y = select_pol(mask, "y")
    mask_xy = select_pol(mask, "xy")

    assert mask_x.shape == (28, 384, 64)
    assert mask_y.shape == (28, 384, 64)
    assert mask_xy.shape == (28, 384, 64)

    summary = summary_stats(mask_x)
    assert summary["nants_used"] == 28
    assert summary["nchans"] == 384
    assert summary["ntime"] == 64

    zf_freq = zap_fraction_over_freq(mask_x)
    zf_ant = zap_fraction_over_ant(mask_x)
    zf_time = zap_fraction_over_time(mask_x)

    assert zf_freq.shape == (384,)
    assert zf_ant.shape == (28,)
    assert zf_time.shape == (64,)


def test_exact_duration_is_applied_from_status(tmp_path):
    data = make_obs(tmp_path)
    _, _, layout = resolve_kurtosis_input(data["mask_path"])

    # 300 s / 64 mask bins
    assert pytest.approx(layout["tbinsize_sec"], rel=0, abs=1e-12) == 300.0 / 64.0
    assert pytest.approx(layout["duration_sec"], rel=0, abs=1e-12) == 300.0


def test_build_waterfall_axis_info_uses_exact_duration(tmp_path):
    data = make_obs(tmp_path)
    _, _, layout = resolve_kurtosis_input(data["mask_path"])

    axis_info = build_waterfall_axis_info(layout, tstart=5, fstart=7)

    assert axis_info["time_start"] == 5
    assert axis_info["channel_start"] == 7
    assert pytest.approx(axis_info["dt_sec"], rel=0, abs=1e-12) == 300.0 / 64.0
    assert axis_info["f0_mhz"] is not None
    assert axis_info["df_mhz"] == 0.5


def test_stream_extract_waterfall_for_named_antenna(tmp_path):
    data = make_obs(tmp_path)
    _, _, layout = resolve_kurtosis_input(data["mask_path"])

    ant_idx = resolve_ant_index(layout, "4g")
    wf, returned_ant_idx = stream_extract_waterfall(
        data["mask_path"],
        layout,
        "x",
        ant=ant_idx,
    )

    assert returned_ant_idx == 15
    assert wf.shape == (384, 64)
    assert int(wf[10:20, 5:8].sum()) > 0
    assert int(wf[200:210, 40:43].sum()) > 0


def test_stream_extract_waterfalls_for_all_ants(tmp_path):
    data = make_obs(tmp_path)
    _, _, layout = resolve_kurtosis_input(data["mask_path"])

    wfs, ant_list = stream_extract_waterfalls(
        data["mask_path"],
        layout,
        "x",
        ants=None,
        tstart=0,
        tend=16,
    )

    assert len(wfs) == 28
    assert ant_list == list(range(28))
    assert wfs[0].shape == (384, 16)
    assert wfs[15].shape == (384, 16)
    assert int(wfs[15][10:20, 5:8].sum()) > 0
