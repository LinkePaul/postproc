import json
from pathlib import Path

import numpy as np

from postproc_common.kurtio import layout_from_metadata, write_mask
from rfiperf.kurtosis import (
    normalize_lo,
    build_spliced_layout,
    build_spliced_layout_from_status,
    load_spliced_mask_from_obs_dir,
    load_spliced_mask_from_file,
    pol_to_index,
    select_pol,
    select_ant,
    summary_stats,
    zap_fraction_over_freq,
    zap_fraction_over_ant,
    zap_fraction_over_time,
)


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
    return data


def make_small_mask():
    mask = np.zeros((2, 4, 3, 2), dtype=np.uint8)

    # x
    mask[0, 0, 0, 0] = 1
    mask[0, 1, 1, 0] = 1
    mask[1, 2, 2, 0] = 1

    # y
    mask[0, 0, 0, 1] = 1
    mask[1, 3, 1, 1] = 1

    return mask


def test_normalize_lo():
    print("\nchecking LO normalization")
    assert normalize_lo("A") == "LoA"
    assert normalize_lo("LoA") == "LoA"
    assert normalize_lo("b") == "LoB"


def test_pol_to_index():
    print("\nchecking polarization parsing")
    assert pol_to_index("x") == 0
    assert pol_to_index("y") == 1
    assert pol_to_index("xy") is None


def test_select_pol():
    print("\nchecking polarization selection")
    mask = make_small_mask()

    mask_x = select_pol(mask, "x")
    mask_y = select_pol(mask, "y")
    mask_xy = select_pol(mask, "xy")

    print("x sum :", mask_x.sum())
    print("y sum :", mask_y.sum())
    print("xy sum:", mask_xy.sum())

    assert mask_x.shape == (2, 4, 3)
    assert mask_y.shape == (2, 4, 3)
    assert mask_xy.shape == (2, 4, 3)

    assert mask_x.sum() == 3
    assert mask_y.sum() == 2
    assert mask_xy.sum() == 4


def test_select_ant():
    print("\nchecking antenna selection")
    mask = make_small_mask()
    mask_x = select_pol(mask, "x")

    all_ant = select_ant(mask_x)
    one_ant = select_ant(mask_x, ant=1)

    print("all ant shape:", all_ant.shape)
    print("one ant shape:", one_ant.shape)

    assert all_ant.shape == (2, 4, 3)
    assert one_ant.shape == (1, 4, 3)
    assert one_ant.sum() == 1


def test_summary_stats():
    print("\nchecking summary stats")
    mask = make_small_mask()
    mask_x = select_pol(mask, "x")

    out = summary_stats(mask_x)
    print(out)

    assert out["ant"] is None
    assert out["nants_used"] == 2
    assert out["nchans"] == 4
    assert out["ntime"] == 3
    assert out["zapped_cells"] == 3
    assert out["total_cells"] == 24
    assert out["zap_fraction"] == 3 / 24


def test_fraction_vectors():
    print("\nchecking zap fraction vectors")
    mask = make_small_mask()
    mask_x = select_pol(mask, "x")

    freq = zap_fraction_over_freq(mask_x)
    ant = zap_fraction_over_ant(mask_x)
    time = zap_fraction_over_time(mask_x)

    print("freq:", freq)
    print("ant :", ant)
    print("time:", time)

    assert np.allclose(freq, [1 / 6, 1 / 6, 1 / 6, 0.0])
    assert np.allclose(ant, [2 / 12, 1 / 12])
    assert np.allclose(time, [1 / 8, 1 / 8, 1 / 8])


def test_build_spliced_layout():
    print("\nchecking spliced layout from metadata list")
    meta_list = [
        {
            "path": "a",
            "lo": "LoA",
            "schan": 352,
            "nants": 2,
            "nchan": 192,
            "npol": 2,
            "piperblk": 8192,
        },
        {
            "path": "b",
            "lo": "LoA",
            "schan": 544,
            "nants": 2,
            "nchan": 192,
            "npol": 2,
            "piperblk": 8192,
        },
    ]

    layout = build_spliced_layout(meta_list, kbsize=256)
    print(layout)

    assert layout["nants"] == 2
    assert layout["nchan"] == 384
    assert layout["npol"] == 2
    assert layout["time_bins_per_block"] == 32
    assert layout["block_shape"] == (2, 384, 32, 2)


def test_build_spliced_layout_from_status(tmp_path):
    print("\nchecking spliced layout from one status file plus nchan")
    status_path = tmp_path / "status_dump.json"
    status = make_status(status_path, 352)

    layout = build_spliced_layout_from_status(status, nchan=384, kbsize=256)
    print(layout)

    assert layout["nants"] == 2
    assert layout["nchan"] == 384
    assert layout["npol"] == 2
    assert layout["block_shape"] == (2, 384, 32, 2)


def test_load_spliced_mask_from_obs_dir(tmp_path):
    print("\nchecking obs directory mode")
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
    mask[1, 10, 5, 1] = 1

    out_path = obs / "LoA_spliced.kurtosismask.bin"
    write_mask(out_path, mask, layout)

    out_mask, lo, out_layout = load_spliced_mask_from_obs_dir(obs, "A", kbsize=256)

    print("lo:", lo)
    print("shape:", out_mask.shape)

    assert lo == "LoA"
    assert out_mask.shape == (2, 384, 64, 2)
    assert np.array_equal(out_mask, mask)
    assert out_layout["nchan"] == 384


def test_load_spliced_mask_from_file(tmp_path):
    print("\nchecking single file mode")
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
    mask[0, 1, 2, 0] = 1

    mask_path = tmp_path / "LoA_spliced.kurtosismask.bin"
    write_mask(mask_path, mask, layout)

    out_mask, lo, out_layout = load_spliced_mask_from_file(
        mask_path,
        status_path,
        384,
        kbsize=256,
    )

    print("lo:", lo)
    print("shape:", out_mask.shape)

    assert lo == "LoA"
    assert out_mask.shape == (2, 384, 64, 2)
    assert np.array_equal(out_mask, mask)
    assert out_layout["nchan"] == 384