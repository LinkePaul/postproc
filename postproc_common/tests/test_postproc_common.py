import json
import numpy as np

from postproc_common.discovery import discover_slices
from postproc_common.metadata import load_all_metadata, check_metadata_compatible
from postproc_common.kurtio import (
    layout_from_metadata,
    write_mask,
    stream_concat_masks,
    read_mask,
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


def test_postproc_common_end_to_end(tmp_path):
    obs = tmp_path / "obs"
    obs.mkdir()

    d1 = obs / "LoA.C0352"
    d2 = obs / "LoA.C0544"
    d1.mkdir()
    d2.mkdir()

    make_status(d1 / "status_dump.json", 352)
    make_status(d2 / "status_dump.json", 544)

    meta1 = {
        "path": d1,
        "lo": "LoA",
        "schan": 352,
        "nants": 2,
        "nchan": 192,
        "npol": 2,
        "piperblk": 8192,
        "chan_bw": 0.5,
        "tbin": 2e-06,
    }
    meta2 = {
        "path": d2,
        "lo": "LoA",
        "schan": 544,
        "nants": 2,
        "nchan": 192,
        "npol": 2,
        "piperblk": 8192,
        "chan_bw": 0.5,
        "tbin": 2e-06,
    }

    layout1 = layout_from_metadata(meta1, kbsize=256)
    layout2 = layout_from_metadata(meta2, kbsize=256)

    mask1 = np.zeros((2, 192, 64, 2), dtype=np.uint8)
    mask2 = np.ones((2, 192, 64, 2), dtype=np.uint8)

    write_mask(d1 / "one.kurtosismask.bin", mask1, layout1)
    write_mask(d2 / "two.kurtosismask.bin", mask2, layout2)

    slice_list = discover_slices(obs, lo="A")
    assert len(slice_list) == 2

    meta_list = load_all_metadata(slice_list)
    assert check_metadata_compatible(meta_list) is True

    out_path = obs / "LoA_spliced.kurtosismask.bin"
    stream_concat_masks(slice_list, meta_list, out_path, kbsize=256)

    out_meta = dict(meta_list[0])
    out_meta["nchan"] = 384
    out_layout = layout_from_metadata(out_meta, kbsize=256)

    out_mask = read_mask(out_path, out_layout)

    assert out_mask.shape == (2, 384, 64, 2)
    assert np.all(out_mask[:, :192, :, :] == 0)
    assert np.all(out_mask[:, 192:, :, :] == 1)
