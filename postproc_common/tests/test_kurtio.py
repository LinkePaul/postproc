import numpy as np
import pytest

from postproc_common.kurtio import (
    layout_from_metadata,
    nblocks_in_file,
    iter_mask_blocks,
    read_mask,
    write_mask,
    read_mask_from_slice,
    check_kurt_compatible,
    concat_masks,
    stream_concat_masks,
)


def make_meta(schan=352, nchan=192, nants=2, npol=2, piperblk=8192):
    return {
        "schan": schan,
        "nchan": nchan,
        "nants": nants,
        "npol": npol,
        "piperblk": piperblk,
        "lo": "LoA",
        "path": "dummy",
    }


def make_mask(layout, nblocks=2, fill=None):
    shape = (
        layout["nants"],
        layout["nchan"],
        nblocks * layout["time_bins_per_block"],
        layout["npol"],
    )

    if fill is None:
        rng = np.random.default_rng(1234)
        return rng.integers(0, 2, size=shape, dtype=np.uint8)

    return np.full(shape, fill, dtype=np.uint8)


def test_layout_from_metadata():
    meta = make_meta()
    layout = layout_from_metadata(meta, kbsize=256)

    assert layout["nants"] == 2
    assert layout["nchan"] == 192
    assert layout["npol"] == 2
    assert layout["time_bins_per_block"] == 32
    assert layout["flags_per_block"] == 2 * 192 * 32 * 2
    assert layout["bytes_per_block"] == (2 * 192 * 32 * 2) // 8
    assert layout["block_shape"] == (2, 192, 32, 2)


def test_layout_from_metadata_bad_division():
    meta = make_meta(piperblk=8193)

    with pytest.raises(ValueError):
        layout_from_metadata(meta, kbsize=256)


def test_write_read_roundtrip(tmp_path):
    meta = make_meta()
    layout = layout_from_metadata(meta, kbsize=256)
    mask = make_mask(layout, nblocks=3)

    path = tmp_path / "test.kurtosismask.bin"
    write_mask(path, mask, layout)

    reread = read_mask(path, layout)
    assert np.array_equal(mask, reread)


def test_nblocks_in_file(tmp_path):
    meta = make_meta()
    layout = layout_from_metadata(meta, kbsize=256)
    mask = make_mask(layout, nblocks=4)

    path = tmp_path / "test.kurtosismask.bin"
    write_mask(path, mask, layout)

    assert nblocks_in_file(path, layout) == 4


def test_iter_mask_blocks(tmp_path):
    meta = make_meta()
    layout = layout_from_metadata(meta, kbsize=256)
    mask = make_mask(layout, nblocks=2)

    path = tmp_path / "test.kurtosismask.bin"
    write_mask(path, mask, layout)

    blocks = list(iter_mask_blocks(path, layout))

    assert len(blocks) == 2
    assert blocks[0].shape == layout["block_shape"]

    tbpb = layout["time_bins_per_block"]
    assert np.array_equal(blocks[0], mask[:, :, 0:tbpb, :])
    assert np.array_equal(blocks[1], mask[:, :, tbpb:2 * tbpb, :])


def test_write_mask_rejects_bad_shape(tmp_path):
    meta = make_meta()
    layout = layout_from_metadata(meta, kbsize=256)
    bad = np.zeros((2, 192, 64), dtype=np.uint8)

    with pytest.raises(ValueError):
        write_mask(tmp_path / "x.bin", bad, layout)


def test_write_mask_rejects_bad_values(tmp_path):
    meta = make_meta()
    layout = layout_from_metadata(meta, kbsize=256)
    mask = make_mask(layout, nblocks=1)
    mask[0, 0, 0, 0] = 2

    with pytest.raises(ValueError):
        write_mask(tmp_path / "x.bin", mask, layout)


def test_read_mask_from_slice(tmp_path):
    meta = make_meta()
    layout = layout_from_metadata(meta, kbsize=256)
    mask = make_mask(layout, nblocks=1)

    path = tmp_path / "x.kurtosismask.bin"
    write_mask(path, mask, layout)

    slice_info = {"path": tmp_path, "kurt_path": path}
    out_mask, out_layout = read_mask_from_slice(slice_info, meta, kbsize=256)

    assert np.array_equal(out_mask, mask)
    assert out_layout["block_shape"] == layout["block_shape"]


def test_check_kurt_compatible_ok():
    meta_list = [
        make_meta(schan=352),
        make_meta(schan=544),
    ]

    assert check_kurt_compatible(meta_list) is True


def test_check_kurt_compatible_bad_gap():
    meta_list = [
        make_meta(schan=352),
        make_meta(schan=736),
    ]

    with pytest.raises(ValueError):
        check_kurt_compatible(meta_list)


def test_concat_masks():
    mask1 = np.zeros((2, 192, 64, 2), dtype=np.uint8)
    mask2 = np.ones((2, 192, 64, 2), dtype=np.uint8)

    out_mask = concat_masks([mask1, mask2])

    assert out_mask.shape == (2, 384, 64, 2)
    assert np.all(out_mask[:, :192, :, :] == 0)
    assert np.all(out_mask[:, 192:, :, :] == 1)


def test_stream_concat_masks(tmp_path):
    meta1 = make_meta(schan=352)
    meta2 = make_meta(schan=544)

    layout1 = layout_from_metadata(meta1, kbsize=256)
    layout2 = layout_from_metadata(meta2, kbsize=256)

    mask1 = make_mask(layout1, nblocks=2, fill=0)
    mask2 = make_mask(layout2, nblocks=2, fill=1)

    path1 = tmp_path / "a.bin"
    path2 = tmp_path / "b.bin"
    write_mask(path1, mask1, layout1)
    write_mask(path2, mask2, layout2)

    slice_list = [
        {"path": tmp_path / "LoA.C0352", "kurt_path": path1},
        {"path": tmp_path / "LoA.C0544", "kurt_path": path2},
    ]
    meta_list = [meta1, meta2]

    out_path = tmp_path / "spliced.bin"
    stream_concat_masks(slice_list, meta_list, out_path, kbsize=256)

    out_meta = make_meta(schan=352, nchan=384)
    out_layout = layout_from_metadata(out_meta, kbsize=256)
    out_mask = read_mask(out_path, out_layout)

    assert out_mask.shape == (2, 384, 64, 2)
    assert np.all(out_mask[:, :192, :, :] == 0)
    assert np.all(out_mask[:, 192:, :, :] == 1)
