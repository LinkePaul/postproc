from pathlib import Path
import numpy as np


def layout_from_metadata(meta, kbsize=256):
    nants = int(meta["nants"])
    nchan = int(meta["nchan"])
    npol = int(meta["npol"])
    piperblk = int(meta["piperblk"])
    schan = int(meta["schan"])

    if piperblk % kbsize != 0:
        raise ValueError(f"piperblk={piperblk} is not divisible by kbsize={kbsize}")

    time_bins_per_block = piperblk // kbsize
    flags_per_block = nants * nchan * time_bins_per_block * npol

    if flags_per_block % 8 != 0:
        raise ValueError(f"flags_per_block={flags_per_block} is not divisible by 8")

    return {
        "nants": nants,
        "nchan": nchan,
        "npol": npol,
        "piperblk": piperblk,
        "kbsize": kbsize,
        "time_bins_per_block": time_bins_per_block,
        "flags_per_block": flags_per_block,
        "bytes_per_block": flags_per_block // 8,
        "block_shape": (nants, nchan, time_bins_per_block, npol),
        "schan": schan,
    }


def nblocks_in_file(mask_path, layout):
    mask_path = Path(mask_path)
    size = mask_path.stat().st_size
    bpb = layout["bytes_per_block"]

    if size % bpb != 0:
        raise ValueError(
            f"File size {size} is not an integer multiple of bytes_per_block={bpb}"
        )

    return size // bpb


def iter_mask_blocks(mask_path, layout):
    mask_path = Path(mask_path)
    bpb = layout["bytes_per_block"]
    shape = layout["block_shape"]

    with mask_path.open("rb") as f:
        while True:
            chunk = f.read(bpb)
            if not chunk:
                break
            if len(chunk) != bpb:
                raise EOFError("Trailing partial kurtosis block")

            raw = np.frombuffer(chunk, dtype=np.uint8)
            bits = np.unpackbits(raw, bitorder="little")
            yield bits.reshape(shape).astype(np.uint8)


def read_mask(mask_path, layout):
    nblocks = nblocks_in_file(mask_path, layout)
    nants = layout["nants"]
    nchan = layout["nchan"]
    npol = layout["npol"]
    tbpb = layout["time_bins_per_block"]

    out = np.empty((nants, nchan, nblocks * tbpb, npol), dtype=np.uint8)

    for i, block in enumerate(iter_mask_blocks(mask_path, layout)):
        t0 = i * tbpb
        t1 = t0 + tbpb
        out[:, :, t0:t1, :] = block

    return out


def write_mask(mask_path, mask, layout):
    mask_path = Path(mask_path)

    if mask.ndim != 4:
        raise ValueError(f"Mask must have 4 dims, got shape {mask.shape}")

    nants, nchan, ntime, npol = mask.shape
    tbpb = layout["time_bins_per_block"]

    if nants != layout["nants"]:
        raise ValueError(f"nants mismatch: expected {layout['nants']}, got {nants}")
    if nchan != layout["nchan"]:
        raise ValueError(f"nchan mismatch: expected {layout['nchan']}, got {nchan}")
    if npol != layout["npol"]:
        raise ValueError(f"npol mismatch: expected {layout['npol']}, got {npol}")
    if ntime % tbpb != 0:
        raise ValueError(f"ntime={ntime} is not divisible by time_bins_per_block={tbpb}")

    nblocks = ntime // tbpb

    with mask_path.open("wb") as f:
        for i in range(nblocks):
            t0 = i * tbpb
            t1 = t0 + tbpb
            block = mask[:, :, t0:t1, :]

            bits = np.asarray(block, dtype=np.uint8).reshape(-1)
            if np.any((bits != 0) & (bits != 1)):
                raise ValueError("Mask values must be 0 or 1")

            raw = np.packbits(bits, bitorder="little")
            f.write(raw.tobytes())


def read_mask_from_slice(slice_info, meta, kbsize=256):
    mask_path = slice_info.get("kurt_path")
    if mask_path is None:
        raise ValueError(f"No kurtosismask.bin for slice {slice_info['path']}")
    layout = layout_from_metadata(meta, kbsize=kbsize)
    return read_mask(mask_path, layout), layout


def get_schan(item):
    return item["schan"]


def check_kurt_compatible(meta_list):
    if not meta_list:
        raise ValueError("No metadata to check")

    ref = meta_list[0]

    for meta in meta_list[1:]:
        for key in ["lo", "nants", "nchan", "npol", "piperblk"]:
            if meta[key] != ref[key]:
                raise ValueError(
                    f"Mismatch in {key}: {ref['path']} has {ref[key]}, "
                    f"{meta['path']} has {meta[key]}"
                )

    meta_list = sorted(meta_list, key=get_schan)

    for i in range(1, len(meta_list)):
        prev = meta_list[i - 1]
        cur = meta_list[i]
        expected = prev["schan"] + prev["nchan"]

        if cur["schan"] != expected:
            raise ValueError(
                f"Non-contiguous SCHAN: expected {expected}, got {cur['schan']} "
                f"between {prev['path']} and {cur['path']}"
            )

    return True


def concat_masks(mask_list):
    if not mask_list:
        raise ValueError("No masks to concatenate")
    return np.concatenate(mask_list, axis=1)


def stream_concat_masks(slice_list, meta_list, out_path, kbsize=256, strict=False):
    if len(slice_list) != len(meta_list):
        raise ValueError("slice_list and meta_list must have same length")

    pairs = list(zip(slice_list, meta_list))
    pairs.sort(key=get_pair_schan)

    layouts = []
    nblocks_list = []

    for slice_info, meta in pairs:
        if slice_info["kurt_path"] is None:
            raise ValueError(f"Missing kurtosismask.bin for {slice_info['path']}")

        layout = layout_from_metadata(meta, kbsize=kbsize)
        layouts.append(layout)
        nblocks_list.append(nblocks_in_file(slice_info["kurt_path"], layout))

    check_kurt_compatible(meta_list)

    min_blocks = min(nblocks_list)
    max_blocks = max(nblocks_list)

    if min_blocks != max_blocks:
        if strict:
            raise ValueError(f"Different block counts: {nblocks_list}")

        print(f"Warning: different block counts {nblocks_list}, cropping to {min_blocks}")
        for (slice_info, _meta), nblocks in zip(pairs, nblocks_list):
            if nblocks != min_blocks:
                print(f"Warning: truncating {slice_info['kurt_path']} from {nblocks} to {min_blocks} blocks")

    iterators = []
    for i in range(len(pairs)):
        slice_info = pairs[i][0]
        layout = layouts[i]
        iterators.append(iter_mask_blocks(slice_info["kurt_path"], layout))

    out_path = Path(out_path)
    with out_path.open("wb") as f:
        for _ in range(min_blocks):
            blocks = [next(it) for it in iterators]
            block = np.concatenate(blocks, axis=1)
            raw = np.packbits(block.reshape(-1), bitorder="little")
            f.write(raw.tobytes())


def get_pair_schan(pair):
    return pair[1]["schan"]
