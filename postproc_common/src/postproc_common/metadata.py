import json
from pathlib import Path


def read_status(status_path):
    status_path = Path(status_path)
    with status_path.open("r") as f:
        return json.load(f)


def load_slice_metadata(slice_info):
    status_path = slice_info.get("status_path")
    if status_path is None:
        raise ValueError(f"No status_dump.json for slice {slice_info['path']}")

    raw = read_status(status_path)

    return {
        "path": slice_info["path"],
        "lo": slice_info["lo"],
        "schan": int(raw["SCHAN"]),
        "subband": raw.get("SUBBAND"),
        "nants": int(raw["NANTS"]),
        "nchan": int(raw["NCHAN"]),
        "npol": int(raw["NPOL"]),
        "piperblk": int(raw["PIPERBLK"]),
        "chan_bw": float(raw["CHAN_BW"]),
        "tbin": float(raw["TBIN"]),
        "source": raw.get("SOURCE") or raw.get("SRC_NAME"),
        "obsbw": float(raw["OBSBW"]) if "OBSBW" in raw else None,
        "obsnchan": int(raw["OBSNCHAN"]) if "OBSNCHAN" in raw else None,
        "raw": raw,
    }


def load_all_metadata(slice_list):
    out = []
    for slice_info in slice_list:
        out.append(load_slice_metadata(slice_info))
    return out


def get_schan(meta):
    return meta["schan"]


def check_metadata_compatible(meta_list):
    if not meta_list:
        raise ValueError("No metadata to check")

    ref = meta_list[0]
    keys = ["lo", "nants", "nchan", "npol", "piperblk", "chan_bw", "tbin"]

    for meta in meta_list[1:]:
        for key in keys:
            if meta[key] != ref[key]:
                raise ValueError(
                    f"Metadata mismatch for {key}: "
                    f"{ref['path']} has {ref[key]}, {meta['path']} has {meta[key]}"
                )

    meta_sorted = sorted(meta_list, key=get_schan)

    for i in range(1, len(meta_sorted)):
        prev = meta_sorted[i - 1]
        cur = meta_sorted[i]
        expected = prev["schan"] + prev["nchan"]

        if cur["schan"] != expected:
            raise ValueError(
                f"Non-contiguous SCHAN: {prev['path']} -> {cur['path']} "
                f"(expected {expected}, got {cur['schan']})"
            )

    return True