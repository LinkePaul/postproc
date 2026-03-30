from pathlib import Path

import numpy as np

from postproc_common.discovery import discover_slices
from postproc_common.metadata import load_all_metadata, read_status
from postproc_common.kurtio import layout_from_metadata, read_mask


def normalize_lo(lo):
    lo = lo.strip()
    if lo.startswith("Lo"):
        return lo
    return "Lo" + lo.upper()


def build_spliced_layout(meta_list, kbsize=256):
    layout = layout_from_metadata(meta_list[0], kbsize=kbsize)
    layout["nchan"] = sum(meta["nchan"] for meta in meta_list)
    layout["flags_per_block"] = (
        layout["nants"] * layout["nchan"] * layout["time_bins_per_block"] * layout["npol"]
    )
    layout["bytes_per_block"] = layout["flags_per_block"] // 8
    layout["block_shape"] = (
        layout["nants"],
        layout["nchan"],
        layout["time_bins_per_block"],
        layout["npol"],
    )
    return layout


def build_spliced_layout_from_status(status, nchan, kbsize=256):
    meta = {
        "schan": int(status["SCHAN"]),
        "nants": int(status["NANTS"]),
        "nchan": int(nchan),
        "npol": int(status["NPOL"]),
        "piperblk": int(status["PIPERBLK"]),
    }
    return layout_from_metadata(meta, kbsize=kbsize)


def load_spliced_mask_from_obs_dir(obs_root, lo, kbsize=256):
    obs_root = Path(obs_root)
    lo = normalize_lo(lo)

    slice_list = discover_slices(obs_root, lo=lo)
    if not slice_list:
        raise ValueError(f"No slice directories found for {lo}")

    meta_list = load_all_metadata(slice_list)
    layout = build_spliced_layout(meta_list, kbsize=kbsize)

    mask_path = obs_root / f"{lo}_spliced.kurtosismask.bin"
    mask = read_mask(mask_path, layout)
    return mask, lo, layout


def load_spliced_mask_from_file(mask_path, status_path, nchan, kbsize=256):
    mask_path = Path(mask_path)
    status = read_status(status_path)
    layout = build_spliced_layout_from_status(status, nchan, kbsize=kbsize)
    mask = read_mask(mask_path, layout)

    lo = None
    name = mask_path.name
    if name.startswith("Lo") and "_" in name:
        lo = name.split("_", 1)[0]

    return mask, lo, layout


def pol_to_index(pol):
    pol = pol.lower()
    if pol == "x":
        return 0
    if pol == "y":
        return 1
    if pol == "xy":
        return None
    raise ValueError(f"Unknown pol: {pol}")


def select_pol(mask, pol):
    idx = pol_to_index(pol)
    if idx is None:
        return np.logical_or(mask[:, :, :, 0], mask[:, :, :, 1]).astype(np.uint8)
    return mask[:, :, :, idx]


def select_ant(mask_pol, ant=None):
    if ant is None:
        return mask_pol
    return mask_pol[ant:ant + 1, :, :]


def summary_stats(mask_pol, ant=None):
    data = select_ant(mask_pol, ant=ant)
    total_cells = data.size
    zapped_cells = int(data.sum())
    zap_fraction = zapped_cells / total_cells if total_cells else 0.0
    return {
        "ant": ant,
        "nants_used": int(data.shape[0]),
        "nchans": int(data.shape[1]),
        "ntime": int(data.shape[2]),
        "zapped_cells": zapped_cells,
        "total_cells": int(total_cells),
        "zap_fraction": zap_fraction,
    }


def zap_fraction_over_freq(mask_pol, ant=None):
    data = select_ant(mask_pol, ant=ant)
    return data.mean(axis=(0, 2))


def zap_fraction_over_ant(mask_pol):
    return mask_pol.mean(axis=(1, 2))


def zap_fraction_over_time(mask_pol, ant=None):
    data = select_ant(mask_pol, ant=ant)
    return data.mean(axis=(0, 1))


def extract_waterfall(
    mask_pol,
    ant,
    tstart=None,
    tend=None,
    fstart=None,
    fend=None,
):
    """
    Extract one antenna's 2D channel-vs-time kurtosis-mask view.

    Parameters
    ----------
    mask_pol : np.ndarray
        Selected polarization mask with shape (nant, nchan, ntime)
    ant : int
        Antenna index
    tstart, tend : int | None
        Optional crop in time-bin coordinates
    fstart, fend : int | None
        Optional crop in channel coordinates

    Returns
    -------
    np.ndarray
        Shape (nchan_crop, ntime_crop)
    """
    if mask_pol.ndim != 3:
        raise ValueError(
            f"Expected selected mask with shape (nant, nchan, ntime), got {mask_pol.shape}"
        )

    nant, nchan, ntime = mask_pol.shape

    if not (0 <= ant < nant):
        raise ValueError(f"Invalid antenna index {ant}, valid range is 0..{nant - 1}")

    t0 = 0 if tstart is None else tstart
    t1 = ntime if tend is None else tend
    f0 = 0 if fstart is None else fstart
    f1 = nchan if fend is None else fend

    if t0 < 0 or t1 < 0 or f0 < 0 or f1 < 0:
        raise ValueError("Slice bounds must be non-negative")
    if t0 > t1:
        raise ValueError(f"Invalid time slice: start={t0} must be <= stop={t1}")
    if f0 > f1:
        raise ValueError(f"Invalid channel slice: start={f0} must be <= stop={f1}")
    if t1 > ntime:
        raise ValueError(f"Invalid time slice: stop={t1} exceeds axis size {ntime}")
    if f1 > nchan:
        raise ValueError(f"Invalid channel slice: stop={f1} exceeds axis size {nchan}")

    return mask_pol[ant, f0:f1, t0:t1]

def extract_waterfalls(
    mask_pol,
    ants=None,
    tstart=None,
    tend=None,
    fstart=None,
    fend=None,
):
    """
    Extract one or more antenna 2D channel-vs-time waterfall slices.

    Parameters
    ----------
    mask_pol : np.ndarray
        Selected polarization mask with shape (nant, nchan, ntime)
    ants : sequence[int] | None
        Antenna indices to extract. If None, use all antennas.
    tstart, tend : int | None
        Optional crop in time-bin coordinates
    fstart, fend : int | None
        Optional crop in channel coordinates

    Returns
    -------
    tuple[list[np.ndarray], list[int]]
        Waterfall slices with shape (nchan_crop, ntime_crop), plus antenna indices
    """
    if mask_pol.ndim != 3:
        raise ValueError(
            f"Expected selected mask with shape (nant, nchan, ntime), got {mask_pol.shape}"
        )

    nant, nchan, ntime = mask_pol.shape

    if ants is None:
        ants = list(range(nant))
    else:
        ants = list(ants)

    t0 = 0 if tstart is None else tstart
    t1 = ntime if tend is None else tend
    f0 = 0 if fstart is None else fstart
    f1 = nchan if fend is None else fend

    if t0 < 0 or t1 < 0 or f0 < 0 or f1 < 0:
        raise ValueError("Slice bounds must be non-negative")
    if t0 > t1:
        raise ValueError(f"Invalid time slice: start={t0} must be <= stop={t1}")
    if f0 > f1:
        raise ValueError(f"Invalid channel slice: start={f0} must be <= stop={f1}")
    if t1 > ntime:
        raise ValueError(f"Invalid time slice: stop={t1} exceeds axis size {ntime}")
    if f1 > nchan:
        raise ValueError(f"Invalid channel slice: stop={f1} exceeds axis size {nchan}")

    out = []
    used_ants = []

    for ant in ants:
        if not (0 <= ant < nant):
            raise ValueError(f"Invalid antenna index {ant}, valid range is 0..{nant - 1}")
        out.append(mask_pol[ant, f0:f1, t0:t1])
        used_ants.append(int(ant))

    return out, used_ants
