from pathlib import Path

import numpy as np

from postproc_common.metadata import read_status
from postproc_common.kurtio import (
    iter_mask_blocks,
    layout_from_metadata,
    nblocks_in_file,
    read_mask,
)


def normalize_lo(lo):
    lo = lo.strip()
    if lo.startswith("Lo"):
        return lo
    return "Lo" + lo.upper()


def infer_lo_from_mask_path(mask_path):
    mask_path = Path(mask_path)
    name = mask_path.name

    for prefix in ("LoA", "LoB", "LoC", "LoD"):
        if name.startswith(prefix):
            return prefix

    return None


def _first_present(mapping, keys, default=None):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _maybe_float(value):
    if value is None:
        return None
    return float(value)


def infer_nchan_from_status(status):
    for key in ("NCHAN", "PKTNCHAN", "OBSNCHAN"):
        if key in status:
            return int(status[key])
    raise ValueError("Could not infer nchan from status_dump.json")


def _slice_status_path(slice_dir):
    return Path(slice_dir) / "status_dump.json"


def _parse_ant_names(status):
    parts = []

    for key in sorted(k for k in status if k.startswith("ANTNMS")):
        value = status.get(key)
        if value:
            parts.extend([x.strip() for x in str(value).split(",") if x.strip()])

    if not parts:
        value = status.get("ANTNAMES")
        if value:
            parts.extend([x.strip() for x in str(value).split(",") if x.strip()])

    return parts


def _display_ant_name(name):
    if not name:
        return name
    if len(name) >= 2 and name[-1].isalpha() and name[-1].isupper():
        return name[:-1]
    return name


def _attach_ant_names(layout, status):
    ant_names_raw = _parse_ant_names(status)
    if ant_names_raw:
        layout = dict(layout)
        layout["ant_names_raw"] = ant_names_raw
        layout["ant_names"] = [_display_ant_name(x) for x in ant_names_raw]
    return layout


def discover_lo_statuses(obs_root, lo=None):
    obs_root = Path(obs_root)

    if lo is not None:
        lo_list = [normalize_lo(lo)]
    else:
        lo_list = sorted(
            {
                p.name.split(".")[0]
                for p in obs_root.iterdir()
                if p.is_dir() and p.name.startswith("Lo") and ".C" in p.name
            }
        )

    if not lo_list:
        raise ValueError(f"No LO slice directories found in {obs_root}")

    all_found = {}

    for lo_name in lo_list:
        statuses = []
        for slice_dir in sorted(obs_root.glob(f"{lo_name}.C*")):
            status_path = _slice_status_path(slice_dir)
            if status_path.exists():
                status = read_status(status_path)
                status["_slice_dir"] = str(slice_dir)
                status["_status_path"] = str(status_path)
                statuses.append(status)

        if statuses:
            statuses.sort(key=lambda s: int(s["SCHAN"]))
            all_found[lo_name] = statuses

    if lo is not None:
        lo_name = normalize_lo(lo)
        if lo_name not in all_found:
            raise ValueError(f"No status_dump.json files found for {lo_name} in {obs_root}")
        return lo_name, all_found[lo_name]

    if len(all_found) == 1:
        lo_name = next(iter(all_found))
        return lo_name, all_found[lo_name]

    raise ValueError(
        f"Multiple LOs found in {obs_root}: {sorted(all_found)}. Please specify --lo."
    )


def _status_duration_sec(status):
    tbin = _maybe_float(_first_present(status, ("TBIN", "tsamp", "TSAMP"), None))
    pktstart = _first_present(status, ("PKTSTART",), None)
    pktstop = _first_present(status, ("PKTSTOP",), None)

    if tbin is None or pktstart is None or pktstop is None:
        return None

    return (int(pktstop) - int(pktstart)) * tbin


def _apply_exact_status_duration(mask_path, layout, status):
    duration_sec = _status_duration_sec(status)
    if duration_sec is None:
        return layout

    nblocks = nblocks_in_file(mask_path, layout)
    ntime = nblocks * int(layout["time_bins_per_block"])
    if ntime <= 0:
        return layout

    layout = dict(layout)
    layout["tbinsize_sec"] = float(duration_sec) / float(ntime)
    layout["duration_sec"] = float(duration_sec)
    return layout


def build_spliced_layout_from_statuses(statuses, kbsize=256):
    if not statuses:
        raise ValueError("No slice statuses provided")

    nants = int(statuses[0]["NANTS"])
    npol = int(statuses[0]["NPOL"])
    piperblk = int(statuses[0]["PIPERBLK"])

    slice_nchans = [infer_nchan_from_status(s) for s in statuses]
    total_nchan = int(sum(slice_nchans))

    meta = {
        "schan": 0,
        "nants": nants,
        "nchan": total_nchan,
        "npol": npol,
        "piperblk": piperblk,
    }
    layout = layout_from_metadata(meta, kbsize=kbsize)

    chan_bw = _maybe_float(
        _first_present(statuses[0], ("CHAN_BW", "CHAN_BW_MHZ", "FOFF", "foff"), None)
    )

    start_freqs = []
    for s in statuses:
        obsfreq = _maybe_float(_first_present(s, ("OBSFREQ", "FCH1", "fch1"), None))
        schan = int(s["SCHAN"])
        this_chan_bw = _maybe_float(
            _first_present(s, ("CHAN_BW", "CHAN_BW_MHZ", "FOFF", "foff"), None)
        )

        if obsfreq is not None and this_chan_bw is not None:
            start_freqs.append(obsfreq - schan * this_chan_bw)

    layout["schan"] = 0
    layout["fch1_mhz"] = start_freqs[0] if start_freqs else None
    layout["foff_mhz"] = chan_bw
    layout["tbinsize_sec"] = None
    layout = _attach_ant_names(layout, statuses[0])

    return layout


def build_single_layout_from_status(status, kbsize=256):
    nchan = infer_nchan_from_status(status)

    meta = {
        "schan": int(status["SCHAN"]),
        "nants": int(status["NANTS"]),
        "nchan": int(nchan),
        "npol": int(status["NPOL"]),
        "piperblk": int(status["PIPERBLK"]),
    }
    layout = layout_from_metadata(meta, kbsize=kbsize)

    layout["schan"] = int(status["SCHAN"])
    layout["fch1_mhz"] = _maybe_float(
        _first_present(status, ("OBSFREQ", "FCH1", "fch1"), None)
    )
    layout["foff_mhz"] = _maybe_float(
        _first_present(status, ("CHAN_BW", "CHAN_BW_MHZ", "FOFF", "foff"), None)
    )
    layout["tbinsize_sec"] = None
    layout = _attach_ant_names(layout, status)

    return layout


def resolve_status_path(mask_path, status_path=None):
    if status_path is not None:
        status_path = Path(status_path).expanduser()
        if not status_path.exists():
            raise ValueError(f"status_dump.json not found: {status_path}")
        return status_path

    mask_path = Path(mask_path).expanduser()
    sibling = mask_path.parent / "status_dump.json"
    if sibling.exists():
        return sibling

    return None


def resolve_ant_index(layout, ant):
    if ant is None:
        return None

    nants = int(layout["nants"])

    if isinstance(ant, int):
        if not (0 <= ant < nants):
            raise ValueError(f"Invalid antenna index {ant}, valid range is 0..{nants - 1}")
        return ant

    ant_str = str(ant).strip()
    if ant_str.isdigit():
        ant_idx = int(ant_str)
        if not (0 <= ant_idx < nants):
            raise ValueError(f"Invalid antenna index {ant_idx}, valid range is 0..{nants - 1}")
        return ant_idx

    ant_names = layout.get("ant_names", [])
    if ant_names:
        lowered = [x.lower() for x in ant_names]
        key = ant_str.lower()
        if key in lowered:
            return lowered.index(key)

    raise ValueError(f"Unknown antenna selector '{ant}'")


def ant_label_for_index(layout, ant_idx):
    ant_names = layout.get("ant_names", [])
    if ant_names and 0 <= ant_idx < len(ant_names):
        return ant_names[ant_idx]
    return str(ant_idx)


def resolve_kurtosis_input(input_path, lo=None, status_path=None, kbsize=256):
    input_path = Path(input_path).expanduser()

    if input_path.is_dir():
        obs_root = input_path
        lo_name, statuses = discover_lo_statuses(obs_root, lo=lo)
        layout = build_spliced_layout_from_statuses(statuses, kbsize=kbsize)
        mask_path = obs_root / f"{lo_name}_spliced.kurtosismask.bin"

        if not mask_path.exists():
            raise ValueError(f"Spliced kurtosis mask not found: {mask_path}")

        layout = _apply_exact_status_duration(mask_path, layout, statuses[0])
        return mask_path, lo_name, layout

    mask_path = input_path
    if not mask_path.exists():
        raise ValueError(f"Mask file not found: {mask_path}")

    lo_guess = infer_lo_from_mask_path(mask_path)

    if lo_guess is not None and "spliced" in mask_path.name:
        try:
            lo_name, statuses = discover_lo_statuses(mask_path.parent, lo=lo_guess)
            layout = build_spliced_layout_from_statuses(statuses, kbsize=kbsize)
            layout = _apply_exact_status_duration(mask_path, layout, statuses[0])
            return mask_path, lo_name, layout
        except ValueError:
            pass

    resolved_status = resolve_status_path(mask_path, status_path=status_path)
    if resolved_status is not None:
        status = read_status(resolved_status)
        layout = build_single_layout_from_status(status, kbsize=kbsize)
        layout = _apply_exact_status_duration(mask_path, layout, status)
        return mask_path, lo_guess, layout

    raise ValueError(
        "Could not resolve mask layout automatically. "
        "For spliced masks, place the file in the obs root with the slice directories. "
        "For single-slice masks, place it next to status_dump.json or pass --status."
    )


def load_spliced_mask_from_obs_dir(obs_root, lo=None, kbsize=256):
    mask_path, lo, layout = resolve_kurtosis_input(obs_root, lo=lo, kbsize=kbsize)
    mask = read_mask(mask_path, layout)
    return mask, lo, layout


def load_spliced_mask_from_file(mask_path, status_path=None, kbsize=256):
    mask_path, lo, layout = resolve_kurtosis_input(
        mask_path,
        status_path=status_path,
        kbsize=kbsize,
    )
    mask = read_mask(mask_path, layout)
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
    if mask_pol.ndim != 3:
        raise ValueError(
            f"Expected selected mask with shape (nant, nchan, ntime), got {mask_pol.shape}"
        )

    nant, nchan, ntime = mask_pol.shape

    if not (0 <= ant < nant):
        raise ValueError(f"Invalid antenna index {ant}, valid range is 0..{nant - 1}")

    t0, t1 = _normalize_slice_bounds(tstart, tend, ntime, axis_name="time")
    f0, f1 = _normalize_slice_bounds(fstart, fend, nchan, axis_name="channel")

    return mask_pol[ant, f0:f1, t0:t1]


def build_waterfall_axis_info(layout, tstart=None, fstart=None):
    channel_start = 0 if fstart is None else int(fstart)
    time_start = 0 if tstart is None else int(tstart)

    return {
        "channel_start": channel_start,
        "time_start": time_start,
        "schan": int(layout.get("schan", 0)),
        "f0_mhz": layout.get("fch1_mhz"),
        "df_mhz": layout.get("foff_mhz"),
        "dt_sec": layout.get("tbinsize_sec"),
    }


def stream_extract_waterfalls(
    mask_path,
    layout,
    pol,
    ants=None,
    tstart=None,
    tend=None,
    fstart=None,
    fend=None,
):
    mask_path = Path(mask_path)

    nant = int(layout["nants"])
    nchan = int(layout["nchan"])
    tbpb = int(layout["time_bins_per_block"])
    nblocks = nblocks_in_file(mask_path, layout)
    ntime = nblocks * tbpb

    if ants is None:
        ant_list = list(range(nant))
    else:
        ant_list = [int(a) for a in ants]

    for ant in ant_list:
        if not (0 <= ant < nant):
            raise ValueError(f"Invalid antenna index {ant}, valid range is 0..{nant - 1}")

    t0, t1 = _normalize_slice_bounds(tstart, tend, ntime, axis_name="time")
    f0, f1 = _normalize_slice_bounds(fstart, fend, nchan, axis_name="channel")

    out = np.empty((len(ant_list), f1 - f0, t1 - t0), dtype=np.uint8)
    pol_idx = pol_to_index(pol)
    ant_idx = np.asarray(ant_list, dtype=int)

    for block_index, block in enumerate(iter_mask_blocks(mask_path, layout)):
        block_t0 = block_index * tbpb
        block_t1 = block_t0 + tbpb

        overlap_t0 = max(t0, block_t0)
        overlap_t1 = min(t1, block_t1)
        if overlap_t0 >= overlap_t1:
            continue

        local_t0 = overlap_t0 - block_t0
        local_t1 = overlap_t1 - block_t0
        out_t0 = overlap_t0 - t0
        out_t1 = overlap_t1 - t0

        if pol_idx is None:
            block_view = np.logical_or(
                block[ant_idx, f0:f1, local_t0:local_t1, 0],
                block[ant_idx, f0:f1, local_t0:local_t1, 1],
            ).astype(np.uint8)
        else:
            block_view = block[ant_idx, f0:f1, local_t0:local_t1, pol_idx]

        out[:, :, out_t0:out_t1] = block_view

    return [out[i] for i in range(len(ant_list))], ant_list


def stream_extract_waterfall(
    mask_path,
    layout,
    pol,
    ant,
    tstart=None,
    tend=None,
    fstart=None,
    fend=None,
):
    data_list, ant_list = stream_extract_waterfalls(
        mask_path,
        layout,
        pol,
        ants=[ant],
        tstart=tstart,
        tend=tend,
        fstart=fstart,
        fend=fend,
    )
    return data_list[0], ant_list[0]


def _normalize_slice_bounds(start, stop, size, axis_name):
    s = 0 if start is None else int(start)
    e = size if stop is None else int(stop)

    if s < 0 or e < 0:
        raise ValueError(f"{axis_name} slice bounds must be non-negative")
    if s > e:
        raise ValueError(
            f"Invalid {axis_name} slice: start={s} must be <= stop={e}"
        )
    if e > size:
        raise ValueError(
            f"Invalid {axis_name} slice: stop={e} exceeds axis size {size}"
        )

    return s, e
