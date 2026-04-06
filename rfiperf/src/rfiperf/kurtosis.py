from pathlib import Path

import numpy as np

from postproc_common.kurtio import iter_mask_blocks, layout_from_metadata, nblocks_in_file, read_mask
from postproc_common.metadata import read_status


# -------- small helpers --------
def normalize_lo(lo):
    lo = lo.strip()
    return lo if lo.startswith("Lo") else "Lo" + lo.upper()


def infer_lo_from_mask_path(mask_path):
    name = Path(mask_path).name
    return next((p for p in ("LoA", "LoB", "LoC", "LoD") if name.startswith(p)), None)


def _first_present(mapping, keys, default=None):
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return default


def _maybe_float(value):
    return None if value is None else float(value)


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
            parts.extend(x.strip() for x in str(value).split(",") if x.strip())
    if not parts and status.get("ANTNAMES"):
        parts.extend(x.strip() for x in str(status["ANTNAMES"]).split(",") if x.strip())
    return parts


def _display_ant_name(name):
    return name[:-1] if name and len(name) >= 2 and name[-1].isalpha() and name[-1].isupper() else name


def _attach_ant_names(layout, status):
    ant_names_raw = _parse_ant_names(status)
    if not ant_names_raw:
        return layout
    out = dict(layout)
    out["ant_names_raw"] = ant_names_raw
    out["ant_names"] = [_display_ant_name(x) for x in ant_names_raw]
    return out


# -------- discovery / metadata --------
def discover_lo_statuses(obs_root, lo=None):
    obs_root = Path(obs_root)
    lo_list = [normalize_lo(lo)] if lo is not None else sorted(
        {p.name.split(".")[0] for p in obs_root.iterdir() if p.is_dir() and p.name.startswith("Lo") and ".C" in p.name}
    )
    if not lo_list:
        raise ValueError(f"No LO slice directories found in {obs_root}")

    found = {}
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
            found[lo_name] = statuses

    if lo is not None:
        lo_name = normalize_lo(lo)
        if lo_name not in found:
            raise ValueError(f"No status_dump.json files found for {lo_name} in {obs_root}")
        return lo_name, found[lo_name]
    if len(found) == 1:
        lo_name = next(iter(found))
        return lo_name, found[lo_name]
    raise ValueError(f"Multiple LOs found in {obs_root}: {sorted(found)}. Please specify --lo.")


def _status_duration_sec(status):
    tbin = _maybe_float(_first_present(status, ("TBIN", "tsamp", "TSAMP")))
    pktstart = _first_present(status, ("PKTSTART",))
    pktstop = _first_present(status, ("PKTSTOP",))
    if tbin is None or pktstart is None or pktstop is None:
        return None
    return (int(pktstop) - int(pktstart)) * tbin


def _slice_channel0_freq_mhz(status):
    # Prefer explicit channel-0 frequency. Otherwise interpret OBSFREQ as slice center.
    fch1 = _maybe_float(_first_present(status, ("FCH1", "fch1")))
    if fch1 is not None:
        return fch1
    obsfreq = _maybe_float(_first_present(status, ("OBSFREQ",)))
    chan_bw = _maybe_float(_first_present(status, ("CHAN_BW", "CHAN_BW_MHZ", "FOFF", "foff")))
    if obsfreq is None or chan_bw is None:
        return None
    return obsfreq - 0.5 * (infer_nchan_from_status(status) - 1) * chan_bw


def _global_channel0_freq_mhz_from_slice(status):
    local_f0 = _slice_channel0_freq_mhz(status)
    chan_bw = _maybe_float(_first_present(status, ("CHAN_BW", "CHAN_BW_MHZ", "FOFF", "foff")))
    if local_f0 is None or chan_bw is None:
        return None
    return local_f0 - int(status["SCHAN"]) * chan_bw


def _apply_exact_status_duration(mask_path, layout, status):
    # Keep the old time logic on purpose. This is the pre-time-fix state.
    duration_sec = _status_duration_sec(status)
    if duration_sec is None:
        return layout
    nblocks = nblocks_in_file(mask_path, layout)
    ntime = nblocks * int(layout["time_bins_per_block"])
    if ntime <= 0:
        return layout
    out = dict(layout)
    out["tbinsize_sec"] = float(duration_sec) / float(ntime)
    out["duration_sec"] = float(duration_sec)
    return out


def build_spliced_layout_from_statuses(statuses, kbsize=256):
    if not statuses:
        raise ValueError("No slice statuses provided")
    first = statuses[0]
    meta = {
        "schan": 0,
        "nants": int(first["NANTS"]),
        "nchan": sum(infer_nchan_from_status(s) for s in statuses),
        "npol": int(first["NPOL"]),
        "piperblk": int(first["PIPERBLK"]),
    }
    layout = layout_from_metadata(meta, kbsize=kbsize)
    chan_bw = _maybe_float(_first_present(first, ("CHAN_BW", "CHAN_BW_MHZ", "FOFF", "foff")))
    global_f0 = next((_global_channel0_freq_mhz_from_slice(s) for s in statuses if _global_channel0_freq_mhz_from_slice(s) is not None), None)
    layout.update({"schan": 0, "fch1_mhz": global_f0, "foff_mhz": chan_bw, "tbinsize_sec": None})
    return _attach_ant_names(layout, first)


def build_single_layout_from_status(status, kbsize=256):
    meta = {
        "schan": int(status["SCHAN"]),
        "nants": int(status["NANTS"]),
        "nchan": infer_nchan_from_status(status),
        "npol": int(status["NPOL"]),
        "piperblk": int(status["PIPERBLK"]),
    }
    layout = layout_from_metadata(meta, kbsize=kbsize)
    layout.update({
        "schan": int(status["SCHAN"]),
        "fch1_mhz": _slice_channel0_freq_mhz(status),
        "foff_mhz": _maybe_float(_first_present(status, ("CHAN_BW", "CHAN_BW_MHZ", "FOFF", "foff"))),
        "tbinsize_sec": None,
    })
    return _attach_ant_names(layout, status)


def resolve_status_path(mask_path, status_path=None):
    if status_path is not None:
        status_path = Path(status_path).expanduser()
        if not status_path.exists():
            raise ValueError(f"status_dump.json not found: {status_path}")
        return status_path
    sibling = Path(mask_path).expanduser().parent / "status_dump.json"
    return sibling if sibling.exists() else None


def resolve_ant_index(layout, ant):
    if ant is None:
        return None
    nants = int(layout["nants"])
    if isinstance(ant, int) or str(ant).strip().isdigit():
        ant = int(ant)
        if not (0 <= ant < nants):
            raise ValueError(f"Invalid antenna index {ant}, valid range is 0..{nants - 1}")
        return ant
    ant_names = [x.lower() for x in layout.get("ant_names", [])]
    key = str(ant).strip().lower()
    if key in ant_names:
        return ant_names.index(key)
    raise ValueError(f"Unknown antenna selector '{ant}'")


def ant_label_for_index(layout, ant_idx):
    ant_names = layout.get("ant_names", [])
    return ant_names[ant_idx] if ant_names and 0 <= ant_idx < len(ant_names) else str(ant_idx)


def resolve_kurtosis_input(input_path, lo=None, status_path=None, kbsize=256):
    input_path = Path(input_path).expanduser()
    if input_path.is_dir():
        lo_name, statuses = discover_lo_statuses(input_path, lo=lo)
        layout = _apply_exact_status_duration(
            input_path / f"{lo_name}_spliced.kurtosismask.bin",
            build_spliced_layout_from_statuses(statuses, kbsize=kbsize),
            statuses[0],
        )
        mask_path = input_path / f"{lo_name}_spliced.kurtosismask.bin"
        if not mask_path.exists():
            raise ValueError(f"Spliced kurtosis mask not found: {mask_path}")
        return mask_path, lo_name, layout

    mask_path = input_path
    if not mask_path.exists():
        raise ValueError(f"Mask file not found: {mask_path}")
    lo_guess = infer_lo_from_mask_path(mask_path)

    if lo_guess and "spliced" in mask_path.name:
        try:
            lo_name, statuses = discover_lo_statuses(mask_path.parent, lo=lo_guess)
            layout = _apply_exact_status_duration(mask_path, build_spliced_layout_from_statuses(statuses, kbsize=kbsize), statuses[0])
            return mask_path, lo_name, layout
        except ValueError:
            pass

    resolved_status = resolve_status_path(mask_path, status_path=status_path)
    if resolved_status is not None:
        status = read_status(resolved_status)
        layout = _apply_exact_status_duration(mask_path, build_single_layout_from_status(status, kbsize=kbsize), status)
        return mask_path, lo_guess, layout

    raise ValueError(
        "Could not resolve mask layout automatically. "
        "For spliced masks, place the file in the obs root with the slice directories. "
        "For single-slice masks, place it next to status_dump.json or pass --status."
    )


# -------- loading / selection --------
def load_spliced_mask_from_obs_dir(obs_root, lo=None, kbsize=256):
    mask_path, lo, layout = resolve_kurtosis_input(obs_root, lo=lo, kbsize=kbsize)
    return read_mask(mask_path, layout), lo, layout


def load_spliced_mask_from_file(mask_path, status_path=None, kbsize=256):
    mask_path, lo, layout = resolve_kurtosis_input(mask_path, status_path=status_path, kbsize=kbsize)
    return read_mask(mask_path, layout), lo, layout


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
    return mask_pol if ant is None else mask_pol[ant : ant + 1]


def summary_stats(mask_pol, ant=None):
    data = select_ant(mask_pol, ant=ant)
    total_cells = data.size
    zapped_cells = int(data.sum())
    return {
        "ant": ant,
        "nants_used": int(data.shape[0]),
        "nchans": int(data.shape[1]),
        "ntime": int(data.shape[2]),
        "zapped_cells": zapped_cells,
        "total_cells": int(total_cells),
        "zap_fraction": zapped_cells / total_cells if total_cells else 0.0,
    }


def zap_fraction_over_freq(mask_pol, ant=None):
    return select_ant(mask_pol, ant=ant).mean(axis=(0, 2))


def zap_fraction_over_ant(mask_pol):
    return mask_pol.mean(axis=(1, 2))


def zap_fraction_over_time(mask_pol, ant=None):
    return select_ant(mask_pol, ant=ant).mean(axis=(0, 1))


# -------- waterfall helpers --------
def _normalize_slice_bounds(start, stop, size, axis_name):
    s = 0 if start is None else int(start)
    e = size if stop is None else int(stop)
    if s < 0 or e < 0:
        raise ValueError(f"{axis_name} slice bounds must be non-negative")
    if s > e:
        raise ValueError(f"Invalid {axis_name} slice: start={s} must be <= stop={e}")
    if e > size:
        raise ValueError(f"Invalid {axis_name} slice: stop={e} exceeds axis size {size}")
    return s, e


def extract_waterfall(mask_pol, ant, tstart=None, tend=None, fstart=None, fend=None):
    if mask_pol.ndim != 3:
        raise ValueError(f"Expected selected mask with shape (nant, nchan, ntime), got {mask_pol.shape}")
    nant, nchan, ntime = mask_pol.shape
    if not (0 <= ant < nant):
        raise ValueError(f"Invalid antenna index {ant}, valid range is 0..{nant - 1}")
    t0, t1 = _normalize_slice_bounds(tstart, tend, ntime, axis_name="time")
    f0, f1 = _normalize_slice_bounds(fstart, fend, nchan, axis_name="channel")
    return mask_pol[ant, f0:f1, t0:t1]


def build_waterfall_axis_info(layout, tstart=None, fstart=None):
    return {
        "channel_start": 0 if fstart is None else int(fstart),
        "time_start": 0 if tstart is None else int(tstart),
        "schan": int(layout.get("schan", 0)),
        "f0_mhz": layout.get("fch1_mhz"),
        "df_mhz": layout.get("foff_mhz"),
        "dt_sec": layout.get("tbinsize_sec"),
    }


def stream_extract_waterfalls(mask_path, layout, pol, ants=None, tstart=None, tend=None, fstart=None, fend=None):
    mask_path = Path(mask_path)
    nant, nchan = int(layout["nants"]), int(layout["nchan"])
    tbpb = int(layout["time_bins_per_block"])
    ntime = nblocks_in_file(mask_path, layout) * tbpb
    ant_list = list(range(nant)) if ants is None else [int(a) for a in ants]
    for ant in ant_list:
        if not (0 <= ant < nant):
            raise ValueError(f"Invalid antenna index {ant}, valid range is 0..{nant - 1}")

    t0, t1 = _normalize_slice_bounds(tstart, tend, ntime, axis_name="time")
    f0, f1 = _normalize_slice_bounds(fstart, fend, nchan, axis_name="channel")
    out = np.empty((len(ant_list), f1 - f0, t1 - t0), dtype=np.uint8)
    ant_idx = np.asarray(ant_list, dtype=int)
    pol_idx = pol_to_index(pol)

    for block_index, block in enumerate(iter_mask_blocks(mask_path, layout)):
        b0, b1 = block_index * tbpb, (block_index + 1) * tbpb
        o0, o1 = max(t0, b0), min(t1, b1)
        if o0 >= o1:
            continue
        lt0, lt1 = o0 - b0, o1 - b0
        ot0, ot1 = o0 - t0, o1 - t0
        if pol_idx is None:
            view = np.logical_or(block[ant_idx, f0:f1, lt0:lt1, 0], block[ant_idx, f0:f1, lt0:lt1, 1]).astype(np.uint8)
        else:
            view = block[ant_idx, f0:f1, lt0:lt1, pol_idx]
        out[:, :, ot0:ot1] = view

    return [out[i] for i in range(len(ant_list))], ant_list


def stream_extract_waterfall(mask_path, layout, pol, ant, tstart=None, tend=None, fstart=None, fend=None):
    data_list, ant_list = stream_extract_waterfalls(
        mask_path, layout, pol, ants=[ant], tstart=tstart, tend=tend, fstart=fstart, fend=fend
    )
    return data_list[0], ant_list[0]
