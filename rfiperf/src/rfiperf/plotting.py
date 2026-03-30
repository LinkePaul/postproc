import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def make_title(base, lo, pol, ant=None):
    title = f"{lo} pol={pol} {base}"
    if ant is not None:
        title += f" ant={ant}"
    return title


def save_zap_fraction_over_freq(values, out_path, lo, pol, ant=None):
    out_path = Path(out_path)
    x = np.arange(len(values))

    plt.figure()
    plt.plot(x, values)
    plt.xlabel("Channel")
    plt.ylabel("Zap fraction")
    plt.title(make_title("zap fraction over channel", lo, pol, ant=ant))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_zap_fraction_over_ant(values, out_path, lo, pol):
    out_path = Path(out_path)
    x = np.arange(len(values))

    plt.figure()
    plt.bar(x, values)
    plt.xlabel("Antenna index")
    plt.ylabel("Zap fraction")
    plt.title(make_title("zap fraction over antenna", lo, pol))
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_zap_fraction_over_time(values, out_path, lo, pol, ant=None):
    out_path = Path(out_path)
    x = np.arange(len(values))

    plt.figure()
    plt.plot(x, values)
    plt.xlabel("Time block")
    plt.ylabel("Zap fraction")
    plt.title(make_title("zap fraction over time block", lo, pol, ant=ant))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def _apply_waterfall_axes(ax, data, axis_info):
    data_t = np.asarray(data, dtype=float).T
    ntime, nchan = data_t.shape

    if axis_info is None:
        im = ax.imshow(
            data_t,
            interpolation="nearest",
            aspect="auto",
            origin="upper",
        )
        ax.set_xlabel("Channel")
        ax.set_ylabel("Time bin")
        return im

    channel_start = int(axis_info.get("channel_start", 0))
    time_start = int(axis_info.get("time_start", 0))
    schan = int(axis_info.get("schan", 0))
    f0_mhz = axis_info.get("f0_mhz", None)
    df_mhz = axis_info.get("df_mhz", None)
    dt_sec = axis_info.get("dt_sec", None)

    use_physical = (
        f0_mhz is not None and
        df_mhz is not None and
        dt_sec is not None
    )

    if not use_physical:
        im = ax.imshow(
            data_t,
            interpolation="nearest",
            aspect="auto",
            origin="upper",
        )
        ax.set_xlabel("Channel")
        ax.set_ylabel("Time bin")
        return im

    chan_abs0 = schan + channel_start
    chan_abs1 = chan_abs0 + nchan
    tbin0 = time_start
    tbin1 = time_start + ntime

    x0 = f0_mhz + chan_abs0 * df_mhz
    x1 = f0_mhz + chan_abs1 * df_mhz
    y0 = tbin0 * dt_sec
    y1 = tbin1 * dt_sec

    im = ax.imshow(
        data_t,
        interpolation="nearest",
        aspect="auto",
        origin="upper",
        extent=[x0, x1, y1, y0],
    )

    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Seconds")

    def freq_to_chan(freq):
        return (np.asarray(freq) - f0_mhz) / df_mhz

    def chan_to_freq(chan):
        return f0_mhz + np.asarray(chan) * df_mhz

    def sec_to_tbin(sec):
        return np.asarray(sec) / dt_sec

    def tbin_to_sec(tbin):
        return np.asarray(tbin) * dt_sec

    secax_x = ax.secondary_xaxis("top", functions=(freq_to_chan, chan_to_freq))
    secax_x.set_xlabel("Channel")

    secax_y = ax.secondary_yaxis("right", functions=(sec_to_tbin, tbin_to_sec))
    secax_y.set_ylabel("Time bin")

    return im


def save_waterfall(
    data,
    out_path,
    lo,
    pol,
    ant,
    tstart=None,
    tend=None,
    fstart=None,
    fend=None,
    axis_info=None,
):
    out_path = Path(out_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_waterfall_axes(ax, data, axis_info=axis_info)

    title = make_title("waterfall", lo, pol, ant=ant)

    crop_parts = []
    if tstart is not None or tend is not None:
        crop_parts.append(
            f"t={0 if tstart is None else tstart}:{'end' if tend is None else tend}"
        )
    if fstart is not None or fend is not None:
        crop_parts.append(
            f"f={0 if fstart is None else fstart}:{'end' if fend is None else fend}"
        )

    if crop_parts:
        title += " " + " ".join(crop_parts)

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_waterfall_grid(
    data_list,
    ant_list,
    out_path,
    lo,
    pol,
    tstart=None,
    tend=None,
    fstart=None,
    fend=None,
    ncols=4,
    axis_info=None,
):
    out_path = Path(out_path)

    nplots = len(data_list)
    if nplots == 0:
        raise ValueError("No waterfall panels to plot")

    ncols = max(1, min(ncols, nplots))
    nrows = math.ceil(nplots / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.8 * ncols, 2.8 * nrows),
        squeeze=False,
    )

    for ax, data, ant in zip(axes.flat, data_list, ant_list):
        _apply_waterfall_axes(ax, data, axis_info=axis_info)
        ax.set_title(f"ant={ant}", fontsize=9)

    for ax in axes.flat[nplots:]:
        ax.axis("off")

    title = make_title("waterfall", lo, pol)

    crop_parts = []
    if tstart is not None or tend is not None:
        crop_parts.append(
            f"t={0 if tstart is None else tstart}:{'end' if tend is None else tend}"
        )
    if fstart is not None or fend is not None:
        crop_parts.append(
            f"f={0 if fstart is None else fstart}:{'end' if fend is None else fend}"
        )

    if crop_parts:
        title += " " + " ".join(crop_parts)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_profile_plot(profile, out_path, title):
    out_path = Path(out_path)

    profile = np.asarray(profile, dtype=float)
    peak_bin = int(np.argmax(profile))
    center_bin = len(profile) // 2
    shift = center_bin - peak_bin
    profile = np.roll(profile, shift)
    profile = profile - np.median(profile)

    x = np.arange(len(profile))

    fig, ax = plt.subplots()
    ax.plot(x, profile)
    ax.set_xlabel("Profile bin")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_profile_overlay(profiles, labels, out_path, normalize=False):
    out_path = Path(out_path)

    fig, ax = plt.subplots(figsize=(8, 5))

    for profile, label in zip(profiles, labels):
        y = np.asarray(profile, dtype=float)

        peak_bin = int(np.argmax(y))
        center_bin = len(y) // 2
        shift = center_bin - peak_bin
        y = np.roll(y, shift)

        y = y - np.median(y)

        if normalize:
            ymax = np.max(np.abs(y))
            if ymax > 0:
                y = y / ymax
            else:
                y = y * 0.0

        x = np.arange(len(y))
        ax.plot(x, y, label=label)

    ax.set_xlabel("Profile bin")
    ax.set_ylabel(
        "Normalized amplitude" if normalize else "Baseline-subtracted amplitude"
    )
    ax.set_title("Folded pulse profile comparison")
    ax.grid(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=True,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
