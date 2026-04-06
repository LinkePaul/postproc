from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

MASK_CMAP = ListedColormap(["#440154", "#FDE725"])


def _save(fig, out_path):
    out_path = Path(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _line_plot(x, y, out_path, *, xlabel, ylabel, title, kind="plot"):
    fig, ax = plt.subplots()
    getattr(ax, kind)(x, y) if kind == "bar" else ax.plot(x, y)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(True, axis="y" if kind == "bar" else "both")
    _save(fig, out_path)


def save_zap_fraction_over_freq(values, out_path, lo, pol, ant=None):
    title = f"{lo} pol={pol} zap fraction over frequency" + (f" ant={ant}" if ant is not None else "")
    _line_plot(np.arange(len(values)), values, out_path, xlabel="Frequency channel", ylabel="Zap fraction", title=title)


def save_zap_fraction_over_ant(values, out_path, lo, pol):
    _line_plot(np.arange(len(values)), values, out_path, xlabel="Antenna index", ylabel="Zap fraction", title=f"{lo} pol={pol} zap fraction over antenna", kind="bar")


def save_zap_fraction_over_time(values, out_path, lo, pol, ant=None):
    title = f"{lo} pol={pol} zap fraction over time" + (f" ant={ant}" if ant is not None else "")
    _line_plot(np.arange(len(values)), values, out_path, xlabel="Time block", ylabel="Zap fraction", title=title)


def _center_profile(profile):
    y = np.asarray(profile, dtype=float)
    y = np.roll(y, len(y) // 2 - int(np.argmax(y)))
    return y - np.median(y)


def save_profile_plot(profile, out_path, title):
    y = _center_profile(profile)
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(y)), y)
    ax.set(xlabel="Profile bin", ylabel="Value", title=title)
    ax.grid(True)
    _save(fig, out_path)


def save_profile_overlay(profiles, labels, out_path, normalize=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    for profile, label in zip(profiles, labels):
        y = _center_profile(profile)
        if normalize and np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        ax.plot(np.arange(len(y)), y, label=label)
    ax.set(
        xlabel="Profile bin",
        ylabel="Normalized amplitude" if normalize else "Baseline-subtracted amplitude",
        title="Folded pulse profile comparison",
    )
    ax.grid(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=True)
    _save(fig, out_path)


def _apply_waterfall_axes(ax, data, axis_info):
    data_t = np.asarray(data, dtype=float).T
    ntime, nchan = data_t.shape
    imshow_kwargs = dict(interpolation="nearest", aspect="auto", origin="upper", vmin=0, vmax=1, cmap=MASK_CMAP)

    if axis_info is None:
        im = ax.imshow(data_t, **imshow_kwargs)
        ax.set(xlabel="Channel", ylabel="Time bin")
        return im

    schan = int(axis_info.get("schan", 0))
    c0 = int(axis_info.get("channel_start", 0))
    t0 = int(axis_info.get("time_start", 0))
    f0 = axis_info.get("f0_mhz")
    df = axis_info.get("df_mhz")
    dt = axis_info.get("dt_sec")
    if f0 is None or df is None or dt is None:
        im = ax.imshow(data_t, **imshow_kwargs)
        ax.set(xlabel="Channel", ylabel="Time bin")
        return im

    x0, x1 = f0 + c0 * df, f0 + (c0 + nchan) * df
    y0, y1 = t0 * dt, (t0 + ntime) * dt
    im = ax.imshow(data_t, extent=[x0, x1, y1, y0], **imshow_kwargs)
    ax.set(xlabel="Frequency [MHz]", ylabel="Seconds")

    secax_x = ax.secondary_xaxis(
        "top",
        functions=(
            lambda freq: schan + (np.asarray(freq) - f0) / df,
            lambda chan: f0 + (np.asarray(chan) - schan) * df,
        ),
    )
    secax_x.set_xlabel("Channel")

    secax_y = ax.secondary_yaxis(
        "right",
        functions=(lambda sec: np.asarray(sec) / dt, lambda tbin: np.asarray(tbin) * dt),
    )
    secax_y.set_ylabel("Time bin")
    return im


def _waterfall_title(prefix, lo, pol, ant=None, tstart=None, tend=None):
    title = f"{prefix} {lo} pol={pol}"
    if ant is not None:
        title += f" ant={ant}"
    if tstart is not None or tend is not None:
        title += f" t={0 if tstart is None else tstart}:{'end' if tend is None else tend}"
    return title


def save_waterfall(data, out_path, lo, pol, ant=None, tstart=None, tend=None, fstart=None, fend=None, axis_info=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    _apply_waterfall_axes(ax, data, axis_info)
    ax.set_title(_waterfall_title("mask waterfall", lo, pol, ant=ant, tstart=tstart, tend=tend))
    _save(fig, out_path)


def save_waterfall_grid(data_list, ant_labels, out_path, lo, pol, tstart=None, tend=None, fstart=None, fend=None, axis_info=None):
    n = len(data_list)
    ncols = min(3, n)
    nrows = ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax, data, label in zip(axes, data_list, ant_labels):
        _apply_waterfall_axes(ax, data, axis_info)
        ax.set_title(f"ant={label}")
    for ax in axes[n:]:
        ax.remove()
    fig.suptitle(_waterfall_title("mask waterfall", lo, pol, tstart=tstart, tend=tend), y=0.98)
    _save(fig, out_path)
