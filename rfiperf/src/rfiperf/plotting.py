from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import math


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
):
    out_path = Path(out_path)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.imshow(
        np.asarray(data, dtype=float).T,
        interpolation="nearest",
        aspect="auto",
        origin="upper",
    )
    ax.set_xlabel("Channel")
    ax.set_ylabel("Time bin")

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
        figsize=(3.2 * ncols, 2.4 * nrows),
        squeeze=False,
    )

    for ax, data, ant in zip(axes.flat, data_list, ant_list):
        ax.imshow(
            np.asarray(data, dtype=float).T,
            interpolation="nearest",
            aspect="auto",
            origin="upper",
        )
        ax.set_title(f"ant={ant}", fontsize=9)
        ax.set_xlabel("Channel")
        ax.set_ylabel("Time bin")

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
    fig.tight_layout(rect=(0, 0, 1, 0.97))
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

        # baseline-center for plotting
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
