from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_zap_fraction_over_freq(values, out_path, lo, pol, ant=None):
    out_path = Path(out_path)
    x = np.arange(len(values))

    plt.figure()
    plt.plot(x, values)
    plt.xlabel("Frequency channel")
    plt.ylabel("Zap fraction")
    title = f"{lo} pol={pol} zap fraction over frequency"
    if ant is not None:
        title += f" ant={ant}"
    plt.title(title)
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
    plt.title(f"{lo} pol={pol} zap fraction over antenna")
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
    title = f"{lo} pol={pol} zap fraction over time"
    if ant is not None:
        title += f" ant={ant}"
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_profile_plot(profile, out_path, title):
    out_path = Path(out_path)
    x = np.arange(len(profile))

    plt.figure()
    plt.plot(x, profile)
    plt.xlabel("Profile bin")
    plt.ylabel("Value")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_profile_overlay(profiles, labels, out_path, normalize=False):
    out_path = Path(out_path)

    fig, ax = plt.subplots(figsize=(8, 5))

    for profile, label in zip(profiles, labels):
        y = np.asarray(profile, dtype=float)
        if normalize:
            ymin = y.min()
            ymax = y.max()
            if ymax > ymin:
                y = (y - ymin) / (ymax - ymin)
            else:
                y = y * 0.0
        x = np.arange(len(y))
        ax.plot(x, y, label=label)

    ax.set_xlabel("Profile bin")
    ax.set_ylabel("Normalized value" if normalize else "Value")
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
