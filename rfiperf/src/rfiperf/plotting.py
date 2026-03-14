from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def make_title(base, lo, pol, ant=None):
    title = f"{base} | {lo} | pol={pol}"
    if ant is not None:
        title += f" | ant={ant}"
    return title


def save_zap_fraction_over_freq(values, out_path, lo, pol, ant=None):
    out_path = Path(out_path)

    x = np.arange(len(values))

    plt.figure()
    plt.plot(x, values)
    plt.xlabel("Channel")
    plt.ylabel("Zap fraction")
    plt.title(make_title("Zap fraction over frequency", lo, pol, ant=ant))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_zap_fraction_over_ant(values, out_path, lo, pol):
    out_path = Path(out_path)

    x = np.arange(len(values))

    plt.figure()
    plt.bar(x, values)
    plt.xlabel("Antenna")
    plt.ylabel("Zap fraction")
    plt.title(make_title("Zap fraction over antenna", lo, pol))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_zap_fraction_over_time(values, out_path, lo, pol, ant=None):
    out_path = Path(out_path)

    x = np.arange(len(values))

    plt.figure()
    plt.plot(x, values)
    plt.xlabel("Time bin")
    plt.ylabel("Zap fraction")
    plt.title(make_title("Zap fraction over time", lo, pol, ant=ant))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    