#!/usr/bin/env python3
"""Plot KuaiRand scaling-ablation AUC against per-batch training GFLOPs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter


ROOT = Path(__file__).resolve().parent

SERIES_SIZES = {
    ("RankMixer", "TokenDim"): ("Tiny", "Small", "Mid", "Large", "Ultra"),
    ("RankMixer", "MaxLen"): ("Tiny", "Small", "Mid", "Large"),
    ("RankMixer", "NumLayers"): ("L2", "L3", "L4", "L6"),
    ("OneTrans", "TokenDim"): ("Tiny", "Small", "Mid", "Ultra"),
    ("OneTrans", "MaxLen"): ("Tiny", "Small", "Mid", "Large"),
    ("OneTrans", "NumLayers"): ("L2", "L4"),
}

# Forward + backward GFLOPs for one configured local training batch.
# Values are produced by benchmark/profile_scaling_ablation_flops.py.
GFLOPS_PER_BATCH = {
    ("RankMixer", "TokenDim"): (37.151047680, 75.088527360, 160.627163136, 370.359140352, 944.441917440),
    ("RankMixer", "MaxLen"): (204.365365248, 282.480082944, 438.709518336, 751.168389120),
    ("RankMixer", "NumLayers"): (370.359140352, 421.898747904, 473.438355456, 576.517570560),
    ("OneTrans", "TokenDim"): (127.806734336, 370.216534016, 1206.149709824, 16052.928905216),
    ("OneTrans", "MaxLen"): (1560.971051008, 2805.437825024, 5487.644901376, 11625.153167360),
    ("OneTrans", "NumLayers"): (17524.209483776, 21434.978533376),
}

# Mean test AUC over six KuaiRand tasks, extracted from Scaling_Law logs.
MEAN_AUC = {
    ("RankMixer", "TokenDim"): (
        0.846841166667,
        0.851045833333,
        0.853617666667,
        0.854320666667,
        0.854555666667,
    ),
    ("RankMixer", "MaxLen"): (
        0.858339166667,
        0.856860833333,
        0.853169166667,
        0.848408833333,
    ),
    ("RankMixer", "NumLayers"): (
        0.854320666667,
        0.852267166667,
        0.853585500000,
        0.851565000000,
    ),
    ("OneTrans", "TokenDim"): (
        0.857176500000,
        0.857941000000,
        0.861094166667,
        0.865491333333,
    ),
    ("OneTrans", "MaxLen"): (
        0.846084666667,
        0.847918333333,
        0.852960666667,
        0.857619500000,
    ),
    ("OneTrans", "NumLayers"): (
        0.852761333333,
        0.859458166667,
    ),
}

MODEL_ORDER = ("RankMixer", "OneTrans")
# Family hues stay blue / red; matched lightness steps keep the six colors harmonious.
SERIES_COLORS = {
    ("RankMixer", "TokenDim"): "#3A7CA5",  # mid steel blue
    ("RankMixer", "MaxLen"): "#7EB2D4",  # soft blue
    ("RankMixer", "NumLayers"): "#1E4F73",  # deep steel blue
    ("OneTrans", "TokenDim"): "#C45C4A",  # mid terracotta
    ("OneTrans", "MaxLen"): "#E19B8E",  # soft rose
    ("OneTrans", "NumLayers"): "#8C3F34",  # deep brick
}
MODEL_COLORS = {
    "RankMixer": SERIES_COLORS[("RankMixer", "TokenDim")],
    "OneTrans": SERIES_COLORS[("OneTrans", "TokenDim")],
}
SCALING_AXIS_ORDER = ("TokenDim", "MaxLen", "NumLayers")
# Layer uses a denser dotted pattern than default ":" for single-column width.
SCALING_AXIS_STYLE = {
    "TokenDim": {
        "label": "TokenDim",
        "linestyle": "-",
        "marker": "o",
        "zorder": 3,
        "size": 28,
        "edgewidth": 1.1,
    },
    "MaxLen": {
        "label": "SeqLen",
        "linestyle": "--",
        "marker": "s",
        "zorder": 4,
        "size": 26,
        "edgewidth": 1.1,
    },
    "NumLayers": {
        "label": "Layer",
        # Denser dotted pattern than default ":" for single-column width.
        "linestyle": (0, (1.15, 1.15)),
        "marker": "D",
        "zorder": 6,
        "size": 30,
        "edgewidth": 1.15,
    },
}


def _draw_tricolor_swatch(ax, x, y, width, height, colors, transform) -> None:
    """Draw a short-wide tricolor flag in axes coordinates."""
    n = len(colors)
    stripe_h = height / n
    for i, color in enumerate(colors):
        ax.add_patch(
            Rectangle(
                (x, y + height - (i + 1) * stripe_h),
                width,
                stripe_h,
                facecolor=color,
                edgecolor="none",
                transform=transform,
                clip_on=False,
                zorder=10,
            )
        )
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor="none",
            edgecolor="#444444",
            linewidth=0.35,
            transform=transform,
            clip_on=False,
            zorder=11,
        )
    )


def model_tricolor(model: str) -> tuple[str, str, str]:
    return tuple(SERIES_COLORS[(model, axis)] for axis in SCALING_AXIS_ORDER)


def _add_model_color_legend(ax) -> float:
    """Neat, centered RankMixer / OneTrans tricolor labels with equal swatches.

    Returns the left edge (axes fraction) of the legend block for alignment.
    """
    trans = ax.transAxes
    # Short-wide swatches ("矮胖"); height kept readable for 3 stripes.
    sw_w, sw_h = 0.080, 0.038
    y_sw = 0.935
    y_txt = y_sw + sw_h / 2.0
    gap = 0.010
    pair_gap = 0.036

    tmp_rm = ax.text(0, 0, "RankMixer", transform=trans, fontsize=6.4, alpha=0)
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    rm_w = tmp_rm.get_window_extent(renderer).transformed(trans.inverted()).width
    tmp_rm.remove()

    rm_entry_w = sw_w + gap + rm_w
    # Keep a tidy left margin instead of centering in the plot.
    x0 = 0.03

    _draw_tricolor_swatch(ax, x0, y_sw, sw_w, sw_h, model_tricolor("RankMixer"), trans)
    ax.text(
        x0 + sw_w + gap,
        y_txt,
        "RankMixer",
        transform=trans,
        fontsize=6.4,
        va="center",
        ha="left",
        color="black",
        clip_on=False,
        zorder=12,
    )

    x1 = x0 + rm_entry_w + pair_gap
    _draw_tricolor_swatch(ax, x1, y_sw, sw_w, sw_h, model_tricolor("OneTrans"), trans)
    ax.text(
        x1 + sw_w + gap,
        y_txt,
        "OneTrans",
        transform=trans,
        fontsize=6.4,
        va="center",
        ha="left",
        color="black",
        clip_on=False,
        zorder=12,
    )
    return x0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "scaling_auc_vs_gflops_kuairand.pdf",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=3.50,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=1.90,
        help="Figure height in inches; shorter default gives a flatter look.",
    )
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.weight": "normal",
            "axes.labelweight": "bold",
            "axes.linewidth": 1.05,
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "stix",
        }
    )


def format_log_tick(value: float, _position: int) -> str:
    exponent = int(round(np.log10(value)))
    return rf"$10^{{{exponent}}}$"


def plot(output_pdf: Path, fig_width: float, fig_height: float) -> None:
    configure_style()
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for model in MODEL_ORDER:
        for axis in SCALING_AXIS_ORDER:
            key = (model, axis)
            color = SERIES_COLORS[key]
            style = SCALING_AXIS_STYLE[axis]
            x = np.asarray(GFLOPS_PER_BATCH[key], dtype=np.float64)
            y = np.asarray(MEAN_AUC[key], dtype=np.float64)
            z = style["zorder"]

            ax.plot(
                x,
                y,
                color=color,
                linestyle=style["linestyle"],
                linewidth=1.25,
                marker="",
                zorder=z,
                solid_capstyle="round",
            )
            # RankMixer Layer-L2 coincides with TokenDim-Large; only that
            # diamond is translucent so the circle underneath still shows.
            if key == ("RankMixer", "NumLayers"):
                ax.scatter(
                    x[1:],
                    y[1:],
                    s=style["size"],
                    c=color,
                    marker=style["marker"],
                    linewidths=style["edgewidth"],
                    edgecolors="white",
                    zorder=z + 0.5,
                    clip_on=False,
                )
                ax.scatter(
                    x[:1],
                    y[:1],
                    s=style["size"],
                    c=color,
                    marker=style["marker"],
                    linewidths=style["edgewidth"],
                    edgecolors="white",
                    alpha=0.75,
                    zorder=z + 0.5,
                    clip_on=False,
                )
            else:
                ax.scatter(
                    x,
                    y,
                    s=style["size"],
                    c=color,
                    marker=style["marker"],
                    linewidths=style["edgewidth"],
                    edgecolors="white",
                    zorder=z + 0.5,
                    clip_on=False,
                )

    ax.set_xscale("log")
    ax.set_xlim(20, 35000)
    ax.set_ylim(0.844, 0.868)
    ax.xaxis.set_major_locator(FixedLocator([100, 1000, 10000]))
    ax.xaxis.set_major_formatter(FuncFormatter(format_log_tick))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_major_locator(FixedLocator([0.846, 0.850, 0.854, 0.858, 0.862, 0.866]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3f}"))

    ax.set_xlabel("Training GFLOPs per Batch", fontsize=8.5, labelpad=3.5)
    ax.set_ylabel("Avg AUC", fontsize=8.5, labelpad=3.5)
    ax.grid(axis="y", color="#C8C8C8", linestyle="--", linewidth=0.55, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", length=2.5, width=0.8, pad=2.0)

    legend_left = _add_model_color_legend(ax)

    axis_handles = []
    for axis in SCALING_AXIS_ORDER:
        style = SCALING_AXIS_STYLE[axis]
        axis_handles.append(
            Line2D(
                [0],
                [0],
                color="#333333",
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=1.15,
                markersize=4.6,
                markeredgewidth=1.0,
                markeredgecolor="white",
                markerfacecolor="#333333",
                label=style["label"],
                solid_capstyle="butt",
                dash_capstyle="butt",
            )
        )
    ax.legend(
        handles=axis_handles,
        loc="upper left",
        bbox_to_anchor=(legend_left, 0.875),
        ncol=1,
        frameon=False,
        fontsize=6.4,
        handlelength=2.8,
        handletextpad=0.45,
        borderaxespad=0.0,
        labelspacing=0.18,
        markerscale=0.9,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.05)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_pdf,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plot(args.output, args.fig_width, args.fig_height)
    for key, mean_auc in MEAN_AUC.items():
        print(
            f"{key[0]}-{key[1]}: "
            + ", ".join(
                f"{size}={auc:.6f}"
                for size, auc in zip(SERIES_SIZES[key], mean_auc)
            )
        )
    print(f"PDF: {args.output}")


if __name__ == "__main__":
    main()
