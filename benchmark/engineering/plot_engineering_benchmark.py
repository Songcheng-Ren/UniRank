#!/usr/bin/env python3
"""Generate UniRank engineering benchmark bar charts from measured CSV data."""

import argparse
import csv
import html
from pathlib import Path


WIDTH = 1100
LEFT = 330
RIGHT = 90
TOP = 110
ROW_HEIGHT = 78
BAR_HEIGHT = 36


def read_float(row, key):
    value = row.get(key, "").strip()
    return float(value) if value else None


def load_ratios(csv_path):
    memory = []
    training_time = []
    with csv_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            name = row["optimization"].strip()
            baseline_memory = read_float(row, "baseline_peak_memory_gib")
            optimized_memory = read_float(row, "optimized_peak_memory_gib")
            baseline_time = read_float(row, "baseline_training_time_seconds")
            optimized_time = read_float(row, "optimized_training_time_seconds")

            if baseline_memory is not None and optimized_memory is not None:
                if baseline_memory <= 0 or optimized_memory < 0:
                    raise ValueError(f"Invalid memory measurement for {name}")
                memory.append(
                    (name, (baseline_memory - optimized_memory) / baseline_memory * 100)
                )

            if baseline_time is not None and optimized_time is not None:
                if baseline_time <= 0 or optimized_time <= 0:
                    raise ValueError(f"Invalid training-time measurement for {name}")
                training_time.append((name, baseline_time, optimized_time))
    return memory, training_time


def write_bar_chart(rows, title, axis_label, output_path, color):
    if not rows:
        raise ValueError(f"No complete measurements available for {title}")

    min_value = min(0.0, min(value for _, value in rows))
    max_value = max(0.0, max(value for _, value in rows))
    padding = max(5.0, (max_value - min_value) * 0.12)
    axis_min = min_value - (padding if min_value < 0 else 0)
    axis_max = max_value + padding
    plot_width = WIDTH - LEFT - RIGHT
    height = TOP + len(rows) * ROW_HEIGHT + 90

    def x(value):
        return LEFT + (value - axis_min) / (axis_max - axis_min) * plot_width

    zero_x = x(0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{WIDTH / 2}" y="46" text-anchor="middle" font-family="Arial, sans-serif" font-size="28" font-weight="600" fill="#17233c">{html.escape(title)}</text>',
        f'<text x="{WIDTH / 2}" y="78" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" fill="#53627a">{html.escape(axis_label)}</text>',
    ]

    for tick_index in range(6):
        value = axis_min + (axis_max - axis_min) * tick_index / 5
        tick_x = x(value)
        parts.append(
            f'<line x1="{tick_x:.1f}" y1="{TOP - 8}" x2="{tick_x:.1f}" y2="{TOP + len(rows) * ROW_HEIGHT}" stroke="#dce2ea" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{tick_x:.1f}" y="{TOP + len(rows) * ROW_HEIGHT + 30}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#66758c">{value:.0f}%</text>'
        )

    parts.append(
        f'<line x1="{zero_x:.1f}" y1="{TOP - 8}" x2="{zero_x:.1f}" y2="{TOP + len(rows) * ROW_HEIGHT}" stroke="#8793a5" stroke-width="2"/>'
    )

    for index, (name, value) in enumerate(rows):
        center_y = TOP + index * ROW_HEIGHT + ROW_HEIGHT / 2
        value_x = x(value)
        bar_x = min(zero_x, value_x)
        bar_width = max(2, abs(value_x - zero_x))
        parts.append(
            f'<text x="{LEFT - 22}" y="{center_y + 6:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="17" fill="#26344d">{html.escape(name)}</text>'
        )
        parts.append(
            f'<rect x="{bar_x:.1f}" y="{center_y - BAR_HEIGHT / 2:.1f}" width="{bar_width:.1f}" height="{BAR_HEIGHT}" rx="5" fill="{color}"/>'
        )
        label_x = value_x + (12 if value >= 0 else -12)
        anchor = "start" if value >= 0 else "end"
        parts.append(
            f'<text x="{label_x:.1f}" y="{center_y + 6:.1f}" text-anchor="{anchor}" font-family="Arial, sans-serif" font-size="17" font-weight="600" fill="#17233c">{value:+.1f}%</text>'
        )

    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_grouped_time_chart(rows, output_path):
    if not rows:
        raise ValueError("No complete training-time measurements available")

    max_value = max(max(baseline, optimized) for _, baseline, optimized in rows)
    axis_max = max_value * 1.12
    plot_width = WIDTH - LEFT - RIGHT
    height = TOP + len(rows) * ROW_HEIGHT + 110

    def x(value):
        return LEFT + value / axis_max * plot_width

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{WIDTH / 2}" y="46" text-anchor="middle" font-family="Arial, sans-serif" font-size="28" font-weight="600" fill="#17233c">Training Time</text>',
        f'<text x="{WIDTH / 2}" y="78" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" fill="#53627a">End-to-end training duration in seconds (lower is better)</text>',
    ]

    for tick_index in range(6):
        value = axis_max * tick_index / 5
        tick_x = x(value)
        parts.append(
            f'<line x1="{tick_x:.1f}" y1="{TOP - 8}" x2="{tick_x:.1f}" y2="{TOP + len(rows) * ROW_HEIGHT}" stroke="#dce2ea" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{tick_x:.1f}" y="{TOP + len(rows) * ROW_HEIGHT + 30}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#66758c">{value:.0f}s</text>'
        )

    for index, (name, baseline, optimized) in enumerate(rows):
        center_y = TOP + index * ROW_HEIGHT + ROW_HEIGHT / 2
        parts.append(
            f'<text x="{LEFT - 22}" y="{center_y + 6:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="17" fill="#26344d">{html.escape(name)}</text>'
        )
        for value, offset, color in (
            (baseline, -BAR_HEIGHT / 2 - 2, "#aab4c4"),
            (optimized, 2, "#24a58a"),
        ):
            bar_y = center_y + offset
            bar_width = max(2, x(value) - LEFT)
            parts.append(
                f'<rect x="{LEFT}" y="{bar_y:.1f}" width="{bar_width:.1f}" height="{BAR_HEIGHT / 2}" rx="4" fill="{color}"/>'
            )
            parts.append(
                f'<text x="{x(value) + 9:.1f}" y="{bar_y + BAR_HEIGHT / 2 - 2:.1f}" text-anchor="start" font-family="Arial, sans-serif" font-size="14" fill="#17233c">{value:.1f}s</text>'
            )

    legend_y = TOP + len(rows) * ROW_HEIGHT + 70
    parts.extend([
        f'<rect x="{WIDTH / 2 - 150}" y="{legend_y - 13}" width="18" height="13" rx="2" fill="#aab4c4"/>',
        f'<text x="{WIDTH / 2 - 124}" y="{legend_y}" font-family="Arial, sans-serif" font-size="15" fill="#26344d">Baseline</text>',
        f'<rect x="{WIDTH / 2 + 28}" y="{legend_y - 13}" width="18" height="13" rx="2" fill="#24a58a"/>',
        f'<text x="{WIDTH / 2 + 54}" y="{legend_y}" font-family="Arial, sans-serif" font-size="15" fill="#26344d">Optimized</text>',
        "</svg>",
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate memory-saving and training-time SVG charts"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).with_name("engineering_benchmark.csv"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "assets" / "figures",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    memory, training_time = load_ratios(args.csv)
    write_bar_chart(
        memory,
        "Peak GPU Memory Saving",
        "Reduction relative to the matched baseline (higher is better)",
        args.output_dir / "engineering_memory_saving.svg",
        "#4776d0",
    )
    write_grouped_time_chart(
        training_time,
        args.output_dir / "engineering_training_time.svg",
    )
    print(f"Charts written to {args.output_dir}")


if __name__ == "__main__":
    main()
