import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, LogFormatterMathtext

def get_files_data(mdl_dir):
    file_labels, average_src, average_drc, average_mdl = [], [], [], []

    if not os.path.exists(mdl_dir):
        raise FileNotFoundError(f"Directory not found: {mdl_dir}")

    file_paths = sorted([os.path.join(mdl_dir, f) for f in os.listdir(mdl_dir) if f.endswith(".json")])
    if not file_paths:
        print(f"No JSON files found in {mdl_dir}")
        return file_labels, average_src, average_drc, average_mdl

    for file_path in file_paths:
        total_src = total_drc = total_mdl =count = 0
        data_list = []
        with open(file_path, "r") as f:
            try:
                for line in f:
                    data_list.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON file: {file_path}")
                continue

            for data in data_list:
                total_src += data.get("SRC", 0)
                total_drc += data.get("DRC", 0)
                total_mdl += data.get("MDL", 0)
                count += 1

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        parts = base_name.split("_")
        label = parts[0] + "_" + parts[2] if len(parts) > 2 else base_name
        file_labels.append(label)

        average_src.append(total_src / count if count else 0)
        average_drc.append(total_drc / count if count else 0)
        average_mdl.append(total_mdl / count if count else 0)
    return file_labels, average_src, average_drc, average_mdl


def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2e}",
            ha='center', va='bottom',
            fontsize=9, color='black'
        )


def plot_data(file_labels, average_src, average_drc, average_mdl):
    if not file_labels:
        print("No data to plot.")
        return

    x = np.arange(len(file_labels))
    width = 0.25

    plt.figure(figsize=(10, 6))
    bars_src = plt.bar(x - width, average_src, width, label='Average SRC', color='skyblue')
    bars_drc = plt.bar(x, average_drc, width, label='Average DRC', color='salmon')
    bars_mdl = plt.bar(x + width, average_mdl, width, label='Average MDL', color='lightgreen')

    plt.xlabel("Files")
    plt.ylabel("MDL Cost")
    plt.title("Average SRC, DRC, and MDL per File")
    plt.xticks(x, file_labels, rotation=45, ha="right")
    plt.legend()

    plt.yscale("log")
    plt.gca().yaxis.set_major_locator(LogLocator(base=10))
    plt.gca().yaxis.set_major_formatter(LogFormatterMathtext())

    add_labels(bars_src)
    add_labels(bars_drc)
    add_labels(bars_mdl)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot average SRC, DRC, and MDL values from JSON files.")
    parser.add_argument(
        "--mdl_dir",
        type=str,
        default=".",
        help="Directory containing JSON files with SRC, DRC, and MDL data."
    )
    args = parser.parse_args()

    file_labels, avg_src, avg_drc, avg_mdl = get_files_data(args.mdl_dir)
    plot_data(file_labels, avg_src, avg_drc, avg_mdl)
    
if __name__ == "__main__":
    main()
