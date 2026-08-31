import csv
import os
import matplotlib.pyplot as plt
from datetime import datetime

LOG_PATH = os.path.join("output", "session_log.csv")
CHART_DIR = os.path.join("output", "charts")

os.makedirs(CHART_DIR, exist_ok=True)


def load_log(path):
    elapsed, ear, mar, perclos, pitch, fusion, blinks, status = ([] for _ in range(8))

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            elapsed.append(float(row["elapsed_sec"]))
            ear.append(float(row["ear"]))
            mar.append(float(row["mar"]))
            perclos.append(float(row["perclos"]))
            pitch.append(float(row["pitch"]))
            fusion.append(float(row.get("fusion_score", 0)))
            blinks.append(int(row["blink_count"]))
            status.append(row["status"])

    return {
        "elapsed": elapsed, "ear": ear, "mar": mar, "perclos": perclos,
        "pitch": pitch, "fusion": fusion, "blinks": blinks, "status": status
    }


def drowsy_regions(elapsed, status):
    """Return list of (start, end) time ranges where status == DROWSY, for shading on charts."""
    regions = []
    start = None
    for t, s in zip(elapsed, status):
        if s == "DROWSY" and start is None:
            start = t
        elif s != "DROWSY" and start is not None:
            regions.append((start, t))
            start = None
    if start is not None:
        regions.append((start, elapsed[-1]))
    return regions


def shade_drowsy(ax, regions):
    for start, end in regions:
        ax.axvspan(start, end, color="red", alpha=0.15)


def plot_ear(data, out_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data["elapsed"], data["ear"], color="#1976d2", linewidth=1)
    shade_drowsy(ax, drowsy_regions(data["elapsed"], data["status"]))
    ax.set_title("Eye Aspect Ratio (EAR) Over Session")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("EAR")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_perclos(data, out_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data["elapsed"], data["perclos"], color="#e64a19", linewidth=1)
    shade_drowsy(ax, drowsy_regions(data["elapsed"], data["status"]))
    ax.set_title("PERCLOS Over Session")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("PERCLOS (%)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fusion(data, out_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data["elapsed"], data["fusion"], color="#7b1fa2", linewidth=1)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Alert threshold")
    shade_drowsy(ax, drowsy_regions(data["elapsed"], data["status"]))
    ax.set_title("Fusion Drowsiness Score Over Session")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Fusion Score")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_combined(data, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    regions = drowsy_regions(data["elapsed"], data["status"])

    axes[0].plot(data["elapsed"], data["ear"], color="#1976d2")
    axes[0].set_ylabel("EAR")
    shade_drowsy(axes[0], regions)

    axes[1].plot(data["elapsed"], data["perclos"], color="#e64a19")
    axes[1].set_ylabel("PERCLOS (%)")
    shade_drowsy(axes[1], regions)

    axes[2].plot(data["elapsed"], data["fusion"], color="#7b1fa2")
    axes[2].axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axes[2].set_ylabel("Fusion Score")
    axes[2].set_xlabel("Time (seconds)")
    shade_drowsy(axes[2], regions)

    fig.suptitle("Driver Drowsiness Session Overview")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    if not os.path.exists(LOG_PATH):
        print(f"No log file found at {LOG_PATH}. Run a session first.")
        return

    data = load_log(LOG_PATH)
    if not data["elapsed"]:
        print("Log file is empty. Run a session first.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plot_ear(data, os.path.join(CHART_DIR, f"ear_{timestamp}.png"))
    plot_perclos(data, os.path.join(CHART_DIR, f"perclos_{timestamp}.png"))
    plot_fusion(data, os.path.join(CHART_DIR, f"fusion_{timestamp}.png"))
    plot_combined(data, os.path.join(CHART_DIR, f"combined_overview_{timestamp}.png"))

    print(f"Charts saved to: {CHART_DIR}")
    print(f" - ear_{timestamp}.png")
    print(f" - perclos_{timestamp}.png")
    print(f" - fusion_{timestamp}.png")
    print(f" - combined_overview_{timestamp}.png")


if __name__ == "__main__":
    main()