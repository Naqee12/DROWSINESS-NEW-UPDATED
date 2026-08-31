import csv
import os
import matplotlib.pyplot as plt
from datetime import datetime

HISTORY_PATH = os.path.join("output", "session_history.csv")
TREND_DIR = os.path.join("output", "trends")
os.makedirs(TREND_DIR, exist_ok=True)


def load_history():
    sessions = []
    if not os.path.exists(HISTORY_PATH):
        return sessions

    with open(HISTORY_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sessions.append(row)
    return sessions


def plot_trends(sessions):
    if len(sessions) < 2:
        print("Need at least 2 sessions to show a trend. Run more sessions first.")
        return

    labels = [f"S{i+1}\n{s['date'][5:16]}" for i, s in enumerate(sessions)]
    avg_perclos = [float(s["avg_perclos"]) for s in sessions]
    max_perclos = [float(s["max_perclos"]) for s in sessions]
    blink_rate = [float(s["blink_rate_per_min"]) for s in sessions]
    drowsy_events = [int(s["drowsy_events"]) for s in sessions]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(labels, avg_perclos, marker="o", color="#e64a19", label="Avg PERCLOS")
    axes[0].plot(labels, max_perclos, marker="o", color="#ff8a65", linestyle="--", label="Max PERCLOS")
    axes[0].set_ylabel("PERCLOS (%)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_title("Fatigue Trend Across Sessions")

    axes[1].plot(labels, blink_rate, marker="o", color="#1976d2")
    axes[1].set_ylabel("Blinks / min")
    axes[1].grid(alpha=0.3)

    axes[2].bar(labels, drowsy_events, color="#7b1fa2")
    axes[2].set_ylabel("Drowsy Events")
    axes[2].set_xlabel("Session")
    axes[2].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out_path = os.path.join(TREND_DIR, f"trend_overview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Trend chart saved to: {out_path}")

    # Simple drift check
    if len(avg_perclos) >= 3:
        recent_avg = sum(avg_perclos[-3:]) / 3
        earlier_avg = sum(avg_perclos[:3]) / 3
        if recent_avg > earlier_avg * 1.2:
            print(f"⚠ PERCLOS trending upward: early sessions avg {earlier_avg:.1f}% -> recent sessions avg {recent_avg:.1f}%")
        else:
            print(f"PERCLOS stable/improving: early sessions avg {earlier_avg:.1f}% -> recent sessions avg {recent_avg:.1f}%")


def main():
    sessions = load_history()
    print(f"Loaded {len(sessions)} session(s) from history.")
    plot_trends(sessions)


if __name__ == "__main__":
    main()