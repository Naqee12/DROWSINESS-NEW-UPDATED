import os
import glob
import csv
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

LOG_PATH = os.path.join("output", "session_log.csv")
CHART_DIR = os.path.join("output", "charts")
SNAPSHOT_DIR = os.path.join("output", "snapshots")
SUMMARY_PATH = os.path.join("output", "session_summary.txt")
REPORT_DIR = os.path.join("output", "reports")

os.makedirs(REPORT_DIR, exist_ok=True)


def read_summary():
    """Parses session_summary.txt into a dict, if it exists."""
    if not os.path.exists(SUMMARY_PATH):
        return {}
    data = {}
    with open(SUMMARY_PATH, "r") as f:
        for line in f:
            if ":" in line:
                key, val = line.split(":", 1)
                data[key.strip()] = val.strip()
    return data


def compute_stats_from_csv():
    """Fallback: compute basic stats directly from CSV if summary.txt is missing."""
    if not os.path.exists(LOG_PATH):
        return {}

    total = 0
    drowsy = 0
    max_perclos = 0
    blinks = 0

    with open(LOG_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if row["status"] == "DROWSY":
                drowsy += 1
            max_perclos = max(max_perclos, float(row["perclos"]))
            blinks = int(row["blink_count"])

    return {
        "Total Frames Analyzed": str(total),
        "Frames Drowsy": f"{drowsy} ({(drowsy/total*100 if total else 0):.1f}% of session)",
        "Max PERCLOS": f"{max_perclos:.1f}%",
        "Total Blinks": str(blinks)
    }


def latest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def generate_report():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(REPORT_DIR, f"drowsiness_report_{timestamp}.pdf")

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                             topMargin=15*mm, bottomMargin=15*mm,
                             leftMargin=18*mm, rightMargin=18*mm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], fontSize=18, spaceAfter=6)
    subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=10, textColor=colors.grey)

    elements = []

    elements.append(Paragraph("Driver Drowsiness Detection - Session Report", title_style))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style))
    elements.append(Spacer(1, 10*mm))

    # ---------- Summary table ----------
    summary = read_summary()
    if not summary:
        summary = compute_stats_from_csv()

    if summary:
        table_data = [["Metric", "Value"]] + [[k, v] for k, v in summary.items() if k != "====" and "SUMMARY" not in k and "=" not in k]
        table = Table(table_data, colWidths=[90*mm, 70*mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263238")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(table)
    else:
        elements.append(Paragraph("No session summary or log data found.", styles["Normal"]))

    elements.append(Spacer(1, 8*mm))

    # ---------- Combined chart ----------
    combined_chart = latest_file(os.path.join(CHART_DIR, "combined_overview_*.png"))
    if combined_chart:
        elements.append(Paragraph("Session Overview", styles["Heading2"]))
        elements.append(Image(combined_chart, width=170*mm, height=153*mm))
        elements.append(Spacer(1, 6*mm))
    else:
        elements.append(Paragraph("No chart found. Run generate_charts.py first for a visual overview.", styles["Normal"]))

    # ---------- Snapshot thumbnails ----------
    snapshots = sorted(glob.glob(os.path.join(SNAPSHOT_DIR, "*.jpg")), key=os.path.getmtime, reverse=True)[:4]
    if snapshots:
        elements.append(Paragraph("Sample Drowsy Event Snapshots", styles["Heading2"]))
        thumb_row = []
        for snap in snapshots:
            thumb_row.append(Image(snap, width=38*mm, height=28*mm))
        thumb_table = Table([thumb_row])
        thumb_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        elements.append(thumb_table)

    doc.build(elements)
    print(f"Report saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_report()