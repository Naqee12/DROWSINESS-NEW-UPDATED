import smtplib
import ssl
import os
from email.message import EmailMessage


def send_drowsy_alert(config, event_count, session_duration_sec, snapshot_path=None):
    """Sends an email (and optionally SMS via carrier gateway) alerting an emergency contact."""
    notif_cfg = config.get("NOTIFICATIONS", {})

    if not notif_cfg.get("ENABLED", False):
        return False

    sender = notif_cfg.get("SENDER_EMAIL", "")
    password = notif_cfg.get("SENDER_APP_PASSWORD", "")
    recipient_email = notif_cfg.get("RECIPIENT_EMAIL", "")
    recipient_sms = notif_cfg.get("RECIPIENT_SMS_GATEWAY", "")

    if not sender or not password or not (recipient_email or recipient_sms):
        print("[notifier] Notification settings incomplete, skipping alert.")
        return False

    minutes = int(session_duration_sec // 60)
    seconds = int(session_duration_sec % 60)

    subject = "Driver Drowsiness Alert"
    body = (
        f"Drowsiness detected {event_count} times during a driving session.\n"
        f"Session duration so far: {minutes}m {seconds}s.\n"
        f"Please check on the driver if possible."
    )

    recipients = [r for r in [recipient_email, recipient_sms] if r]

    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = ", ".join(recipients)
        msg.set_content(body)

        if snapshot_path and os.path.exists(snapshot_path):
            with open(snapshot_path, "rb") as f:
                img_data = f.read()
            msg.add_attachment(img_data, maintype="image", subtype="jpeg", filename="drowsy_snapshot.jpg")

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender, password)
            server.send_message(msg)

        print(f"[notifier] Alert sent to: {', '.join(recipients)}")
        return True

    except Exception as e:
        print(f"[notifier] Failed to send alert: {e}")
        return False