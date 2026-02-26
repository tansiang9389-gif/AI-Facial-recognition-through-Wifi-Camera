"""
Maturity Tracker
- Accumulates per-camera maturity counts over time
- Generates daily maturity distribution reports
- Sends harvest-readiness alerts via Telegram
- Provides historical trend data for the web dashboard

Classes: Premature, Mature, Potential
"""

import os
import json
import time
import threading
from datetime import datetime, timedelta
from collections import defaultdict

from config import (
    CLASS_NAMES, CLASS_EMOJI,
    REPORTS_DIR, HARVEST_ALERT_THRESHOLD, HARVEST_ALERT_COOLDOWN_MINUTES,
    DAILY_REPORT_ENABLED, DAILY_REPORT_HOUR, TELEGRAM_ENABLED,
)
from coconut_detector import send_alert_async


class MaturityTracker:
    """Tracks coconut maturity detections and generates reports/alerts."""

    def __init__(self):
        # Current session counts per camera: {cam_label: {class_name: count}}
        self._session_counts = defaultdict(lambda: defaultdict(int))
        # Daily accumulator: {date_str: {cam_label: {class_name: count}}}
        self._daily_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        # Harvest alert cooldown per camera
        self._last_harvest_alert = {}
        # Lock for thread-safe access
        self._lock = threading.Lock()
        # Historical data file
        self._history_file = os.path.join(REPORTS_DIR, "maturity_history.json")
        # Load existing history
        self._history = self._load_history()
        # Start daily report scheduler
        if DAILY_REPORT_ENABLED:
            self._start_daily_scheduler()

    def _load_history(self):
        """Load historical trend data from disk."""
        if os.path.exists(self._history_file):
            try:
                with open(self._history_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_history(self):
        """Persist historical data to disk."""
        try:
            with open(self._history_file, "w") as f:
                json.dump(self._history, f, indent=2)
        except Exception as e:
            print(f"[Tracker] Save error: {e}")

    def record_detection(self, camera_label, detections):
        """Record a batch of detections from a single frame."""
        counts = {cls: 0 for cls in CLASS_NAMES}
        for det in detections:
            cls = det["class_name"]
            if cls in counts:
                counts[cls] += 1

        today = datetime.now().strftime("%Y-%m-%d")
        with self._lock:
            for cls, cnt in counts.items():
                self._session_counts[camera_label][cls] += cnt
                self._daily_counts[today][camera_label][cls] += cnt

        # Check harvest-readiness alert (Mature coconuts)
        mature_count = counts.get("Mature", 0)
        if mature_count >= HARVEST_ALERT_THRESHOLD:
            self._check_harvest_alert(camera_label, mature_count, counts)

        return counts

    def _check_harvest_alert(self, camera_label, mature_count, counts):
        """Send harvest-readiness alert if cooldown has elapsed."""
        now = time.time()
        last = self._last_harvest_alert.get(camera_label, 0)
        cooldown_sec = HARVEST_ALERT_COOLDOWN_MINUTES * 60

        if now - last >= cooldown_sec:
            self._last_harvest_alert[camera_label] = now
            emoji = CLASS_EMOJI
            msg = (
                f"\U0001F965 HARVEST ALERT - {camera_label}\n"
                f"Mature coconuts detected: {mature_count}\n"
                f"(threshold: {HARVEST_ALERT_THRESHOLD})\n\n"
                f"Distribution:\n"
                f"  {emoji.get('Premature', '')} Premature: {counts.get('Premature', 0)}\n"
                f"  {emoji.get('Mature', '')} Mature: {counts.get('Mature', 0)}\n"
                f"  {emoji.get('Potential', '')} Potential: {counts.get('Potential', 0)}\n\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            send_alert_async(msg)

    def get_camera_distribution(self, camera_label):
        """Get current session maturity distribution for a camera."""
        with self._lock:
            data = dict(self._session_counts.get(camera_label, {}))
        return {cls: data.get(cls, 0) for cls in CLASS_NAMES}

    def get_all_distributions(self):
        """Get maturity distributions for all cameras."""
        with self._lock:
            result = {}
            for cam, counts in self._session_counts.items():
                result[cam] = {cls: counts.get(cls, 0) for cls in CLASS_NAMES}
            return result

    def get_today_distributions(self):
        """Get today's maturity distributions for all cameras."""
        today = datetime.now().strftime("%Y-%m-%d")
        with self._lock:
            if today not in self._daily_counts:
                return {}
            result = {}
            for cam, counts in self._daily_counts[today].items():
                result[cam] = {cls: counts.get(cls, 0) for cls in CLASS_NAMES}
            return result

    def get_trend_data(self, days=14):
        """Get historical trend data for the dashboard charts."""
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        result = {}
        # Include saved history
        for date_str, cams in self._history.items():
            if date_str >= cutoff:
                result[date_str] = cams
        # Include today's live data
        today = datetime.now().strftime("%Y-%m-%d")
        with self._lock:
            if today in self._daily_counts:
                today_data = {}
                for cam, counts in self._daily_counts[today].items():
                    today_data[cam] = dict(counts)
                result[today] = today_data
        return dict(sorted(result.items()))

    def finalize_day(self, date_str=None):
        """Finalize a day's data into history and generate a report."""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        with self._lock:
            if date_str in self._daily_counts:
                day_data = {}
                for cam, counts in self._daily_counts[date_str].items():
                    day_data[cam] = dict(counts)
                self._history[date_str] = day_data
                self._save_history()
                return day_data
        return {}

    def generate_daily_report(self):
        """Generate and send a daily maturity distribution report."""
        today = datetime.now().strftime("%Y-%m-%d")
        day_data = self.finalize_day(today)

        if not day_data:
            return None

        emoji = CLASS_EMOJI
        lines = [
            f"\U0001F4CA Daily Coconut Maturity Report - {today}",
            "",
        ]
        for cam, counts in sorted(day_data.items()):
            premature = counts.get("Premature", 0)
            mature = counts.get("Mature", 0)
            potential = counts.get("Potential", 0)
            total = premature + mature + potential
            lines.append(
                f"\U0001F4F7 {cam}:\n"
                f"   {emoji.get('Premature', '')} Premature: {premature}\n"
                f"   {emoji.get('Mature', '')} Mature: {mature}\n"
                f"   {emoji.get('Potential', '')} Potential: {potential}\n"
                f"   Total: {total}"
            )
        lines.append("")
        lines.append("--")

        report_text = "\n".join(lines)

        # Save report to file
        report_path = os.path.join(REPORTS_DIR, f"report_{today}.txt")
        with open(report_path, "w") as f:
            f.write(report_text)

        # Send via Telegram
        if TELEGRAM_ENABLED:
            send_alert_async(report_text)

        print(f"[Tracker] Daily report generated: {report_path}")
        return report_text

    def _start_daily_scheduler(self):
        """Start a background thread that triggers daily reports."""
        def scheduler_loop():
            last_report_date = None
            while True:
                now = datetime.now()
                if now.hour >= DAILY_REPORT_HOUR and now.strftime("%Y-%m-%d") != last_report_date:
                    last_report_date = now.strftime("%Y-%m-%d")
                    self.generate_daily_report()
                time.sleep(60)

        t = threading.Thread(target=scheduler_loop, daemon=True)
        t.start()
