"""
progress.py - Lightweight console progress for long local operations.
"""

from __future__ import annotations

import sys
import threading
import time
from types import TracebackType


def _format_elapsed(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


class ProgressTimer:
    """Print a same-line spinner with elapsed time while an operation runs."""

    _SPINNER_FRAMES = ("|", "/", "-", "\\")

    def __init__(
        self,
        message: str,
        *,
        done_message: str = "Done",
        interval_seconds: float = 2.0,
    ) -> None:
        self.message = message
        self.done_message = done_message
        self.interval_seconds = interval_seconds
        self._started_at = 0.0
        self._frame_index = 0
        self._last_line_length = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "ProgressTimer":
        self._started_at = time.monotonic()
        self._write_status()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        elapsed = _format_elapsed(time.monotonic() - self._started_at)
        if exc_type is None:
            self._write_final(f"  {self.done_message} in {elapsed}")
        else:
            self._write_final(f"  Failed after {elapsed}")

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self._frame_index += 1
            self._write_status()

    def _write_status(self) -> None:
        elapsed = _format_elapsed(time.monotonic() - self._started_at)
        frame = self._SPINNER_FRAMES[self._frame_index % len(self._SPINNER_FRAMES)]
        self._write_line(f"  {frame} {self.message} {elapsed} elapsed")

    def _write_final(self, line: str) -> None:
        with self._lock:
            padding = " " * max(0, self._last_line_length - len(line))
            sys.stdout.write(f"\r{line}{padding}\n")
            sys.stdout.flush()
            self._last_line_length = 0

    def _write_line(self, line: str) -> None:
        with self._lock:
            padding = " " * max(0, self._last_line_length - len(line))
            sys.stdout.write(f"\r{line}{padding}")
            sys.stdout.flush()
            self._last_line_length = len(line)
