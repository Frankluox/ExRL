"""Cooperative checkpoint requests and explicit Asia/Shanghai GPU hours."""

from __future__ import annotations

from datetime import datetime, timedelta
import os
import signal
import time
from zoneinfo import ZoneInfo


TIMEZONE = ZoneInfo("Asia/Shanghai")


def night_window(now=None):
    """Return (may_run, next_09_00_timestamp); daytime is [09:00, 21:00)."""
    now = datetime.now(TIMEZONE) if now is None else now.astimezone(TIMEZONE)
    allowed = now.hour >= 21 or now.hour < 9
    morning = now.replace(hour=9, minute=0, second=0, microsecond=0)
    if now.hour >= 9:
        morning += timedelta(days=1)
    return allowed, morning.timestamp()


class CheckpointPause(Exception):
    """Raised only at collector/update boundaries, never by a signal handler."""


class CheckpointControl:
    def __init__(self, stop_at_unix=None, enabled=False):
        self.stop_at_unix = stop_at_unix
        self.enabled = bool(enabled or stop_at_unix is not None)
        self.reason = None
        self.owner = os.getpid()
        self.previous = {}

    def install(self):
        if self.enabled:
            for signum in (signal.SIGUSR1, signal.SIGTERM):
                self.previous[signum] = signal.getsignal(signum)
                signal.signal(signum, self.request)

    def request(self, signum, _frame=None):
        if os.getpid() == self.owner:
            self.reason = "signal_%d" % signum

    def check(self):
        if self.stop_at_unix is not None and time.time() >= self.stop_at_unix:
            self.reason = "daily_09_00_deadline"
        if self.reason is not None:
            raise CheckpointPause(self.reason)

    def close(self):
        for signum, previous in self.previous.items():
            signal.signal(signum, previous)
