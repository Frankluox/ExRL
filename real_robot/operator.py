"""Timestamp operator stage labels independently of policy inference."""

from __future__ import annotations

import threading
import time


class Operator:
    def __init__(self, stages):
        self.stages = int(stages)
        self.events = []
        self.failure = None
        self.cancelled = False
        self.active = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        while True:
            try:
                command = input().strip().lower()
            except EOFError:
                self.cancelled = True
                return
            with self.lock:
                if command == "q":
                    self.cancelled = True
                    return
                if not self.active:
                    if command == "start":
                        self.active = True
                    continue
                if command == "s" and len(self.events) < self.stages:
                    self.events.append(time.monotonic())
                elif command == "f":
                    self.failure = time.monotonic()

    def start_episode(self):
        with self.lock:
            self.events, self.failure, self.active = [], None, False
        print(
            "Reset the task, then enter start. During rollout: s = stage/success, f = failure, q = quit.",
            flush=True,
        )
        while not self.active:
            if self.cancelled:
                raise KeyboardInterrupt
            time.sleep(0.02)

    def snapshot(self):
        with self.lock:
            return list(self.events), self.failure, self.cancelled

    def end_episode(self):
        with self.lock:
            self.active = False
