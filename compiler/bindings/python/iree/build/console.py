# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import IO

import shutil
import textwrap
import threading
import traceback
from iree.build.executor import BuildDependency, ProgressReporter


class ConsoleProgressReporter(ProgressReporter):
    def __init__(
        self,
        out: IO,
        *,
        rich_console: bool = True,
        long_display_time_threshold: int = 5,
    ):
        self.out = out
        self.rich_console = rich_console
        self.long_display_time_threshold = long_display_time_threshold
        self.display_lines: list[str] = []
        self.inflight_deps: set[BuildDependency] = set()
        self.finished_deps: set[BuildDependency] = set()
        self.most_recent_dep: BuildDependency | None = None
        self.all_deps: set[BuildDependency] = set()
        self.poller_thread: threading.Thread | None = None
        self.lock = threading.RLock()
        self.exit_poller_event = threading.Event()

    @property
    def started_count(self) -> int:
        return len(self.finished_deps) + len(self.inflight_deps)

    def reset_display(self):
        # Clean all known displayed lines.
        if not self.rich_console:
            return
        for line in reversed(self.display_lines):
            print(f"\033[A{' ' * len(line)}", file=self.out, end="\r")

    def draw_display(self):
        for line in self.display_lines:
            print(line, file=self.out)

    def refresh(self):
        current_deps = list(self.inflight_deps)
        if not current_deps:
            return
        new_display_lines = []
        if not self.rich_console:
            if not self.most_recent_dep:
                return
            progress_prefix = f"[{self.started_count + 1}/{len(self.all_deps)}]"
            new_display_lines.append(f"{progress_prefix} {self.most_recent_dep}")
        else:
            current_deps.sort(key=lambda dep: dep.execution_time)
            active_deps = [d for d in current_deps if d.invoke_time is not None]
            if not active_deps:
                active_deps = current_deps
            focus_dep = active_deps[0]
            longest_time = active_deps[-1].execution_time

            progress_prefix = f"[{self.started_count + 1}/{len(self.all_deps)}]"
            if longest_time > self.long_display_time_threshold:
                # Do a long display.
                long_count = 15
                report_count = min(long_count, len(active_deps))
                report_deps = active_deps[-report_count:]
                new_display_lines.append(
                    f"{progress_prefix} Waiting for long running actions:"
                )
                for dep in report_deps:
                    new_display_lines.append(
                        f"    {dep} ({round(dep.execution_time)}s)"
                    )
                remaining_count = len(active_deps) - report_count
                if remaining_count > 0:
                    new_display_lines.append(f"    ... and {remaining_count} more")
            else:
                # Summary display
                new_display_lines.append(f"{progress_prefix} {focus_dep}")

        # Reduce flicker by only refreshing if changed.
        if new_display_lines != self.display_lines:
            self.reset_display()
            self.display_lines.clear()
            self.display_lines.extend(new_display_lines)
            self.draw_display()

    def start_graph(self, all_deps: set[BuildDependency]):
        with self.lock:
            self.all_deps.update(all_deps)
            self.inflight_deps.clear()
            self.finished_deps.clear()
            if self.rich_console:
                self.poller_thread = threading.Thread(
                    target=self._poll, name="ConsolePoller", daemon=True
                )
                self.poller_thread.start()

    def start_dep(self, dep: BuildDependency):
        with self.lock:
            self.inflight_deps.add(dep)
            self.most_recent_dep = dep
            self.refresh()

    def finish_dep(self, dep: BuildDependency):
        with self.lock:
            self.finished_deps.add(dep)
            if dep in self.inflight_deps:
                self.inflight_deps.remove(dep)
            self.refresh()

    def report_failure(self, dep: "BuildDependency"):
        if dep.is_dependence_failure:
            return
        with self.lock:
            self.reset_display()
            self.display_lines.clear()
            print(f"ERROR: Building '{dep}' failed:", file=self.out)
            if dep.failure:
                failure_formatted = "".join(traceback.format_exception(dep.failure))
                print(f"{textwrap.indent(failure_formatted, '    ')}\n", file=self.out)

    def end_graph(self):
        if self.rich_console:
            self.exit_poller_event.set()
            self.poller_thread.join()
        with self.lock:
            self.reset_display()
            self.display_lines.clear()

            success_count = 0
            failed_count = 0
            for dep in self.finished_deps:
                if dep.failure:
                    failed_count += 1
                else:
                    success_count += 1
            if failed_count == 0:
                print(f"Successfully built {success_count} actions", file=self.out)

    def _poll(self):
        while not self.exit_poller_event.wait(timeout=1):
            with self.lock:
                self.refresh()
