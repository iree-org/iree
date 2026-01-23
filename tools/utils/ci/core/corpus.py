# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Corpus management for CI failures.

The corpus stores CI failure logs and metadata in a structured directory:
- corpus.jsonl: Main index (streaming JSONL format)
- daily/: Daily fetch manifests
- runs/: Per-run metadata files
- logs/: Raw log files organized by run
- classification/: Classification results and cache
- unrecognized/: Failures needing attention

This design allows incremental updates and reclassification without re-fetching.
"""

import json
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Corpus:
    """Manages the CI failure corpus with JSONL index."""

    def __init__(self, corpus_dir: Path) -> None:
        """Initialize corpus at given directory.

        Args:
            corpus_dir: Root directory for corpus storage
        """
        self.corpus_dir = Path(corpus_dir)
        self.index_path = self.corpus_dir / "corpus.jsonl"
        self.daily_dir = self.corpus_dir / "daily"
        self.runs_dir = self.corpus_dir / "runs"
        self.logs_dir = self.corpus_dir / "logs"
        self.classification_dir = self.corpus_dir / "classification"
        self.unrecognized_dir = self.corpus_dir / "unrecognized"
        self.config_path = self.corpus_dir / "config.json"

        # Ensure directories exist.
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure corpus directory structure exists."""
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.daily_dir.mkdir(exist_ok=True)
        self.runs_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.classification_dir.mkdir(exist_ok=True)
        (self.classification_dir / "cache").mkdir(exist_ok=True)
        (self.classification_dir / "history").mkdir(exist_ok=True)
        self.unrecognized_dir.mkdir(exist_ok=True)
        (self.unrecognized_dir / "samples").mkdir(exist_ok=True)

        # Create default config if doesn't exist.
        if not self.config_path.exists():
            default_config = {
                "corpus_version": "2.0",
                "created_at": datetime.now().isoformat(),
                "github_repo": "iree-org/iree",
                "fetch_settings": {
                    "default_limit": 100,
                    "include_branches": ["*"],  # All branches by default.
                },
                "classification_settings": {
                    "iree_ci_triage_args": ["--json", "--no-context"],
                    "cache_results": True,
                    "cache_ttl_days": 30,
                },
            }
            self.config_path.write_text(json.dumps(default_config, indent=2))

    def add_run(self, run_data: dict[str, Any]) -> bool:
        """Add a run to the corpus (deduped).

        Args:
            run_data: Run metadata dict with at minimum:
                - run_id: str
                - created_at: ISO 8601 timestamp
                - branch: str
                - workflow: str
                - conclusion: str

        Returns:
            True if new run added, False if duplicate.
        """
        run_id = str(run_data["run_id"])

        # Check for duplicate.
        if self.has_run(run_id):
            return False

        # Append to corpus.jsonl.
        with open(self.index_path, "a", encoding="utf-8") as f:
            json.dump(run_data, f)
            f.write("\n")

        # Store full run metadata in runs/ directory.
        run_path = self.runs_dir / f"{run_id}.json"
        run_path.write_text(json.dumps(run_data, indent=2))

        # Update daily manifest.
        date = datetime.fromisoformat(
            run_data["created_at"].replace("Z", "+00:00")
        ).date()
        daily_path = self.daily_dir / f"{date}.json"

        daily_data = {"date": str(date), "runs": []}
        if daily_path.exists():
            daily_data = json.loads(daily_path.read_text())

        if run_id not in daily_data["runs"]:
            daily_data["runs"].append(run_id)

        daily_path.write_text(json.dumps(daily_data, indent=2))

        return True

    def has_run(self, run_id: str) -> bool:
        """Check if run already in corpus.

        Args:
            run_id: Run ID to check

        Returns:
            True if run exists, False otherwise
        """
        run_id = str(run_id)

        # Quick check: does the run metadata file exist?
        if (self.runs_dir / f"{run_id}.json").exists():
            return True

        # Fallback: scan corpus.jsonl (slower).
        if not self.index_path.exists():
            return False

        with open(self.index_path) as f:
            for line in f:
                data = json.loads(line)
                if str(data.get("run_id")) == run_id:
                    return True
        return False

    def get_runs(
        self,
        since: datetime | None = None,
        branch: str | None = None,
        status: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Query runs from corpus with filters.

        Args:
            since: Only return runs created after this datetime
            branch: Filter by branch name
            status: Filter by conclusion (e.g., 'failure', 'success')

        Yields:
            Run metadata dicts
        """
        if not self.index_path.exists():
            return

        with open(self.index_path) as f:
            for line in f:
                data = json.loads(line)

                # Apply filters.
                if since:
                    created = datetime.fromisoformat(
                        data["created_at"].replace("Z", "+00:00")
                    )
                    if created < since:
                        continue

                if branch and data.get("branch") != branch:
                    continue

                if status and data.get("conclusion") != status:
                    continue

                yield data

    def get_unclassified(self) -> Iterator[dict[str, Any]]:
        """Get runs that haven't been classified yet.

        Yields:
            Run metadata dicts for unclassified runs
        """
        for run in self.get_runs():
            if not run.get("classified", False):
                yield run

    def get_last_fetched_at(self) -> datetime | None:
        """Get the most recent fetch attempt timestamp.

        Returns:
            Last fetch attempt as datetime (timezone-aware), or None if never fetched
        """
        # First check config for last_fetch_at (updated after each fetch attempt).
        if self.config_path.exists():
            config = json.loads(self.config_path.read_text())
            last_fetch_str = config.get("last_fetch_at")
            if last_fetch_str:
                last_fetch = datetime.fromisoformat(last_fetch_str)
                # Ensure timezone-aware.
                if last_fetch.tzinfo is None:
                    last_fetch = last_fetch.replace(tzinfo=timezone.utc)
                return last_fetch

        # Fallback: find most recent fetched_at from runs (for backwards compat).
        if not self.index_path.exists():
            return None

        last_fetched = None
        for run in self.get_runs():
            fetched_at_str = run.get("fetched_at")
            if fetched_at_str:
                fetched_at = datetime.fromisoformat(fetched_at_str)
                # Ensure timezone-aware.
                if fetched_at.tzinfo is None:
                    fetched_at = fetched_at.replace(tzinfo=timezone.utc)
                if last_fetched is None or fetched_at > last_fetched:
                    last_fetched = fetched_at

        return last_fetched

    def save_log(self, run_id: str, job_id: str, content: str) -> Path:
        """Save raw log to disk.

        Args:
            run_id: Run ID
            job_id: Job ID
            content: Log content

        Returns:
            Path to saved log file
        """
        run_id = str(run_id)
        job_id = str(job_id)

        run_dir = self.logs_dir / run_id
        run_dir.mkdir(exist_ok=True)

        log_path = run_dir / f"{job_id}.log"
        log_path.write_text(content)

        return log_path

    def get_log_path(self, run_id: str, job_id: str) -> Path | None:
        """Get path to a log file if it exists.

        Args:
            run_id: Run ID
            job_id: Job ID

        Returns:
            Path to log file, or None if doesn't exist
        """
        run_id = str(run_id)
        job_id = str(job_id)

        log_path = self.logs_dir / run_id / f"{job_id}.log"
        return log_path if log_path.exists() else None

    def save_annotations(
        self, run_id: str, job_id: str, annotations: list[dict[str, Any]]
    ) -> Path:
        """Save job annotations to disk.

        Args:
            run_id: Run ID
            job_id: Job ID
            annotations: List of annotation dictionaries from GitHub API

        Returns:
            Path to saved annotations file
        """
        run_id = str(run_id)
        job_id = str(job_id)

        run_dir = self.logs_dir / run_id
        run_dir.mkdir(exist_ok=True)

        annotations_path = run_dir / f"{job_id}.annotations.json"
        annotations_path.write_text(json.dumps(annotations, indent=2))

        return annotations_path

    def get_annotations(self, run_id: str, job_id: str) -> list[dict[str, Any]] | None:
        """Get annotations for a job if they exist.

        Args:
            run_id: Run ID
            job_id: Job ID

        Returns:
            List of annotation dictionaries, or None if file doesn't exist
        """
        run_id = str(run_id)
        job_id = str(job_id)

        annotations_path = self.logs_dir / run_id / f"{job_id}.annotations.json"
        if not annotations_path.exists():
            return None

        return json.loads(annotations_path.read_text())

    def get_run_metadata(self, run_id: str) -> dict[str, Any] | None:
        """Get full metadata for a run.

        Args:
            run_id: Run ID

        Returns:
            Run metadata dict, or None if not found
        """
        run_id = str(run_id)
        run_path = self.runs_dir / f"{run_id}.json"

        if not run_path.exists():
            return None

        return json.loads(run_path.read_text())

    def save_run_metadata(self, run_id: str, metadata: dict[str, Any]) -> None:
        """Save or update run metadata.

        Args:
            run_id: Run ID
            metadata: Run metadata dict
        """
        run_id = str(run_id)
        run_path = self.runs_dir / f"{run_id}.json"
        run_path.write_text(json.dumps(metadata, indent=2))

    def get_stats(self) -> dict[str, Any]:
        """Get corpus statistics.

        Returns:
            Dict with corpus stats:
            - total_runs: int
            - total_logs: int
            - size_bytes: int
            - time_span: dict with start/end dates
            - by_branch: dict of run counts per branch
        """
        stats = {
            "total_runs": 0,
            "total_logs": 0,
            "size_bytes": 0,
            "time_span": {"start": None, "end": None},
            "by_branch": {},
        }

        if not self.index_path.exists():
            return stats

        # Count runs and analyze.
        with open(self.index_path) as f:
            for line in f:
                run = json.loads(line)
                stats["total_runs"] += 1

                # Track time span.
                created = run.get("created_at", "")
                if (
                    not stats["time_span"]["start"]
                    or created < stats["time_span"]["start"]
                ):
                    stats["time_span"]["start"] = created
                if not stats["time_span"]["end"] or created > stats["time_span"]["end"]:
                    stats["time_span"]["end"] = created

                # Count by branch.
                branch = run.get("branch", "unknown")
                stats["by_branch"][branch] = stats["by_branch"].get(branch, 0) + 1

        # Count logs and total size.
        for run_dir in self.logs_dir.iterdir():
            if run_dir.is_dir():
                for log_file in run_dir.glob("*.log"):
                    stats["total_logs"] += 1
                    stats["size_bytes"] += log_file.stat().st_size

        return stats
