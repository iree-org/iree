# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GitHub CI data fetcher for corpus building.

Fetches workflow runs and job logs from GitHub Actions using the GitHub CLI.
Follows the patterns established in iree-ci-triage for consistency.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from common import console

from ci.core.corpus import Corpus
from ci.core.github_client import GitHubClient, GitHubClientError
from ci.core.utils import is_meta_job


@dataclass
class FetchResult:
    """Results of a fetch operation."""

    runs_fetched: int
    runs_new: int
    runs_duplicate: int
    logs_fetched: int
    logs_failed: int
    errors: list[str]


class GitHubFetcher:
    """Fetches CI data from GitHub and stores in corpus."""

    def __init__(
        self, client: GitHubClient, corpus: Corpus, args: Any | None = None
    ) -> None:
        """Initialize fetcher.

        Args:
            client: GitHub client instance
            corpus: Corpus instance for storage
            args: Optional args namespace for progress reporting (uses quiet flag)
        """
        self.client = client
        self.corpus = corpus
        self.args = args

    def _chunk_time_range(
        self, since: datetime, until: datetime, chunk_days: int = 30
    ) -> list[tuple[datetime, datetime]]:
        """Split a time range into chunks for incremental fetching.

        Args:
            since: Start of time range (inclusive)
            until: End of time range (exclusive)
            chunk_days: Days per chunk (default: 30 for ~monthly chunks)

        Returns:
            List of (start, end) datetime tuples representing chunks
        """
        chunks = []
        current = since

        while current < until:
            chunk_end = min(current + timedelta(days=chunk_days), until)
            chunks.append((current, chunk_end))
            current = chunk_end

        return chunks

    def fetch_since(
        self,
        since: datetime,
        limit: int = 100,
        branch: str | None = None,
        status: str = "failure",
    ) -> FetchResult:
        """Fetch all failure runs since a timestamp using time-based chunking.

        Args:
            since: Fetch runs created after this datetime
            limit: Maximum number of runs to fetch per chunk
            branch: Filter by branch (None = all branches)
            status: Run status filter (default: 'failure')

        Returns:
            FetchResult with statistics
        """
        result = FetchResult(
            runs_fetched=0,
            runs_new=0,
            runs_duplicate=0,
            logs_fetched=0,
            logs_failed=0,
            errors=[],
        )

        # Split time range into monthly chunks.
        until = datetime.now(timezone.utc)
        chunks = self._chunk_time_range(since, until, chunk_days=30)

        if branch:
            console.note(
                f"  Fetching in {len(chunks)} time chunks from branch '{branch}'...",
                args=self.args,
            )
        else:
            console.note(
                f"  Fetching in {len(chunks)} time chunks from ALL branches...",
                args=self.args,
            )

        # Process each time chunk.
        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks, 1):
            chunk_start_str = chunk_start.strftime("%Y-%m-%d")
            chunk_end_str = chunk_end.strftime("%Y-%m-%d")

            console.note(
                f"  Chunk {chunk_idx}/{len(chunks)}: {chunk_start_str} to {chunk_end_str}...",
                args=self.args,
            )

            # Fetch runs for this chunk.
            try:
                if branch:
                    # Fetch from specific branch.
                    chunk_runs_raw = self.client.get_runs_for_branch(
                        branch,
                        status=status,
                        limit=limit,
                        created_since=chunk_start_str,
                    )
                else:
                    # Fetch from ALL branches (no branch filter).
                    chunk_runs_raw = self.client.get_all_runs(
                        status=status,
                        limit=limit,
                        created_since=chunk_start_str,
                    )

                # Filter by time range (client-side, since gh --created only supports date-level >=).
                chunk_runs = []
                for run in chunk_runs_raw:
                    created = datetime.fromisoformat(
                        run.created_at.replace("Z", "+00:00")
                    )
                    # Check both bounds: chunk_start <= created < chunk_end.
                    if chunk_start <= created < chunk_end:
                        chunk_runs.append(run)

            except GitHubClientError as e:
                result.errors.append(
                    f"Chunk {chunk_idx}/{len(chunks)} ({chunk_start_str} to {chunk_end_str}): "
                    f"Failed to fetch: {e}"
                )
                continue

            # Process runs from this chunk.
            for run in chunk_runs:
                result.runs_fetched += 1

                # Try to add to corpus.
                if self._fetch_and_store_run(run.run_id, result):
                    result.runs_new += 1
                else:
                    result.runs_duplicate += 1

            # Show chunk completion summary.
            console.note(
                f"    Chunk {chunk_idx}/{len(chunks)} complete: "
                f"+{len(chunk_runs)} runs processed "
                f"(total: {result.runs_new} new, {result.logs_fetched} logs)",
                args=self.args,
            )

        return result

    def fetch_run(self, run_id: str) -> FetchResult:
        """Fetch a specific run and its logs.

        Args:
            run_id: Workflow run ID to fetch

        Returns:
            FetchResult with statistics
        """
        result = FetchResult(
            runs_fetched=0,
            runs_new=0,
            runs_duplicate=0,
            logs_fetched=0,
            logs_failed=0,
            errors=[],
        )

        if self._fetch_and_store_run(run_id, result):
            result.runs_new += 1
        else:
            result.runs_duplicate += 1
        return result

    def fetch_pr(self, pr_number: int, status: str = "failure") -> FetchResult:
        """Fetch failures from a PR.

        Args:
            pr_number: Pull request number
            status: Run status filter (default: 'failure')

        Returns:
            FetchResult with statistics
        """
        result = FetchResult(
            runs_fetched=0,
            runs_new=0,
            runs_duplicate=0,
            logs_fetched=0,
            logs_failed=0,
            errors=[],
        )

        try:
            # Get runs for this PR.
            runs = self.client.get_runs_for_pr(pr_number, status=status, limit=10)

            for run in runs:
                result.runs_fetched += 1
                if self._fetch_and_store_run(run.run_id, result):
                    result.runs_new += 1
                else:
                    result.runs_duplicate += 1

        except GitHubClientError as e:
            result.errors.append(f"Failed to fetch PR #{pr_number}: {e}")

        return result

    def _check_annotation_infrastructure_flake(
        self, annotations: list[Any]
    ) -> tuple[bool, str]:
        """Check if annotations indicate infrastructure flake.

        Args:
            annotations: List of annotation dictionaries from GitHub API

        Returns:
            Tuple of (is_flake, error_code) where is_flake is True if annotations
            contain infrastructure failure patterns
        """
        # Infrastructure flake patterns in annotations.
        patterns = [
            ("No space left on device", "RUNNER_DISK_FULL"),
            ("System.IO.IOException", "RUNNER_IO_ERROR"),
            ("GitHub.Runner.Worker", "RUNNER_CRASH"),
            ("Unhandled exception", "RUNNER_UNHANDLED_EXCEPTION"),
            ("The self-hosted runner", "RUNNER_LOST"),
            ("connection to the server was lost", "RUNNER_CONNECTION_LOST"),
        ]

        for annotation in annotations:
            message = annotation.get("message", "")
            for pattern, error_code in patterns:
                if pattern in message:
                    return (True, error_code)

        return (False, "")

    def _fetch_and_store_run(self, run_id: str, result: FetchResult) -> bool:
        """Fetch and store a single run.

        Args:
            run_id: Run ID to fetch
            result: FetchResult to update with stats

        Returns:
            True if new run added, False if duplicate
        """
        try:
            # Fetch run metadata.
            is_existing_run = self.corpus.has_run(run_id)
            if is_existing_run:
                console.note(
                    f"  Checking run {run_id} for missing data...", args=self.args
                )
            else:
                console.note(f"  Fetching run {run_id}...", args=self.args)

            run_data = self.client.get_run(run_id)
            if not run_data:
                result.errors.append(f"Run {run_id} not found")
                return False

            # Get failed jobs.
            all_failed_jobs = self.client.get_failed_jobs(run_id)

            # Filter out meta/summary jobs (they just aggregate other failures).
            failed_jobs = [job for job in all_failed_jobs if not is_meta_job(job.name)]

            skipped_meta = len(all_failed_jobs) - len(failed_jobs)
            if skipped_meta > 0:
                console.note(
                    f"    Skipped {skipped_meta} meta/summary jobs",
                    args=self.args,
                )

            console.note(
                f"    Found {len(failed_jobs)} failed jobs in run {run_id}",
                args=self.args,
            )

            # Build run metadata for corpus.
            run_meta = {
                # Core identifiers.
                "run_id": run_id,
                "workflow": run_data.get("workflowName", ""),
                "workflow_id": run_data.get("workflowDatabaseId", ""),
                "name": run_data.get("name", ""),
                # Git/commit info.
                "head_sha": run_data.get("headSha", ""),
                "branch": run_data.get("headBranch", ""),
                # Trigger info.
                "event": run_data.get("event", ""),
                "pr_number": run_data.get("number"),  # None if not PR event.
                # Status and timing.
                "status": run_data.get("status", ""),
                "conclusion": run_data.get("conclusion", ""),
                "attempt": run_data.get("attempt", 1),
                "created_at": run_data.get("createdAt", ""),
                "started_at": run_data.get("startedAt", ""),
                "updated_at": run_data.get("updatedAt", ""),
                # Display and links.
                "display_title": run_data.get("displayTitle", ""),
                "url": run_data.get("url", ""),
                # Job stats.
                "jobs": len(run_data.get("jobs", [])),
                "failed_jobs": len(failed_jobs),
                # Corpus metadata.
                "fetched_at": datetime.now().isoformat(),
                "classified": False,
            }

            # Add to corpus (only if new run).
            if not is_existing_run and not self.corpus.add_run(run_meta):
                return False

            # Fetch logs and annotations for failed jobs.
            any_data_fetched = False
            for i, job in enumerate(failed_jobs, 1):
                try:
                    # Check if log already exists and is non-empty.
                    log_path = self.corpus.get_log_path(run_id, job.job_id)
                    log_exists = (
                        log_path and log_path.exists() and log_path.stat().st_size > 0
                    )

                    # Fetch annotations first (fast, <1s).
                    annotations = self.client.get_job_annotations(job.job_id)

                    # Always save annotations if they exist.
                    if annotations:
                        self.corpus.save_annotations(run_id, job.job_id, annotations)
                        any_data_fetched = True

                    # Check if annotations indicate infrastructure flake.
                    is_flake, error_code = self._check_annotation_infrastructure_flake(
                        annotations
                    )

                    if is_flake:
                        # Skip log download for infrastructure flakes.
                        console.note(
                            f"    Skipping log {i}/{len(failed_jobs)}: {job.name} "
                            f"(infrastructure flake: {error_code})",
                            args=self.args,
                        )
                        continue

                    # Skip log download if already exists and non-empty.
                    if log_exists:
                        console.note(
                            f"    Skipping log {i}/{len(failed_jobs)}: {job.name} "
                            f"(already exists, {log_path.stat().st_size} bytes)",
                            args=self.args,
                        )
                        continue

                    # Not an infrastructure flake and doesn't exist - download log.
                    console.note(
                        f"    Downloading log {i}/{len(failed_jobs)}: {job.name}",
                        args=self.args,
                    )
                    log_content = self.client.get_job_log(run_id, job.job_id)
                    if log_content:
                        self.corpus.save_log(run_id, job.job_id, log_content)
                        result.logs_fetched += 1
                        any_data_fetched = True
                    else:
                        result.logs_failed += 1
                        result.errors.append(f"No log available for job {job.job_id}")

                except GitHubClientError as e:
                    result.logs_failed += 1
                    result.errors.append(str(e))

            # For existing runs, only return True if we fetched new data.
            if is_existing_run:
                if not any_data_fetched:
                    console.note(
                        f"  Run {run_id} already complete (no missing data)",
                        args=self.args,
                    )
                return any_data_fetched

            # New run was added.
            return True

        except GitHubClientError as e:
            result.errors.append(f"Failed to fetch run {run_id}: {e}")
            return False
