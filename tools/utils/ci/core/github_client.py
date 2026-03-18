# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GitHub CLI wrapper for fetching CI workflow data.

This module provides a Python interface to the GitHub CLI (gh) for:
- Checking authentication status
- Fetching workflow runs
- Fetching job details and logs
- Handling errors and rate limiting
"""

import json
import subprocess
import time
from dataclasses import dataclass
from typing import Any

from common import console


@dataclass
class WorkflowRun:
    """GitHub Actions workflow run."""

    run_id: str
    workflow_name: str
    conclusion: str
    created_at: str
    head_branch: str
    display_title: str


@dataclass
class Job:
    """GitHub Actions job."""

    job_id: str
    name: str
    conclusion: str
    runner_name: str | None
    started_at: str
    completed_at: str


class GitHubClientError(Exception):
    """Exception raised for GitHub CLI errors."""

    pass


class GitHubClient:
    """Client for interacting with GitHub via the gh CLI."""

    def __init__(self, repo: str = "iree-org/iree") -> None:
        """Initialize the GitHub client.

        Args:
            repo: GitHub repository in owner/repo format
        """
        self.repo = repo

    def _run_gh_command_with_retry(
        self,
        cmd: list[str],
        timeout: int = 120,
        max_retries: int = 3,
        base_delay: float = 5.0,
    ) -> subprocess.CompletedProcess:
        """Run a gh CLI command with retry logic on timeout.

        Args:
            cmd: Command to run (list of strings)
            timeout: Timeout in seconds for each attempt
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff

        Returns:
            CompletedProcess result from successful execution

        Raises:
            GitHubClientError: If command fails after all retries
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                return subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=timeout,
                )

            except subprocess.TimeoutExpired as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Exponential backoff: 5s, 10s, 20s.
                    delay = base_delay * (2**attempt)
                    console.note(
                        f"    Retry {attempt + 1}/{max_retries - 1} after timeout "
                        f"(waiting {delay:.0f}s before retry)..."
                    )
                    time.sleep(delay)
                    continue
                # Final attempt failed.
                raise GitHubClientError(
                    f"gh CLI command timed out after {max_retries} retries"
                ) from e

            except subprocess.CalledProcessError as e:
                # Don't retry on non-timeout errors.
                raise GitHubClientError(f"gh CLI command failed: {e.stderr}") from e

        # Should not reach here, but safety fallback.
        raise GitHubClientError(
            f"gh CLI command failed after {max_retries} retries"
        ) from last_error

    def check_cli_available(self) -> bool:
        """Check if gh CLI is installed.

        Returns:
            True if gh CLI is available, False otherwise
        """
        try:
            subprocess.run(
                ["gh", "--version"],
                capture_output=True,
                check=True,
                timeout=5,
            )
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

    def check_authenticated(self) -> bool:
        """Check if gh CLI is authenticated.

        Returns:
            True if authenticated, False otherwise
        """
        try:
            subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True,
                check=True,
                timeout=5,
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def get_runs_for_pr(
        self, pr_number: int, status: str | None = None, limit: int | None = 10
    ) -> list[WorkflowRun]:
        """Get workflow runs for a pull request.

        Args:
            pr_number: Pull request number
            status: Filter by run status (e.g., 'failure', 'success')
            limit: Maximum number of runs to fetch (None = fetch all runs for commit)

        Returns:
            List of WorkflowRun objects, sorted by creation time (newest first)

        Raises:
            GitHubClientError: If gh CLI command fails
        """
        # First, get the head SHA for the PR.
        cmd = [
            "gh",
            "pr",
            "view",
            str(pr_number),
            "--repo",
            self.repo,
            "--json",
            "headRefOid,headRefName",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            pr_data = json.loads(result.stdout)
            head_sha = pr_data.get("headRefOid")

            if not head_sha:
                raise GitHubClientError(f"Could not get head SHA for PR #{pr_number}")

            # Now fetch runs for this commit/branch.
            return self.get_runs_for_commit(head_sha, status, limit)

        except subprocess.CalledProcessError as e:
            raise GitHubClientError(
                f"Failed to fetch PR #{pr_number}: {e.stderr}"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise GitHubClientError("gh CLI command timed out") from e
        except json.JSONDecodeError as e:
            raise GitHubClientError(f"Failed to parse gh CLI output: {e}") from e

    def get_runs_for_commit(
        self, commit_sha: str, status: str | None = None, limit: int | None = 10
    ) -> list[WorkflowRun]:
        """Get workflow runs for a specific commit.

        Args:
            commit_sha: Git commit SHA (can be short or full)
            status: Filter by run status (e.g., 'failure', 'success')
            limit: Maximum number of runs to fetch (None = fetch all runs)

        Returns:
            List of WorkflowRun objects, sorted by creation time (newest first)

        Raises:
            GitHubClientError: If gh CLI command fails
        """
        cmd = [
            "gh",
            "run",
            "list",
            "--repo",
            self.repo,
            "--commit",
            commit_sha,
            "--json",
            "databaseId,workflowName,conclusion,createdAt,headBranch,displayTitle",
        ]

        if limit is not None:
            cmd.extend(["--limit", str(limit)])

        if status:
            cmd.extend(["--status", status])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            runs_data = json.loads(result.stdout)

            runs = []
            for run_data in runs_data:
                runs.append(
                    WorkflowRun(
                        run_id=str(run_data["databaseId"]),
                        workflow_name=run_data.get("workflowName", ""),
                        conclusion=run_data.get("conclusion", ""),
                        created_at=run_data.get("createdAt", ""),
                        head_branch=run_data.get("headBranch", ""),
                        display_title=run_data.get("displayTitle", ""),
                    )
                )
            return runs

        except subprocess.CalledProcessError as e:
            raise GitHubClientError(
                f"Failed to fetch runs for commit {commit_sha}: {e.stderr}"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise GitHubClientError("gh CLI command timed out") from e
        except json.JSONDecodeError as e:
            raise GitHubClientError(f"Failed to parse gh CLI output: {e}") from e

    def get_all_runs(
        self,
        status: str | None = None,
        limit: int = 10,
        created_since: str | None = None,
    ) -> list[WorkflowRun]:
        """Get workflow runs from ALL branches (not filtered by branch).

        Args:
            status: Filter by run status (e.g., 'failure', 'success')
            limit: Maximum number of runs to fetch
            created_since: Filter runs created after this date (ISO format, e.g., '2025-01-01')

        Returns:
            List of WorkflowRun objects from all branches, sorted by creation time (newest first)

        Raises:
            GitHubClientError: If gh CLI command fails
        """
        cmd = [
            "gh",
            "run",
            "list",
            "--repo",
            self.repo,
            # No --branch flag = fetch from ALL branches.
            "--json",
            "databaseId,workflowName,conclusion,createdAt,headBranch,displayTitle",
            "--limit",
            str(limit),
        ]

        if status:
            cmd.extend(["--status", status])

        if created_since:
            # Format: >=YYYY-MM-DD for server-side filtering.
            cmd.extend(["--created", f">={created_since}"])

        try:
            # Use retry helper with 120s timeout per attempt.
            result = self._run_gh_command_with_retry(cmd, timeout=120)
            runs_data = json.loads(result.stdout)

            runs = []
            for run_data in runs_data:
                runs.append(
                    WorkflowRun(
                        run_id=str(run_data["databaseId"]),
                        workflow_name=run_data.get("workflowName", ""),
                        conclusion=run_data.get("conclusion", ""),
                        created_at=run_data.get("createdAt", ""),
                        head_branch=run_data.get("headBranch", ""),
                        display_title=run_data.get("displayTitle", ""),
                    )
                )
            return runs

        except subprocess.CalledProcessError as e:
            raise GitHubClientError(
                f"Failed to fetch runs: {e.stderr if e.stderr else str(e)}"
            ) from e

    def get_runs_for_branch(
        self,
        branch: str,
        status: str | None = None,
        limit: int = 10,
        created_since: str | None = None,
    ) -> list[WorkflowRun]:
        """Get workflow runs for a specific branch.

        Args:
            branch: Branch name to filter by
            status: Filter by run status (e.g., 'failure', 'success')
            limit: Maximum number of runs to fetch
            created_since: Filter runs created after this date (ISO format, e.g., '2025-01-01')

        Returns:
            List of WorkflowRun objects, sorted by creation time (newest first)

        Raises:
            GitHubClientError: If gh CLI command fails
        """
        cmd = [
            "gh",
            "run",
            "list",
            "--repo",
            self.repo,
            "--branch",
            branch,
            "--json",
            "databaseId,workflowName,conclusion,createdAt,headBranch,displayTitle",
            "--limit",
            str(limit),
        ]

        if status:
            cmd.extend(["--status", status])

        if created_since:
            # Format: >=YYYY-MM-DD for server-side filtering.
            cmd.extend(["--created", f">={created_since}"])

        try:
            # Use retry helper with 120s timeout per attempt.
            result = self._run_gh_command_with_retry(cmd, timeout=120)
            runs_data = json.loads(result.stdout)

            runs = []
            for run_data in runs_data:
                runs.append(
                    WorkflowRun(
                        run_id=str(run_data["databaseId"]),
                        workflow_name=run_data.get("workflowName", ""),
                        conclusion=run_data.get("conclusion", ""),
                        created_at=run_data.get("createdAt", ""),
                        head_branch=run_data.get("headBranch", ""),
                        display_title=run_data.get("displayTitle", ""),
                    )
                )
            return runs

        except GitHubClientError:
            # Re-raise errors from retry helper.
            raise
        except json.JSONDecodeError as e:
            raise GitHubClientError(f"Failed to parse gh CLI output: {e}") from e

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Get details for a specific workflow run.

        Args:
            run_id: Workflow run ID

        Returns:
            Run data as dictionary, or None if not found

        Raises:
            GitHubClientError: If gh CLI command fails
        """
        cmd = [
            "gh",
            "run",
            "view",
            run_id,
            "--repo",
            self.repo,
            "--json",
            "jobs,conclusion,createdAt,displayTitle,headBranch,workflowName,headSha,event,url,number,status,attempt,startedAt,updatedAt,name,workflowDatabaseId",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            return json.loads(result.stdout)

        except subprocess.CalledProcessError as e:
            if "could not resolve to a" in e.stderr.lower():
                return None
            raise GitHubClientError(f"Failed to fetch run {run_id}: {e.stderr}") from e
        except subprocess.TimeoutExpired as e:
            raise GitHubClientError("gh CLI command timed out") from e
        except json.JSONDecodeError as e:
            raise GitHubClientError(f"Failed to parse gh CLI output: {e}") from e

    def get_jobs(self, run_id: str) -> list[Job]:
        """Get all jobs for a workflow run.

        Args:
            run_id: Workflow run ID

        Returns:
            List of Job objects

        Raises:
            GitHubClientError: If gh CLI command fails
        """
        run_data = self.get_run(run_id)
        if not run_data:
            return []

        jobs_data = run_data.get("jobs", [])
        jobs = []

        for job_data in jobs_data:
            jobs.append(
                Job(
                    job_id=str(job_data["databaseId"]),
                    name=job_data.get("name", ""),
                    conclusion=job_data.get("conclusion", ""),
                    runner_name=job_data.get("runnerName"),
                    started_at=job_data.get("startedAt", ""),
                    completed_at=job_data.get("completedAt", ""),
                )
            )

        return jobs

    def get_failed_jobs(self, run_id: str) -> list[Job]:
        """Get only failed jobs for a workflow run.

        Args:
            run_id: Workflow run ID

        Returns:
            List of Job objects with conclusion == 'failure'

        Raises:
            GitHubClientError: If gh CLI command fails
        """
        all_jobs = self.get_jobs(run_id)
        return [job for job in all_jobs if job.conclusion == "failure"]

    def get_job_metadata(self, job_id: str) -> dict:
        """Get metadata for a specific job (including run_id).

        Args:
            job_id: Job ID

        Returns:
            Dictionary with job metadata including run_id

        Raises:
            GitHubClientError: If gh CLI command fails or job not found
        """
        try:
            cmd = [
                "gh",
                "api",
                f"/repos/{self.repo}/actions/jobs/{job_id}",
                "--jq",
                "{job_id: .id, run_id: .run_id, name: .name, status: .status, conclusion: .conclusion}",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)

        except subprocess.CalledProcessError as e:
            raise GitHubClientError(f"Failed to fetch job {job_id}: {e.stderr}") from e
        except json.JSONDecodeError as e:
            raise GitHubClientError(f"Failed to parse job metadata: {e}") from e

    def get_job_log(self, run_id: str, job_id: str) -> str | None:
        """Get the log for a specific job.

        Args:
            run_id: Workflow run ID
            job_id: Job ID

        Returns:
            Log content as string, or None if not available

        Raises:
            GitHubClientError: If gh CLI command fails
        """
        cmd = [
            "gh",
            "run",
            "view",
            run_id,
            "--repo",
            self.repo,
            "--log",
            "--job",
            job_id,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
            return result.stdout

        except subprocess.CalledProcessError as e:
            # Log might not be available yet or run was deleted.
            if "no logs found" in e.stderr.lower():
                return None
            raise GitHubClientError(
                f"Failed to fetch log for job {job_id}: {e.stderr}"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise GitHubClientError(
                "gh CLI command timed out while fetching log"
            ) from e

    def get_job_annotations(self, job_id: str) -> list[dict[str, Any]]:
        """Get annotations for a specific job.

        Annotations contain error messages from GitHub Actions infrastructure,
        including runner crashes, disk full errors, and other system failures.

        Args:
            job_id: Job ID

        Returns:
            List of annotation dictionaries (empty if no annotations or job not found)

        Raises:
            GitHubClientError: If gh CLI command fails (except 404)
        """
        cmd = [
            "gh",
            "api",
            f"/repos/{self.repo}/check-runs/{job_id}/annotations",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            annotations = json.loads(result.stdout)
            return annotations if isinstance(annotations, list) else []

        except subprocess.CalledProcessError as e:
            # 404 means no annotations exist (normal for successful jobs).
            if "could not resolve to" in e.stderr.lower() or "404" in e.stderr:
                return []
            raise GitHubClientError(
                f"Failed to fetch annotations for job {job_id}: {e.stderr}"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise GitHubClientError(
                "gh CLI command timed out while fetching annotations"
            ) from e
        except json.JSONDecodeError as e:
            raise GitHubClientError(f"Failed to parse annotations: {e}") from e


def check_gh_cli_setup() -> tuple[bool, str]:
    """Check if gh CLI is properly set up.

    Returns:
        Tuple of (success, error_message)
        If success is True, error_message is empty.
        If success is False, error_message contains helpful guidance.
    """
    client = GitHubClient()

    # Check if gh CLI is installed.
    if not client.check_cli_available():
        return (
            False,
            "GitHub CLI (gh) not found. Install it from: https://cli.github.com",
        )

    # Check if authenticated.
    if not client.check_authenticated():
        return (
            False,
            "GitHub CLI is not authenticated. Run: gh auth login",
        )

    return (True, "")
