# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for GitHub CLI wrapper."""

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project tools/utils to path for imports.
sys.path.insert(0, str(Path(__file__).parents[2]))

from ci.core import github_client


class TestGitHubClientSetup(unittest.TestCase):
    """Tests for GitHub CLI availability and authentication."""

    @patch("subprocess.run")
    def test_check_cli_available(self, mock_run):
        """Test gh CLI availability check."""
        mock_run.return_value = MagicMock(returncode=0)

        client = github_client.GitHubClient()
        self.assertTrue(client.check_cli_available())

        # Verify command was correct.
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        self.assertEqual(args, ["gh", "--version"])

    @patch("subprocess.run")
    def test_check_cli_not_available(self, mock_run):
        """Test gh CLI not available."""
        mock_run.side_effect = FileNotFoundError()

        client = github_client.GitHubClient()
        self.assertFalse(client.check_cli_available())

    @patch("subprocess.run")
    def test_check_authenticated(self, mock_run):
        """Test gh CLI authentication check."""
        mock_run.return_value = MagicMock(returncode=0)

        client = github_client.GitHubClient()
        self.assertTrue(client.check_authenticated())

        args = mock_run.call_args[0][0]
        self.assertEqual(args, ["gh", "auth", "status"])

    @patch("subprocess.run")
    def test_check_not_authenticated(self, mock_run):
        """Test gh CLI not authenticated."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "gh")

        client = github_client.GitHubClient()
        self.assertFalse(client.check_authenticated())


class TestWorkflowRunQueries(unittest.TestCase):
    """Tests for fetching workflow runs."""

    @patch("subprocess.run")
    def test_get_runs_for_pr(self, mock_run):
        """Test fetching runs for a PR."""
        # Mock gh pr view response.
        pr_response = '{"headRefOid": "abc123", "headRefName": "main"}'
        runs_response = '[{"databaseId": 12345, "workflowName": "CI", "conclusion": "failure", "createdAt": "2025-01-01", "headBranch": "main", "displayTitle": "Test"}]'

        mock_run.side_effect = [
            MagicMock(stdout=pr_response, returncode=0),
            MagicMock(stdout=runs_response, returncode=0),
        ]

        client = github_client.GitHubClient()
        runs = client.get_runs_for_pr(12345)

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].run_id, "12345")
        self.assertEqual(runs[0].workflow_name, "CI")

        # Verify two subprocess calls (pr view, then run list).
        self.assertEqual(mock_run.call_count, 2)

    @patch("subprocess.run")
    def test_get_runs_for_commit(self, mock_run):
        """Test fetching runs for a commit."""
        runs_response = '[{"databaseId": 67890, "workflowName": "PkgCI", "conclusion": "failure", "createdAt": "2025-01-01", "headBranch": "main", "displayTitle": "Test"}]'

        mock_run.return_value = MagicMock(stdout=runs_response, returncode=0)

        client = github_client.GitHubClient()
        runs = client.get_runs_for_commit("abc123def")

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].run_id, "67890")

        # Verify command args.
        args = mock_run.call_args[0][0]
        self.assertIn("--commit", args)
        self.assertIn("abc123def", args)

    @patch("subprocess.run")
    def test_get_runs_for_commit_with_status(self, mock_run):
        """Test fetching runs with status filter."""
        runs_response = "[]"

        mock_run.return_value = MagicMock(stdout=runs_response, returncode=0)

        client = github_client.GitHubClient()
        client.get_runs_for_commit("abc123", status="success")

        # Verify status flag included.
        args = mock_run.call_args[0][0]
        self.assertIn("--status", args)
        self.assertIn("success", args)

    @patch("subprocess.run")
    def test_get_runs_for_branch(self, mock_run):
        """Test fetching runs for a branch."""
        runs_response = '[{"databaseId": 11111, "workflowName": "Test", "conclusion": "failure", "createdAt": "2025-01-01", "headBranch": "main", "displayTitle": "Test"}]'

        mock_run.return_value = MagicMock(stdout=runs_response, returncode=0)

        client = github_client.GitHubClient()
        runs = client.get_runs_for_branch("main")

        self.assertEqual(len(runs), 1)

        # Verify command args.
        args = mock_run.call_args[0][0]
        self.assertIn("--branch", args)
        self.assertIn("main", args)


class TestJobQueries(unittest.TestCase):
    """Tests for fetching job details."""

    @patch("subprocess.run")
    def test_get_jobs(self, mock_run):
        """Test fetching jobs for a run."""
        run_response = '{"jobs": [{"databaseId": 111, "name": "test_job", "conclusion": "failure", "startedAt": "2025-01-01", "completedAt": "2025-01-01"}]}'

        mock_run.return_value = MagicMock(stdout=run_response, returncode=0)

        client = github_client.GitHubClient()
        jobs = client.get_jobs("12345")

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].job_id, "111")
        self.assertEqual(jobs[0].name, "test_job")

    @patch("subprocess.run")
    def test_get_failed_jobs(self, mock_run):
        """Test fetching only failed jobs."""
        run_response = '{"jobs": [{"databaseId": 111, "name": "failed", "conclusion": "failure", "startedAt": "2025-01-01", "completedAt": "2025-01-01"}, {"databaseId": 222, "name": "passed", "conclusion": "success", "startedAt": "2025-01-01", "completedAt": "2025-01-01"}]}'

        mock_run.return_value = MagicMock(stdout=run_response, returncode=0)

        client = github_client.GitHubClient()
        failed = client.get_failed_jobs("12345")

        # Should only return failed job.
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].name, "failed")

    @patch("subprocess.run")
    def test_get_job_log(self, mock_run):
        """Test fetching job log."""
        log_content = "This is a test log\nWith multiple lines\n"

        mock_run.return_value = MagicMock(stdout=log_content, returncode=0)

        client = github_client.GitHubClient()
        log = client.get_job_log("12345", "67890")

        self.assertEqual(log, log_content)

        # Verify command args.
        args = mock_run.call_args[0][0]
        self.assertIn("--log", args)
        self.assertIn("--job", args)
        self.assertIn("67890", args)


class TestErrorHandling(unittest.TestCase):
    """Tests for error handling."""

    @patch("subprocess.run")
    def test_timeout_error(self, mock_run):
        """Test handling subprocess timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("gh", 30)

        client = github_client.GitHubClient()

        with self.assertRaises(github_client.GitHubClientError):
            client.get_runs_for_commit("abc123")

    @patch("subprocess.run")
    def test_json_parse_error(self, mock_run):
        """Test handling invalid JSON response."""
        mock_run.return_value = MagicMock(stdout="invalid json", returncode=0)

        client = github_client.GitHubClient()

        with self.assertRaises(github_client.GitHubClientError):
            client.get_runs_for_commit("abc123")


if __name__ == "__main__":
    unittest.main()
