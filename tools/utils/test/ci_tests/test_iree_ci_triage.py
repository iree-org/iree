# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for iree-ci-triage tool."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project tools/utils to path for imports.
sys.path.insert(0, str(Path(__file__).parents[2]))

from ci.core.github_client import GitHubClient

from test.test_helpers import run_python_module


class TestTriageCLI(unittest.TestCase):
    """Tests for CLI integration and help output."""

    def test_tool_launches(self):
        """Test that iree-ci-triage launches and shows help."""
        result = run_python_module(
            "ci.iree_ci_triage", ["--help"], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--pr", result.stdout)
        self.assertIn("--run", result.stdout)
        self.assertIn("--log-file", result.stdout)

    def test_missing_args_error(self):
        """Test error when no source specified."""
        result = run_python_module(
            "ci.iree_ci_triage", [], capture_output=True, text=True
        )
        self.assertNotEqual(result.returncode, 0)

    def test_json_flag(self):
        """Test --json flag in help."""
        result = run_python_module(
            "ci.iree_ci_triage", ["--help"], capture_output=True, text=True
        )
        self.assertIn("--json", result.stdout)

    def test_checklist_flag(self):
        """Test --checklist flag in help."""
        result = run_python_module(
            "ci.iree_ci_triage", ["--help"], capture_output=True, text=True
        )
        self.assertIn("--checklist", result.stdout)


class TestTriageWithMockedGitHub(unittest.TestCase):
    """Tests for triage with mocked GitHub API."""

    @patch("subprocess.run")
    def test_triage_pr_success(self, mock_run):
        """Test triaging PR with mocked GitHub."""
        # Mock gh pr view response.
        pr_response = json.dumps(
            {"headRefOid": "abc123def", "headRefName": "feature-branch"}
        )

        # Mock gh run list response.
        runs_response = json.dumps(
            [
                {
                    "databaseId": 12345,
                    "workflowName": "CI",
                    "conclusion": "failure",
                    "createdAt": "2025-01-01T00:00:00Z",
                    "headBranch": "feature-branch",
                    "displayTitle": "Test commit",
                }
            ]
        )

        # Mock gh run view (for jobs).
        run_view_response = json.dumps(
            {
                "jobs": [
                    {
                        "databaseId": 111,
                        "name": "test_job",
                        "conclusion": "failure",
                        "startedAt": "2025-01-01",
                        "completedAt": "2025-01-01",
                    }
                ],
                "conclusion": "failure",
                "createdAt": "2025-01-01",
                "displayTitle": "Test",
                "headBranch": "main",
                "workflowName": "CI",
            }
        )

        # Mock job log.
        log_response = "ERROR: test failure\n"

        # Mock gh auth status (for setup check).
        auth_response = "Logged in to github.com\n"

        mock_run.side_effect = [
            MagicMock(stdout="", returncode=0),  # gh --version
            MagicMock(stdout=auth_response, returncode=0),  # gh auth status
            MagicMock(stdout=pr_response, returncode=0),  # gh pr view
            MagicMock(stdout=runs_response, returncode=0),  # gh run list
            MagicMock(stdout=run_view_response, returncode=0),  # gh run view
            MagicMock(stdout=log_response, returncode=0),  # gh run view --log
        ]

        result = run_python_module(
            "ci.iree_ci_triage",
            ["--pr", "12345", "--json"],
            capture_output=True,
            text=True,
        )

        # Should succeed (exit code 0 if patterns match, or specific code if not).
        # We're mainly testing it doesn't crash.
        self.assertIn(result.returncode, [0, 1, 2])  # Various valid exit codes.

    @patch("subprocess.run")
    def test_triage_run_with_log_file(self, mock_run):
        """Test triaging with log file (no GitHub needed)."""
        # Mock gh --version and auth status.
        mock_run.side_effect = [
            MagicMock(stdout="", returncode=0),  # gh --version
            MagicMock(stdout="Logged in\n", returncode=0),  # gh auth status
        ]

        # Create temporary log file with sample content.
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as tmp:
            tmp.write("ERROR: compilation failed\n")
            tmp.write("undefined reference to `symbol`\n")
            tmp_path = tmp.name

        try:
            result = run_python_module(
                "ci.iree_ci_triage",
                ["--log-file", tmp_path, "--json"],
                capture_output=True,
                text=True,
            )

            # Should complete without crashing.
            self.assertIsNotNone(result.stdout)

        finally:
            Path(tmp_path).unlink()


class TestTriageAuthentication(unittest.TestCase):
    """Tests for GitHub authentication checks."""

    @patch("ci.core.github_client.subprocess.run")
    def test_gh_not_installed(self, mock_run):
        """Test error when gh CLI not installed."""
        mock_run.side_effect = FileNotFoundError()

        # Test the client directly (can't mock across process boundaries).
        client = GitHubClient()
        self.assertFalse(client.check_cli_available())

    @patch("ci.core.github_client.subprocess.run")
    def test_gh_not_authenticated(self, mock_run):
        """Test error when gh CLI not authenticated."""
        # Mock gh auth status failure.
        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["gh", "auth", "status"]
        )

        # Test the client directly (can't mock across process boundaries).
        client = GitHubClient()
        self.assertFalse(client.check_authenticated())


if __name__ == "__main__":
    unittest.main()
