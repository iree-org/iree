# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for iree-ci-garden corpus management tool."""

import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project tools/utils to path for imports.
sys.path.insert(0, str(Path(__file__).parents[2]))

from ci.core.classifier import Classifier
from ci.core.corpus import Corpus
from ci.core.fetcher import GitHubFetcher
from ci.core.github_client import GitHubClient, GitHubClientError

from test.test_helpers import run_python_module


class TestCLIIntegration(unittest.TestCase):
    """Tests for CLI integration and help output."""

    def test_tool_launches(self):
        """Test that iree-ci-garden launches and shows help."""
        result = run_python_module(
            "ci.iree_ci_garden", ["--help"], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("fetch", result.stdout)
        self.assertIn("classify", result.stdout)
        self.assertIn("status", result.stdout)
        self.assertIn("search", result.stdout)

    def test_fetch_help(self):
        """Test fetch subcommand help."""
        result = run_python_module(
            "ci.iree_ci_garden", ["fetch", "--help"], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--since", result.stdout)
        self.assertIn("--pr", result.stdout)
        self.assertIn("--run", result.stdout)

    def test_classify_help(self):
        """Test classify subcommand help."""
        result = run_python_module(
            "ci.iree_ci_garden",
            ["classify", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--run", result.stdout)

    def test_status_help(self):
        """Test status subcommand help."""
        result = run_python_module(
            "ci.iree_ci_garden", ["status", "--help"], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--json", result.stdout)

    def test_missing_command_error(self):
        """Test error when no command specified."""
        result = run_python_module(
            "ci.iree_ci_garden", [], capture_output=True, text=True
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No command specified", result.stderr)


class TestCorpusManagement(unittest.TestCase):
    """Tests for corpus directory management and metadata."""

    def setUp(self):
        """Create temporary corpus directory."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.corpus = Corpus(Path(self.temp_dir.name))

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_corpus_initialization(self):
        """Test corpus directory structure creation."""
        self.assertTrue(self.corpus.corpus_dir.exists())
        self.assertTrue(self.corpus.logs_dir.exists())
        self.assertTrue(self.corpus.runs_dir.exists())
        self.assertTrue(self.corpus.classification_dir.exists())
        self.assertTrue(self.corpus.unrecognized_dir.exists())
        self.assertTrue(self.corpus.config_path.exists())

    def test_config_version(self):
        """Test that config has correct version."""
        config = json.loads(self.corpus.config_path.read_text())
        self.assertEqual(config["corpus_version"], "2.0")

    def test_add_run_success(self):
        """Test adding run metadata."""
        run_data = {
            "run_id": "12345",
            "head_sha": "abc123def456",
            "branch": "main",
            "workflow": "CI",
            "event": "push",
            "pr_number": None,
            "conclusion": "failure",
            "attempt": 1,
            "created_at": "2025-01-01T00:00:00Z",
        }
        result = self.corpus.add_run(run_data)
        self.assertTrue(result)
        self.assertTrue(self.corpus.has_run("12345"))

    def test_duplicate_run_rejected(self):
        """Test that duplicate runs are rejected."""
        run_data = {
            "run_id": "12345",
            "head_sha": "abc123",
            "branch": "main",
            "workflow": "CI",
            "conclusion": "failure",
            "created_at": "2025-01-01T00:00:00Z",
        }
        self.corpus.add_run(run_data)
        result = self.corpus.add_run(run_data)
        self.assertFalse(result)

    def test_save_and_get_log(self):
        """Test saving and retrieving log content."""
        log_content = "ERROR: test failure\nStack trace...\n"
        self.corpus.save_log("12345", "job_111", log_content)

        log_path = self.corpus.get_log_path("12345", "job_111")
        self.assertIsNotNone(log_path)
        self.assertTrue(log_path.exists())
        self.assertEqual(log_path.read_text(), log_content)

    def test_get_stats(self):
        """Test corpus statistics."""
        # Empty corpus.
        stats = self.corpus.get_stats()
        self.assertEqual(stats["total_runs"], 0)
        self.assertEqual(stats["total_logs"], 0)

        # Add run and log.
        self.corpus.add_run(
            {"run_id": "12345", "created_at": "2025-01-01", "branch": "main"}
        )
        self.corpus.save_log("12345", "111", "log content")

        stats = self.corpus.get_stats()
        self.assertEqual(stats["total_runs"], 1)
        self.assertEqual(stats["total_logs"], 1)


class TestFetcherWithMocks(unittest.TestCase):
    """Tests for fetcher with mocked GitHub API."""

    def setUp(self):
        """Create temporary corpus and mocked client."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.corpus = Corpus(Path(self.temp_dir.name))
        self.client = GitHubClient()

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    @patch("ci.core.github_client.subprocess.run")
    def test_fetch_single_run(self, mock_run):
        """Test fetching single run with mocked GitHub."""
        # Mock gh run view response (v2.0 schema).
        run_response = json.dumps(
            {
                "workflowName": "CI",
                "workflowDatabaseId": 67146182,
                "name": "CI",
                "headSha": "abc123def456",
                "headBranch": "main",
                "event": "push",
                "number": 12345,
                "status": "completed",
                "conclusion": "failure",
                "attempt": 1,
                "createdAt": "2025-01-01T00:00:00Z",
                "startedAt": "2025-01-01T00:00:00Z",
                "updatedAt": "2025-01-01T01:00:00Z",
                "displayTitle": "Test commit",
                "url": "https://github.com/test/repo/actions/runs/12345",
                "jobs": [
                    {
                        "databaseId": 111,
                        "name": "test_job",
                        "conclusion": "failure",
                        "startedAt": "2025-01-01",
                        "completedAt": "2025-01-01",
                    }
                ],
            }
        )

        # Mock responses for full fetch flow:
        # 1. get_run (direct call)
        # 2. get_run (from get_failed_jobs)
        # 3. get_job_annotations (returns empty list)
        # 4. get_job_log (returns log content)
        log_response = "Sample log content\nERROR: test failure\n"

        mock_run.side_effect = [
            MagicMock(stdout=run_response, returncode=0),  # get_run (direct)
            MagicMock(stdout=run_response, returncode=0),  # get_run (from get_jobs)
            MagicMock(stdout="[]", returncode=0),  # get_job_annotations
            MagicMock(stdout=log_response, returncode=0),  # get_job_log
        ]

        fetcher = GitHubFetcher(self.client, self.corpus)
        result = fetcher.fetch_run("12345")

        self.assertEqual(result.runs_new, 1)
        self.assertEqual(result.logs_fetched, 1)
        self.assertTrue(self.corpus.has_run("12345"))


class TestTimeChunking(unittest.TestCase):
    """Tests for time-based chunking logic."""

    def setUp(self):
        """Create temporary corpus and fetcher."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.corpus = Corpus(Path(self.temp_dir.name))
        self.client = GitHubClient()
        self.fetcher = GitHubFetcher(self.client, self.corpus)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_chunk_time_range_single_month(self):
        """Test chunking a single month."""
        since = datetime(2025, 1, 1, tzinfo=timezone.utc)
        until = datetime(2025, 1, 31, tzinfo=timezone.utc)

        chunks = self.fetcher._chunk_time_range(since, until, chunk_days=30)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0][0], since)
        self.assertEqual(chunks[0][1], until)

    def test_chunk_time_range_multiple_months(self):
        """Test chunking multiple months."""
        since = datetime(2025, 1, 1, tzinfo=timezone.utc)
        until = datetime(2025, 11, 18, tzinfo=timezone.utc)

        chunks = self.fetcher._chunk_time_range(since, until, chunk_days=30)

        # Should create ~10-11 chunks (320+ days / 30 = ~10-11 chunks).
        self.assertGreater(len(chunks), 9)
        self.assertLess(len(chunks), 12)

        # First chunk should start at since.
        self.assertEqual(chunks[0][0], since)

        # Last chunk should end at until.
        self.assertEqual(chunks[-1][1], until)

        # Chunks should be contiguous.
        for i in range(len(chunks) - 1):
            self.assertEqual(chunks[i][1], chunks[i + 1][0])


class TestRetryLogic(unittest.TestCase):
    """Tests for retry logic with timeout."""

    def setUp(self):
        """Create GitHub client."""
        self.client = GitHubClient()

    @patch("subprocess.run")
    @patch("time.sleep")  # Mock sleep to speed up tests.
    def test_retry_on_timeout(self, mock_sleep, mock_run):
        """Test that retry happens on timeout."""
        # First two attempts timeout, third succeeds.
        mock_run.side_effect = [
            subprocess.TimeoutExpired(["gh", "run", "list"], 120),
            subprocess.TimeoutExpired(["gh", "run", "list"], 120),
            MagicMock(stdout="[]", returncode=0),
        ]

        # Should succeed on third attempt.
        result = self.client._run_gh_command_with_retry(
            ["gh", "run", "list"], timeout=120
        )

        self.assertIsNotNone(result)
        self.assertEqual(mock_run.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)  # Sleep between retries.

    @patch("subprocess.run")
    @patch("time.sleep")
    def test_retry_exhausted(self, mock_sleep, mock_run):
        """Test that error is raised after all retries exhausted."""
        # All attempts timeout.
        mock_run.side_effect = subprocess.TimeoutExpired(["gh", "run", "list"], 120)

        # Should raise error after 3 attempts.
        with self.assertRaises(GitHubClientError) as context:
            self.client._run_gh_command_with_retry(
                ["gh", "run", "list"], timeout=120, max_retries=3
            )

        self.assertIn("timed out after 3 retries", str(context.exception))
        self.assertEqual(mock_run.call_count, 3)

    @patch("subprocess.run")
    def test_no_retry_on_non_timeout_error(self, mock_run):
        """Test that non-timeout errors are not retried."""
        # Simulate CalledProcessError (non-timeout).
        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["gh", "run", "list"], stderr="API error"
        )

        # Should raise immediately without retry.
        with self.assertRaises(GitHubClientError) as context:
            self.client._run_gh_command_with_retry(["gh", "run", "list"], timeout=120)

        self.assertIn("gh CLI command failed", str(context.exception))
        self.assertEqual(mock_run.call_count, 1)  # Only one attempt.


class TestClassifierWithMocks(unittest.TestCase):
    """Tests for classifier with mocked iree-ci-triage."""

    def setUp(self):
        """Create temporary corpus."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.corpus = Corpus(Path(self.temp_dir.name))

        # Add sample run and log.
        self.corpus.add_run(
            {
                "run_id": "12345",
                "head_sha": "abc123",
                "branch": "main",
                "workflow": "CI",
                "conclusion": "failure",
                "created_at": "2025-01-01T00:00:00Z",
            }
        )
        # Include "FAILED:" marker to activate BuildErrorExtractor.
        self.corpus.save_log(
            "12345",
            "111",
            "FAILED: compiler/CMakeFiles/test.o\n"
            "undefined reference to `missing_symbol'\n",
        )

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_classify_recognized_failure(self):
        """Test classification with recognized pattern."""
        # The log already contains "undefined reference" pattern which
        # BuildErrorExtractor recognizes as a linker error.
        classifier = Classifier(self.corpus)
        result = classifier.classify_log("12345", "111")

        self.assertIsNotNone(result)
        self.assertTrue(result.recognized)
        # Build errors are recognized.
        self.assertTrue(len(result.categories) > 0)

    def test_classify_unrecognized_failure(self):
        """Test classification with unrecognized pattern."""
        # Save a log with no recognizable pattern.
        self.corpus.save_log("12345", "999", "Random gibberish that matches nothing\n")

        classifier = Classifier(self.corpus)
        result = classifier.classify_log("12345", "999")

        self.assertIsNotNone(result)
        self.assertFalse(result.recognized)


class TestStatusCommand(unittest.TestCase):
    """Tests for status command integration."""

    def test_status_with_empty_corpus(self):
        """Test status on empty corpus."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # --corpus-dir must come before subcommand.
            result = run_python_module(
                "ci.iree_ci_garden",
                ["--corpus-dir", temp_dir, "status"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0)
            self.assertIn("Total runs: 0", result.stdout)
            self.assertIn("Total logs: 0", result.stdout)

    def test_status_json_output(self):
        """Test JSON output format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # --corpus-dir must come before subcommand.
            result = run_python_module(
                "ci.iree_ci_garden",
                ["--corpus-dir", temp_dir, "status", "--json"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0)
            data = json.loads(result.stdout)
            self.assertIn("statistics", data)
            self.assertIn("recognition", data)


if __name__ == "__main__":
    unittest.main()
