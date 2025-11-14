# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""IREE CI Failure Triage Tool.

Fetches CI failure logs from GitHub Actions and performs automated triage:
- Identifies errors (compile errors, test failures, GPU crashes, sanitizer issues, etc.)
- Extracts actionable issues vs infrastructure failures
- Generates fix checklists sorted by severity
- Supports both human-readable and JSON output

Usage:
    iree-ci-triage --run 12345678                  # Triage all failed jobs in a run
    iree-ci-triage --pr 12345                      # Triage latest failure from PR
    iree-ci-triage --commit abc123def              # Triage latest failure from commit
    iree-ci-triage --branch main                   # Triage latest failure from branch
    iree-ci-triage --run 12345678 --job 987654321  # Triage specific job
    iree-ci-triage --run 12345678 --json           # JSON output for automation

Examples:
    # Human-readable triage of all failures in a run
    iree-ci-triage --run 12345678

    # Triage latest failure from a pull request
    iree-ci-triage --pr 12345

    # Find successful runs for comparison
    iree-ci-triage --pr 12345 --status success

    # JSON output for piping to other tools
    iree-ci-triage --run 12345678 --json | jq '.issues'
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Import IREE tool utilities.
sys.path.insert(0, str(Path(__file__).parent.parent))
# Import extractors.
from common import cli, console, exit_codes
from common.extractors.bazel_error import BazelErrorExtractor
from common.extractors.build_error import BuildErrorExtractor
from common.extractors.cmake_error import CMakeErrorExtractor
from common.extractors.codeql_error import CodeQLErrorExtractor
from common.extractors.ctest_error import CTestErrorExtractor
from common.extractors.infrastructure_flake import InfrastructureFlakeExtractor
from common.extractors.mlir_compiler import MLIRCompilerExtractor
from common.extractors.onnx_test import ONNXTestExtractor
from common.extractors.precommit import PrecommitErrorExtractor
from common.extractors.pytest_error import PytestErrorExtractor
from common.extractors.sanitizer import SanitizerExtractor
from common.log_buffer import LogBuffer
from common.triage_engine import TriageEngine
from common.triage_result import TriageResult

# Import CI modules.
from ci.core import github_client


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="IREE CI failure triage tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input source (mutually exclusive).
    input_group = parser.add_argument_group("Input")
    source_group = input_group.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--run",
        metavar="RUN_ID",
        help="GitHub Actions workflow run ID",
    )
    source_group.add_argument(
        "--pr",
        type=int,
        metavar="PR_NUMBER",
        help="Pull request number (uses latest failed run)",
    )
    source_group.add_argument(
        "--commit",
        metavar="SHA",
        help="Commit SHA (uses latest failed run)",
    )
    source_group.add_argument(
        "--branch",
        metavar="BRANCH",
        help="Branch name (uses latest failed run)",
    )
    source_group.add_argument(
        "--job",
        metavar="JOB_ID",
        help="Specific job ID to triage (fetches run automatically)",
    )
    source_group.add_argument(
        "--log-file",
        metavar="FILE",
        type=Path,
        help="Analyze local log file instead of fetching from GitHub (for testing)",
    )
    input_group.add_argument(
        "--job-name",
        metavar="NAME",
        help="Job name (optional, defaults to log filename when using --log-file)",
    )
    input_group.add_argument(
        "--repo",
        default="iree-org/iree",
        metavar="OWNER/REPO",
        help="GitHub repository (default: iree-org/iree)",
    )
    input_group.add_argument(
        "--status",
        default="failure",
        metavar="STATUS",
        help="Filter by run status (default: failure)",
    )
    input_group.add_argument(
        "--save-logs",
        metavar="DIR",
        type=Path,
        help="Save fetched logs to directory (for debugging/testing)",
    )
    input_group.add_argument(
        "--download-all-logs",
        action="store_true",
        help="Download logs for all jobs (not just failed ones)",
    )

    # Output format.
    output_group = parser.add_argument_group("Output")
    cli.add_common_output_flags(parser)
    output_group.add_argument(
        "--checklist",
        action="store_true",
        help="Output simple checklist format (for LLMs)",
    )
    output_group.add_argument(
        "--no-context",
        action="store_true",
        help="Don't include error context in markdown output",
    )

    return parser.parse_args()


def check_dependencies(args: argparse.Namespace) -> int:
    """Check that required dependencies are available.

    Args:
        args: Parsed command line arguments

    Returns:
        exit_codes.SUCCESS if all dependencies OK, error code otherwise
    """
    # Check gh CLI.
    success, error_msg = github_client.check_gh_cli_setup()
    if not success:
        console.error(error_msg, args=args)
        console.note(
            "Install: https://cli.github.com",
            args=args,
        )
        return exit_codes.NOT_FOUND

    return exit_codes.SUCCESS


def create_ci_triage_engine() -> TriageEngine:
    """Create TriageEngine with all extractors for CI log analysis.

    Returns:
        TriageEngine configured with all available extractors.
    """
    return TriageEngine(
        [
            SanitizerExtractor(),
            MLIRCompilerExtractor(),
            BuildErrorExtractor(),
            BazelErrorExtractor(),
            CMakeErrorExtractor(),
            CTestErrorExtractor(),
            InfrastructureFlakeExtractor(),
            ONNXTestExtractor(),
            PrecommitErrorExtractor(),
            PytestErrorExtractor(),
            CodeQLErrorExtractor(),
        ]
    )


def resolve_run_id(client: github_client.GitHubClient, args: argparse.Namespace) -> str:
    """Resolve --pr/--commit/--branch/--job to a workflow run ID.

    Args:
        client: GitHubClient instance
        args: Command line arguments

    Returns:
        Workflow run ID string

    Raises:
        SystemExit: If resolution fails
    """
    # If explicit run ID provided, use it.
    if args.run:
        return args.run

    # If standalone --job provided, fetch run_id from job metadata.
    if args.job and not (args.pr or args.commit or args.branch):
        try:
            console.note(f"Fetching run ID for job {args.job}...", args=args)
            job_metadata = client.get_job_metadata(args.job)
            run_id = str(job_metadata["run_id"])
            console.note(
                f"Job {args.job}: {job_metadata['name']} (run {run_id})",
                args=args,
            )
            return run_id
        except github_client.GitHubClientError as e:
            console.error(f"Failed to fetch job metadata: {e}", args=args)
            sys.exit(exit_codes.ERROR)

    # Otherwise, resolve PR/commit/branch to latest run.
    try:
        if args.pr:
            console.note(
                f"Fetching latest {args.status} run for PR #{args.pr}...",
                args=args,
            )
            runs = client.get_runs_for_pr(args.pr, status=args.status, limit=1)

        elif args.commit:
            console.note(
                f"Fetching latest {args.status} run for commit {args.commit}...",
                args=args,
            )
            runs = client.get_runs_for_commit(args.commit, status=args.status, limit=1)

        elif args.branch:
            console.note(
                f"Fetching latest {args.status} run for branch {args.branch}...",
                args=args,
            )
            runs = client.get_runs_for_branch(args.branch, status=args.status, limit=1)

        else:
            console.error("No run source specified", args=args)
            sys.exit(exit_codes.ERROR)

        if not runs:
            source = f"PR #{args.pr}" if args.pr else args.commit or args.branch
            console.error(
                f"No {args.status} runs found for {source}",
                args=args,
            )
            sys.exit(exit_codes.NOT_FOUND)

        run = runs[0]
        console.note(
            f"Using run {run.run_id}: {run.workflow_name} ({run.conclusion})",
            args=args,
        )
        return run.run_id

    except github_client.GitHubClientError as e:
        console.error(f"Failed to resolve run: {e}", args=args)
        sys.exit(exit_codes.ERROR)


def fetch_job_data(
    client: github_client.GitHubClient,
    run_id: str,
    job_id: Optional[str],
    args: argparse.Namespace,
) -> tuple[List[github_client.Job], List[tuple[github_client.Job, str]]]:
    """Fetch ALL jobs and download logs for failed/requested jobs.

    Args:
        client: GitHubClient instance
        run_id: Workflow run ID
        job_id: Optional specific job ID
        args: Command line arguments

    Returns:
        (all_jobs, job_logs_with_content) - ALL jobs for status, logs for failed

    Raises:
        SystemExit: If fetch fails
    """
    try:
        # Get ALL jobs to check actual GitHub API status.
        console.note(f"Fetching jobs for run {run_id}...", args=args)
        all_jobs = client.get_jobs(run_id)

        if not all_jobs:
            console.note("No jobs found in this run.", args=args)
            return ([], [])

        if job_id:
            # Specific job requested.
            target_job = next((j for j in all_jobs if j.job_id == job_id), None)

            if not target_job:
                console.error(
                    f"Job {job_id} not found in run {run_id}",
                    args=args,
                )
                sys.exit(exit_codes.NOT_FOUND)

            console.note(f"Fetching log for job {job_id}...", args=args)
            log_content = client.get_job_log(run_id, job_id)
            if not log_content:
                console.error(
                    f"Log not available for job {job_id}",
                    args=args,
                )
                sys.exit(exit_codes.NOT_FOUND)

            return (all_jobs, [(target_job, log_content)])

        # Determine which jobs to download logs for.
        if args.download_all_logs:
            jobs_to_download = all_jobs
            console.note(
                f"Downloading logs for all {len(all_jobs)} job(s)...",
                args=args,
            )
        else:
            # Only download failed jobs by default (source of truth: GitHub API).
            jobs_to_download = [j for j in all_jobs if j.conclusion == "failure"]
            failed_count = len(jobs_to_download)
            console.note(
                f"Found {failed_count} failed job(s) in run {run_id}",
                args=args,
            )

            if failed_count == 0:
                console.note("No failed jobs to analyze.", args=args)
                return (all_jobs, [])

        # Download logs.
        job_logs = []
        for i, job in enumerate(jobs_to_download, 1):
            console.note(
                f"  [{i}/{len(jobs_to_download)}] Fetching log for: {job.name}",
                args=args,
            )
            log_content = client.get_job_log(run_id, job.job_id)
            if log_content:
                job_logs.append((job, log_content))
            else:
                console.warning(
                    f"  Log not available for job {job.job_id}",
                    args=args,
                )

        return (all_jobs, job_logs)

    except github_client.GitHubClientError as e:
        console.error(f"GitHub API error: {e}", args=args)
        sys.exit(exit_codes.ERROR)


def analyze_job(
    job: github_client.Job,
    log_content: str,
    args: argparse.Namespace,
) -> TriageResult:
    """Analyze a single job log.

    Args:
        job: Job object
        log_content: Log content string
        args: Command line arguments

    Returns:
        TriageResult instance
    """
    console.note(f"Analyzing job: {job.name}", args=args)

    # Create log buffer with auto-detection of format.
    log_buffer = LogBuffer(log_content, auto_detect_format=True)

    # Run triage engine.
    engine = create_ci_triage_engine()
    result = engine.analyze(log_buffer)

    console.note(
        f"  Found {len(result.issues)} issue(s) from {len(result.extractors_run)} extractor(s)",
        args=args,
    )

    if result.extractor_errors:
        console.warn(
            f"  {len(result.extractor_errors)} extractor(s) encountered errors",
            args=args,
        )

    return result


def save_logs_to_tmp(
    run_id: str,
    job_logs: List[tuple[github_client.Job, str]],
    args: argparse.Namespace,
) -> Dict[str, Path]:
    """Save job logs to /tmp and return job_id -> path mapping.

    Args:
        run_id: Workflow run ID
        job_logs: List of (Job, log_content) tuples
        args: Command line arguments

    Returns:
        Dictionary mapping job_id -> log_path
    """
    output_dir = Path(f"/tmp/iree-ci-triage/run_{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    log_files = {}
    for job, log_content in job_logs:
        # Strip GitHub Actions prefixes to save tokens.
        log_buffer = LogBuffer(log_content, auto_detect_format=True)
        stripped_content = log_buffer.content

        # Sanitize job name for filename.
        safe_name = job.name.replace(" / ", "_").replace(" ", "_").replace("/", "_")
        log_file = output_dir / f"job_{job.job_id}_{safe_name}.log"
        log_file.write_text(stripped_content)
        log_files[job.job_id] = log_file

    if log_files and not args.quiet:
        console.note(f"Saved {len(log_files)} log(s) to: {output_dir}", args=args)

    return log_files


def is_meta_job(job_name: str) -> bool:
    """Check if a job is a meta/summary job.

    Args:
        job_name: Job name string

    Returns:
        True if job is a meta-job (aggregate, summary, etc.)
    """
    meta_keywords = ["summary", "aggregate", "pkgci_summary", "job-summary"]
    job_name_lower = job_name.lower()
    return any(keyword in job_name_lower for keyword in meta_keywords)


def output_results(
    all_jobs: List[github_client.Job],
    job_results: List[tuple[github_client.Job, TriageResult]],
    log_files: Dict[str, Path],
    args: argparse.Namespace,
) -> None:
    """Output triage results with GitHub status as source of truth.

    Args:
        all_jobs: ALL jobs from run (for status display)
        job_results: Triage results for analyzed jobs
        log_files: job_id -> Path mapping for log files
        args: Command line arguments
    """
    # Separate jobs by GitHub API conclusion.
    failed_jobs = [j for j in all_jobs if j.conclusion == "failure"]
    passed_jobs = [j for j in all_jobs if j.conclusion == "success"]
    other_jobs = [j for j in all_jobs if j.conclusion not in ("failure", "success")]

    if args.json:
        # JSON output for automation.
        output = {
            "jobs": [
                {
                    "job_id": job.job_id,
                    "name": job.name,
                    "conclusion": job.conclusion,  # GitHub API status!
                    "log_path": str(log_files.get(job.job_id, "")),
                    "extracted_issues": (
                        [
                            {
                                "title": issue.message[
                                    :100
                                ],  # Use first 100 chars of message as title.
                                "message": issue.message,
                                "severity": issue.severity.name
                                if hasattr(issue.severity, "name")
                                else str(issue.severity),
                                "extractor": issue.source_extractor,
                            }
                            for issue in result.issues
                        ]
                        if (
                            result := next(
                                (r for j, r in job_results if j.job_id == job.job_id),
                                None,
                            )
                        )
                        else []
                    ),
                    "extraction_status": (
                        "success"
                        if (
                            result := next(
                                (r for j, r in job_results if j.job_id == job.job_id),
                                None,
                            )
                        )
                        and result.issues
                        else "incomplete"
                        if result
                        else "not_analyzed"
                    ),
                }
                for job in all_jobs
            ],
            "summary": {
                "total_jobs": len(all_jobs),
                "failed": len(failed_jobs),
                "passed": len(passed_jobs),
                "other": len(other_jobs),
                "logs_downloaded": len(log_files),
            },
        }
        console.print_json(output, args=args)
        return

    # Human-readable output.
    if not failed_jobs:
        console.out("✓ All jobs passed")
        console.out(f"  Total jobs: {len(all_jobs)}")
        if log_files:
            console.out("  Logs: /tmp/iree-ci-triage/run_*")
        return

    # Show failed jobs with details.
    console.out(f"\n{'=' * 80}")
    console.out(f"FAILED JOBS ({len(failed_jobs)}/{len(all_jobs)}):")
    console.out(f"{'=' * 80}\n")

    for job in failed_jobs:
        console.out(f"✗ Job {job.job_id}: {job.name}")
        console.out("  Status: failure (from GitHub API)")

        # Show log file path.
        if job.job_id in log_files:
            console.out(f"  Log: {log_files[job.job_id]}")

        # Show extracted issues if available.
        result = next((r for j, r in job_results if j.job_id == job.job_id), None)
        if result and result.issues:
            console.out(f"\n  Extracted Issues ({len(result.issues)}):")
            for issue in result.issues[:5]:  # Limit to first 5.
                console.out(f"    • {issue.title}")
                if issue.message and len(issue.message) < 100:
                    console.out(f"      {issue.message}")
            if len(result.issues) > 5:
                console.out(f"    ... and {len(result.issues) - 5} more")
        elif result:
            console.out("\n  ⚠ WARNING: No issues extracted - manual review needed")
            if job.job_id in log_files:
                console.out(f"  See full log: {log_files[job.job_id]}")
        else:
            console.out("\n  (Log not analyzed)")

        console.out("")

    console.out(f"{'=' * 80}")
    console.out(
        f"Summary: {len(failed_jobs)} failed, {len(passed_jobs)} passed, {len(other_jobs)} other"
    )
    console.out(f"{'=' * 80}\n")


def main(args: argparse.Namespace) -> int:
    """Main entry point for iree-ci-triage.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code
    """
    # Auto-enable quiet mode for JSON output to keep stdout clean.
    if args.json:
        args.quiet = True

    # Handle --log-file (local testing mode).
    if args.log_file:
        if not args.log_file.exists():
            console.error(f"Log file not found: {args.log_file}", args=args)
            return exit_codes.NOT_FOUND

        console.note(f"Reading log from: {args.log_file}", args=args)
        log_content = args.log_file.read_text()

        # Use provided job name or default to log filename.
        job_name = args.job_name or args.log_file.stem

        # Create synthetic job object for local testing.
        job = github_client.Job(
            job_id=job_name,
            name=job_name,
            conclusion="failure",
            runner_name=None,
            started_at="",
            completed_at="",
        )
        all_jobs = [job]
        job_logs = [(job, log_content)]
        log_files = {job.job_id: args.log_file}
        any_failures = True

    else:
        # Check dependencies for GitHub mode.
        status = check_dependencies(args)
        if status != exit_codes.SUCCESS:
            return status

        # Create GitHub client.
        client = github_client.GitHubClient(repo=args.repo)

        # Resolve run ID from --pr/--commit/--branch or use --run directly.
        run_id = resolve_run_id(client, args)

        # Fetch ALL jobs and logs for failed/requested jobs.
        all_jobs, job_logs = fetch_job_data(client, run_id, args.job, args)

        # Always save logs to /tmp.
        log_files = save_logs_to_tmp(run_id, job_logs, args)

        # Also save to custom directory if requested.
        if args.save_logs:
            args.save_logs.mkdir(parents=True, exist_ok=True)
            console.note(f"Saving logs to: {args.save_logs}", args=args)
            for job, log_content in job_logs:
                # Strip log prefixes (GitHub Actions, etc.) to save tokens.
                log_buffer = LogBuffer(log_content, auto_detect_format=True)
                stripped_content = log_buffer.content

                # Sanitize job name for filename.
                safe_name = job.name.replace(" / ", "_").replace(" ", "_")
                log_file = args.save_logs / f"{run_id}_{job.job_id}_{safe_name}.log"
                log_file.write_text(stripped_content)
                console.note(f"  Saved: {log_file.name}", args=args)

        # Determine exit code from GitHub API (source of truth).
        failed_jobs = [j for j in all_jobs if j.conclusion == "failure"]
        any_failures = len(failed_jobs) > 0

    # Analyze downloaded logs.
    job_results = []
    for job, log_content in job_logs:
        result = analyze_job(job, log_content, args)
        job_results.append((job, result))

    # Output results showing GitHub status FIRST.
    output_results(all_jobs, job_results, log_files, args)

    # Exit code reflects ACTUAL job status from GitHub API.
    return exit_codes.ERROR if any_failures else exit_codes.SUCCESS


if __name__ == "__main__":
    args = parse_arguments()
    sys.exit(main(args))
