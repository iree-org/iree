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
- Multi-workflow support: PR/commit mode fetches ALL workflows (CI, Lint, PkgCI, etc.)

Usage:
    iree-ci-triage --run 12345678                  # Triage all failed jobs in a run
    iree-ci-triage --pr 12345                      # Triage ALL workflows for PR (CI, Lint, PkgCI, etc.)
    iree-ci-triage --commit abc123def              # Triage ALL workflows for commit
    iree-ci-triage --branch main                   # Triage latest workflow run from branch
    iree-ci-triage --run 12345678 --job 987654321  # Triage specific job
    iree-ci-triage --pr 12345 --json               # JSON output with workflow grouping

Examples:
    # Human-readable triage of all workflows in a PR (matches GitHub UI)
    iree-ci-triage --pr 12345

    # Triage all workflows for a specific commit
    iree-ci-triage --commit abc123def

    # Find successful runs for comparison
    iree-ci-triage --pr 12345 --status success

    # JSON output for piping to other tools (new schema with workflow grouping)
    iree-ci-triage --pr 12345 --json | jq '.runs[].workflow_name'
    iree-ci-triage --pr 12345 --json | jq '.runs[] | select(.workflow_name == "CI") | .jobs[].extracted_issues'
"""

import argparse
import sys
from pathlib import Path

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
        help="Pull request number (fetches ALL workflow runs for PR's HEAD commit)",
    )
    source_group.add_argument(
        "--commit",
        metavar="SHA",
        help="Commit SHA (fetches ALL workflow runs for this commit)",
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


def resolve_run_ids(
    client: github_client.GitHubClient, args: argparse.Namespace
) -> list[github_client.WorkflowRun]:
    """Resolve --pr/--commit/--branch/--job/--run to workflow run(s).

    For --pr and --commit: Returns ALL workflow runs for the commit (matching GitHub UI).
    For --branch and --run: Returns single workflow run.

    Args:
        client: GitHubClient instance
        args: Command line arguments

    Returns:
        List of WorkflowRun objects (may be single element for --run/--branch)

    Raises:
        SystemExit: If resolution fails
    """
    # If explicit run ID provided, fetch its metadata.
    if args.run:
        try:
            console.note(f"Fetching workflow run {args.run}...", args=args)
            run_data = client.get_run(args.run)
            if not run_data:
                console.error(f"Run {args.run} not found", args=args)
                sys.exit(exit_codes.NOT_FOUND)

            run = github_client.WorkflowRun(
                run_id=args.run,
                workflow_name=run_data.get("workflowName", ""),
                conclusion=run_data.get("conclusion", ""),
                created_at=run_data.get("createdAt", ""),
                head_branch=run_data.get("headBranch", ""),
                display_title=run_data.get("displayTitle", ""),
            )
            console.note(
                f"Using run {run.run_id}: {run.workflow_name} ({run.conclusion})",
                args=args,
            )
            return [run]

        except github_client.GitHubClientError as e:
            console.error(f"Failed to fetch run {args.run}: {e}", args=args)
            sys.exit(exit_codes.ERROR)

    # If standalone --job provided, fetch run from job metadata.
    if args.job and not (args.pr or args.commit or args.branch):
        try:
            console.note(f"Fetching run ID for job {args.job}...", args=args)
            job_metadata = client.get_job_metadata(args.job)
            run_id = str(job_metadata["run_id"])

            console.note(
                f"Job {args.job}: {job_metadata['name']} (run {run_id})", args=args
            )

            # Fetch full run metadata.
            run_data = client.get_run(run_id)
            if not run_data:
                console.error(f"Run {run_id} not found for job {args.job}", args=args)
                sys.exit(exit_codes.NOT_FOUND)

            run = github_client.WorkflowRun(
                run_id=run_id,
                workflow_name=run_data.get("workflowName", ""),
                conclusion=run_data.get("conclusion", ""),
                created_at=run_data.get("createdAt", ""),
                head_branch=run_data.get("headBranch", ""),
                display_title=run_data.get("displayTitle", ""),
            )
            return [run]

        except github_client.GitHubClientError as e:
            console.error(f"Failed to fetch job metadata: {e}", args=args)
            sys.exit(exit_codes.ERROR)

    # Otherwise, resolve PR/commit/branch to workflow run(s).
    try:
        if args.pr:
            console.note(
                f"Fetching all {args.status} workflow runs for PR #{args.pr}...",
                args=args,
            )
            runs = client.get_runs_for_pr(args.pr, status=args.status, limit=None)

        elif args.commit:
            console.note(
                f"Fetching all {args.status} workflow runs for commit {args.commit}...",
                args=args,
            )
            runs = client.get_runs_for_commit(
                args.commit, status=args.status, limit=None
            )

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

        # Log summary of what we found.
        if len(runs) == 1:
            run = runs[0]
            console.note(
                f"Found 1 run: {run.workflow_name} ({run.conclusion})",
                args=args,
            )
        else:
            workflow_names = [r.workflow_name for r in runs]
            console.note(
                f"Found {len(runs)} workflow runs: {', '.join(workflow_names)}",
                args=args,
            )

        return runs

    except github_client.GitHubClientError as e:
        console.error(f"Failed to resolve runs: {e}", args=args)
        sys.exit(exit_codes.ERROR)


def fetch_all_job_data(
    client: github_client.GitHubClient,
    workflow_runs: list[github_client.WorkflowRun],
    job_id: str | None,
    args: argparse.Namespace,
) -> list[dict]:
    """Fetch jobs and logs for multiple workflow runs.

    Args:
        client: GitHubClient instance
        workflow_runs: List of workflow runs to fetch jobs for
        job_id: Optional specific job ID (only used when single run)
        args: Command line arguments

    Returns:
        List of dicts with structure:
        {
            "workflow_run": WorkflowRun,
            "all_jobs": List[Job],
            "job_logs": List[tuple[Job, str]]
        }

    Raises:
        SystemExit: If fetch fails
    """
    workflow_data = []

    for run_idx, run in enumerate(workflow_runs, 1):
        run_id = run.run_id

        try:
            # Get ALL jobs to check actual GitHub API status.
            if len(workflow_runs) > 1:
                console.note(
                    f"[{run_idx}/{len(workflow_runs)}] Fetching jobs for {run.workflow_name} (run {run_id})...",
                    args=args,
                )
            else:
                console.note(f"Fetching jobs for run {run_id}...", args=args)

            all_jobs = client.get_jobs(run_id)

            if not all_jobs:
                console.note(
                    f"  No jobs found in {run.workflow_name}.",
                    args=args,
                )
                workflow_data.append(
                    {
                        "workflow_run": run,
                        "all_jobs": [],
                        "job_logs": [],
                    }
                )
                continue

            if job_id:
                # Specific job requested (only valid for single run).
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

                workflow_data.append(
                    {
                        "workflow_run": run,
                        "all_jobs": all_jobs,
                        "job_logs": [(target_job, log_content)],
                    }
                )
                continue

            # Determine which jobs to download logs for.
            if args.download_all_logs:
                jobs_to_download = all_jobs
                console.note(
                    f"  Downloading logs for all {len(all_jobs)} job(s)...",
                    args=args,
                )
            else:
                # Only download failed jobs by default (source of truth: GitHub API).
                jobs_to_download = [j for j in all_jobs if j.conclusion == "failure"]
                failed_count = len(jobs_to_download)
                console.note(
                    f"  Found {failed_count} failed job(s) in {run.workflow_name}",
                    args=args,
                )

                if failed_count == 0:
                    console.note(
                        f"  No failed jobs to analyze in {run.workflow_name}.",
                        args=args,
                    )
                    workflow_data.append(
                        {
                            "workflow_run": run,
                            "all_jobs": all_jobs,
                            "job_logs": [],
                        }
                    )
                    continue

            # Download logs.
            job_logs = []
            for i, job in enumerate(jobs_to_download, 1):
                console.note(
                    f"    [{i}/{len(jobs_to_download)}] Fetching log for: {job.name}",
                    args=args,
                )
                log_content = client.get_job_log(run_id, job.job_id)
                if log_content:
                    job_logs.append((job, log_content))
                else:
                    console.warn(
                        f"    Log not available for job {job.job_id}",
                        args=args,
                    )

            workflow_data.append(
                {
                    "workflow_run": run,
                    "all_jobs": all_jobs,
                    "job_logs": job_logs,
                }
            )

        except github_client.GitHubClientError as e:
            console.error(
                f"GitHub API error for {run.workflow_name}: {e}",
                args=args,
            )
            sys.exit(exit_codes.ERROR)

    return workflow_data


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
    workflow_data: list[dict],
    args: argparse.Namespace,
) -> dict[str, dict[str, Path]]:
    """Save job logs to /tmp organized by workflow and return nested path mapping.

    Args:
        workflow_data: List of workflow data dicts (from fetch_all_job_data)
        args: Command line arguments

    Returns:
        Nested dictionary: {run_id: {job_id: log_path}}
    """
    all_log_files = {}
    total_saved = 0

    for data in workflow_data:
        run = data["workflow_run"]
        job_logs = data["job_logs"]

        if not job_logs:
            continue

        # Create workflow-specific subdirectory.
        # Sanitize workflow name for directory.
        safe_workflow_name = (
            run.workflow_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        )
        output_dir = Path(f"/tmp/iree-ci-triage/run_{run.run_id}/{safe_workflow_name}")
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
            total_saved += 1

        all_log_files[run.run_id] = log_files

        if log_files and not args.quiet:
            console.note(
                f"  Saved {len(log_files)} log(s) for {run.workflow_name} to: {output_dir}",
                args=args,
            )

    if total_saved > 0 and not args.quiet:
        base_dir = Path("/tmp/iree-ci-triage")
        console.note(f"Total: {total_saved} log(s) saved to {base_dir}", args=args)

    return all_log_files


def output_results(
    workflow_data: list[dict],
    workflow_results: list[dict],
    log_files: dict[str, dict[str, Path]],
    args: argparse.Namespace,
) -> None:
    """Output triage results with GitHub status as source of truth.

    Args:
        workflow_data: List of workflow data dicts (from fetch_all_job_data)
        workflow_results: List of workflow results with analyzed jobs
        log_files: Nested dict {run_id: {job_id: Path}}
        args: Command line arguments
    """
    # Count total jobs across all workflows.
    total_failed = 0
    total_passed = 0
    total_other = 0
    total_jobs = 0

    for data in workflow_data:
        all_jobs = data["all_jobs"]
        total_jobs += len(all_jobs)
        total_failed += len([j for j in all_jobs if j.conclusion == "failure"])
        total_passed += len([j for j in all_jobs if j.conclusion == "success"])
        total_other += len(
            [j for j in all_jobs if j.conclusion not in ("failure", "success")]
        )

    if args.json:
        # JSON output for automation (new multi-workflow schema).
        runs = []
        for data, results in zip(workflow_data, workflow_results, strict=False):
            run = data["workflow_run"]
            all_jobs = data["all_jobs"]
            job_results = results["job_results"]
            run_log_files = log_files.get(run.run_id, {})

            runs.append(
                {
                    "run_id": run.run_id,
                    "workflow_name": run.workflow_name,
                    "conclusion": run.conclusion,
                    "created_at": run.created_at,
                    "jobs": [
                        {
                            "job_id": job.job_id,
                            "name": job.name,
                            "conclusion": job.conclusion,  # GitHub API status!
                            "log_path": str(run_log_files.get(job.job_id, "")),
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
                                        (
                                            r
                                            for j, r in job_results
                                            if j.job_id == job.job_id
                                        ),
                                        None,
                                    )
                                )
                                else []
                            ),
                            "extraction_status": (
                                "success"
                                if (
                                    result := next(
                                        (
                                            r
                                            for j, r in job_results
                                            if j.job_id == job.job_id
                                        ),
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
                }
            )

        output = {
            "runs": runs,
            "summary": {
                "total_workflows": len(workflow_data),
                "total_jobs": total_jobs,
                "failed": total_failed,
                "passed": total_passed,
                "other": total_other,
                "logs_downloaded": sum(len(files) for files in log_files.values()),
            },
        }
        console.print_json(output, args=args)
        return

    # Human-readable output.
    if total_failed == 0:
        console.out("✓ All jobs passed across all workflows")
        console.out(f"  Total workflows: {len(workflow_data)}")
        console.out(f"  Total jobs: {total_jobs}")
        if log_files:
            console.out("  Logs: /tmp/iree-ci-triage/run_*")
        return

    # Show failed jobs grouped by workflow.
    console.out(f"\n{'=' * 80}")
    console.out(
        f"FAILED JOBS ({total_failed}/{total_jobs} across {len(workflow_data)} workflow(s)):"
    )
    console.out(f"{'=' * 80}\n")

    for data, results in zip(workflow_data, workflow_results, strict=False):
        run = data["workflow_run"]
        all_jobs = data["all_jobs"]
        job_results = results["job_results"]
        run_log_files = log_files.get(run.run_id, {})

        failed_jobs = [j for j in all_jobs if j.conclusion == "failure"]

        if not failed_jobs:
            continue  # Skip workflows with no failures.

        console.out(f"─── {run.workflow_name} (run {run.run_id}) ───")
        console.out(f"Failed: {len(failed_jobs)}/{len(all_jobs)} jobs\n")

        for job in failed_jobs:
            console.out(f"  ✗ Job {job.job_id}: {job.name}")
            console.out("    Status: failure (from GitHub API)")

            # Show log file path.
            if job.job_id in run_log_files:
                console.out(f"    Log: {run_log_files[job.job_id]}")

            # Show extracted issues if available.
            result = next((r for j, r in job_results if j.job_id == job.job_id), None)
            if result and result.issues:
                console.out(f"\n    Extracted Issues ({len(result.issues)}):")
                for issue in result.issues[:5]:  # Limit to first 5.
                    console.out(f"      • {issue.message}")
                if len(result.issues) > 5:
                    console.out(f"      ... and {len(result.issues) - 5} more")
            elif result:
                console.out(
                    "\n    ⚠ WARNING: No issues extracted - manual review needed"
                )
                if job.job_id in run_log_files:
                    console.out(f"    See full log: {run_log_files[job.job_id]}")
            else:
                console.out("\n    (Log not analyzed)")

            console.out("")

    console.out(f"{'=' * 80}")
    console.out(
        f"Summary: {total_failed} failed, {total_passed} passed, {total_other} other across {len(workflow_data)} workflow(s)"
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

        # Create synthetic job and workflow for local testing.
        job = github_client.Job(
            job_id=job_name,
            name=job_name,
            conclusion="failure",
            runner_name=None,
            started_at="",
            completed_at="",
        )
        run = github_client.WorkflowRun(
            run_id="local",
            workflow_name="Local Test",
            conclusion="failure",
            created_at="",
            head_branch="",
            display_title="",
        )

        # Create workflow_data structure.
        workflow_data = [
            {
                "workflow_run": run,
                "all_jobs": [job],
                "job_logs": [(job, log_content)],
            }
        ]

        # Analyze the job.
        result = analyze_job(job, log_content, args)
        workflow_results = [{"job_results": [(job, result)]}]

        # Create log files structure.
        log_files = {"local": {job.job_id: args.log_file}}
        any_failures = True

    else:
        # Check dependencies for GitHub mode.
        status = check_dependencies(args)
        if status != exit_codes.SUCCESS:
            return status

        # Create GitHub client.
        client = github_client.GitHubClient(repo=args.repo)

        # Resolve workflow runs from --pr/--commit/--branch/--run/--job.
        workflow_runs = resolve_run_ids(client, args)

        # Fetch ALL jobs and logs for failed/requested jobs across all workflows.
        workflow_data = fetch_all_job_data(client, workflow_runs, args.job, args)

        # Always save logs to /tmp.
        log_files = save_logs_to_tmp(workflow_data, args)

        # Also save to custom directory if requested.
        if args.save_logs:
            args.save_logs.mkdir(parents=True, exist_ok=True)
            console.note(f"Saving logs to: {args.save_logs}", args=args)

            for data in workflow_data:
                run = data["workflow_run"]
                job_logs = data["job_logs"]

                for job, log_content in job_logs:
                    # Strip log prefixes (GitHub Actions, etc.) to save tokens.
                    log_buffer = LogBuffer(log_content, auto_detect_format=True)
                    stripped_content = log_buffer.content

                    # Sanitize job name for filename.
                    safe_name = job.name.replace(" / ", "_").replace(" ", "_")
                    log_file = (
                        args.save_logs / f"{run.run_id}_{job.job_id}_{safe_name}.log"
                    )
                    log_file.write_text(stripped_content)
                    console.note(f"  Saved: {log_file.name}", args=args)

        # Analyze downloaded logs for each workflow.
        workflow_results = []
        for data in workflow_data:
            job_logs = data["job_logs"]

            job_results = []
            for job, log_content in job_logs:
                result = analyze_job(job, log_content, args)
                job_results.append((job, result))

            workflow_results.append({"job_results": job_results})

        # Determine exit code from GitHub API (source of truth).
        any_failures = False
        for data in workflow_data:
            all_jobs = data["all_jobs"]
            failed_jobs = [j for j in all_jobs if j.conclusion == "failure"]
            if failed_jobs:
                any_failures = True
                break

    # Output results showing GitHub status FIRST.
    output_results(workflow_data, workflow_results, log_files, args)

    # Exit code reflects ACTUAL job status from GitHub API.
    return exit_codes.ERROR if any_failures else exit_codes.SUCCESS


if __name__ == "__main__":
    args = parse_arguments()
    sys.exit(main(args))
