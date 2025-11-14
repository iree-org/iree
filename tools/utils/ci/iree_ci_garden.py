# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""IREE CI Garden - CI Corpus Management Tool.

Manages a corpus of CI failures for pattern development and health tracking.
Fetch failures from GitHub Actions, classify them using iree-ci-triage, track
recognition rates, and generate TODO lists for unrecognized patterns.

Commands:
    fetch       - Fetch new CI failures from GitHub Actions
    classify    - Run classification on new/unclassified logs (incremental)
    reclassify  - Re-run extractors on entire corpus (after pattern changes)
    status      - Show corpus health metrics and recognition rate
    search      - Search corpus logs for error patterns
    garden      - Interactive gardening workflow (coming soon)

Examples:
    # Daily workflow: fetch overnight failures and classify
    iree-ci-garden fetch
    iree-ci-garden classify
    iree-ci-garden status

    # Fetch failures from specific PR
    iree-ci-garden fetch --pr 22625

    # Search for specific error pattern
    iree-ci-garden search "undefined reference"

    # REQUIRED after updating extractor patterns
    iree-ci-garden reclassify

See 'iree-ci-garden <command> --help' for more information on each command.
"""

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Import shared modules.
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import cli, console, exit_codes

from ci.core.classifier import Classifier
from ci.core.corpus import Corpus
from ci.core.fetcher import GitHubFetcher
from ci.core.gardener import Gardener
from ci.core.github_client import GitHubClient, check_gh_cli_setup


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="IREE CI corpus management tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Subcommands.
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Fetch command.
    fetch_parser = subparsers.add_parser(
        "fetch",
        help="Fetch CI failures from GitHub Actions",
        description="""
Fetch workflow run failures from GitHub Actions and download failed job logs.
By default, fetches failures from the last 24 hours from main and staging branches.
Failed job logs are stored in the corpus for later classification and analysis.
        """,
    )
    fetch_parser.add_argument(
        "--since",
        help="Fetch since date (YYYY-MM-DD). Default: 24 hours ago. "
        "Maximum: 90 days ago (GitHub log retention limit)",
    )
    fetch_parser.add_argument(
        "--branch",
        help="Filter by specific branch. Default: main and staging",
    )
    fetch_parser.add_argument(
        "--pr",
        type=int,
        help="Fetch failures from specific PR number",
    )
    fetch_parser.add_argument(
        "--run",
        help="Fetch specific workflow run ID",
    )
    fetch_parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of runs to fetch per branch (default: 1000). "
        "Increase for large time ranges (e.g., --limit 5000 for several months)",
    )
    fetch_parser.add_argument(
        "--repo",
        default="iree-org/iree",
        help="GitHub repository in owner/repo format (default: iree-org/iree)",
    )
    cli.add_common_output_flags(fetch_parser)

    # Classify command.
    classify_parser = subparsers.add_parser(
        "classify",
        help="Classify new/unclassified logs (incremental)",
        description="""
Run extractors on new or unclassified corpus logs. This is fast because it:
- Only processes logs not yet classified
- Uses cached results for previously classified logs

Use this for daily corpus updates after fetching new failures.

After classification, check ~/.iree-ci-corpus/unrecognized/TODO.md
for failures that need new extractor patterns.
        """,
    )
    classify_parser.add_argument(
        "--run",
        help="Classify only logs from specific run ID",
    )
    cli.add_common_output_flags(classify_parser)

    # Reclassify command.
    reclassify_parser = subparsers.add_parser(
        "reclassify",
        help="Re-run extractors on entire corpus (REQUIRED after pattern changes)",
        description="""
Re-run all extractors on the entire corpus, ignoring cache. This is REQUIRED after:
- Adding new patterns to extractors (e.g., infrastructure_flake.py)
- Modifying existing extractor logic
- Updating extractor activation keywords

This command is slower than 'classify' because it processes all logs, but ensures
the corpus reflects your latest extractor changes.

IMPORTANT: Always run this after modifying any extractor code or patterns.
        """,
    )
    reclassify_parser.add_argument(
        "--run",
        help="Reclassify only logs from specific run ID",
    )
    cli.add_common_output_flags(reclassify_parser)

    # Status command.
    status_parser = subparsers.add_parser(
        "status",
        help="Show corpus health metrics",
        description="""
Display corpus statistics including total runs, logs, recognition rate, and top
failure categories. Use --json for programmatic access to metrics.

The recognition rate shows what percentage of failures are identified by existing
patterns. A high rate (>90%) indicates good pattern coverage.
        """,
    )
    cli.add_common_output_flags(status_parser)

    # Search command.
    search_parser = subparsers.add_parser(
        "search",
        help="Search corpus logs for patterns",
        description="""
Search all corpus logs for a regex pattern. Useful for finding specific errors,
investigating common failure modes, or validating pattern coverage.

Results show matching log paths and line numbers for easy investigation.
        """,
    )
    search_parser.add_argument(
        "pattern",
        help="Regex pattern to search for (case-insensitive)",
    )
    search_parser.add_argument(
        "--category",
        help="Filter results by failure category",
    )
    search_parser.add_argument(
        "--since",
        help="Search only logs from date onwards (YYYY-MM-DD)",
    )
    cli.add_common_output_flags(search_parser)

    # Garden command (interactive).
    subparsers.add_parser("garden", help="Interactive gardening")

    # Global options.
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        help="Corpus directory (default: ~/.iree-ci-corpus)",
    )

    cli.add_common_output_flags(parser)

    return parser.parse_args()


def _get_validated_fetch_since_time(
    args: argparse.Namespace, corpus: Corpus
) -> datetime | None:
    """Determine and validate the 'since' datetime for fetching.

    Args:
        args: Parsed arguments with potential --since flag
        corpus: Corpus instance for looking up last fetch time

    Returns:
        Validated datetime (timezone-aware), or None if validation fails
    """
    if args.since:
        since = datetime.fromisoformat(args.since)
        # Ensure timezone-aware.
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)
    else:
        # Default: fetch since last fetch, or last 24 hours if corpus is empty.
        last_fetched = corpus.get_last_fetched_at()
        if last_fetched:
            since = last_fetched
            console.note(f"Using last fetch time: {since.isoformat()}", args=args)
        else:
            since = datetime.now(timezone.utc) - timedelta(days=1)
            console.note("Corpus is empty, fetching last 24 hours", args=args)

    # Enforce 90-day limit (GitHub log retention policy).
    now = datetime.now(timezone.utc)
    days_ago = (now - since).days
    if days_ago > 90:
        console.error(
            f"Cannot fetch runs older than 90 days (requested: {days_ago} days ago)",
            args=args,
        )
        console.error(
            "GitHub deletes workflow logs after 90 days. "
            f"Oldest fetchable date: {(now - timedelta(days=90)).strftime('%Y-%m-%d')}",
            args=args,
        )
        return None

    return since


def _display_fetch_results(
    result: Any, has_critical_errors: bool, corpus: Corpus, args: argparse.Namespace
) -> None:
    """Display fetch results summary and errors.

    Args:
        result: Fetch result object with stats
        has_critical_errors: Whether critical errors occurred
        corpus: Corpus instance for stats
        args: Parsed arguments
    """
    console.out("\nSummary:")
    console.out(f"  New runs added: {result.runs_new}")
    console.out(f"  Duplicate runs: {result.runs_duplicate}")
    console.out(f"  New logs downloaded: {result.logs_fetched}")
    if result.logs_failed > 0:
        console.warn(f"  Failed to fetch {result.logs_failed} logs", args=args)

    stats = corpus.get_stats()
    console.out(
        f"  Total corpus: {stats['total_runs']} runs, {stats['total_logs']} logs"
    )

    if result.errors:
        # Show errors as critical (error) or warnings based on severity.
        if has_critical_errors:
            console.out("\nCritical Errors:")
            for error in result.errors[:10]:  # Limit to first 10.
                console.error(f"  {error}", args=args)
            if result.runs_new > 0:
                console.warn("  Partial fetch completed before errors", args=args)
        else:
            console.out("\nWarnings:")
            for error in result.errors[:10]:
                console.warn(f"  {error}", args=args)


def cmd_fetch(args: argparse.Namespace, corpus: Corpus) -> int:
    """Execute fetch command.

    Args:
        args: Parsed arguments
        corpus: Corpus instance

    Returns:
        Exit code
    """
    # Check gh CLI setup.
    success, error_msg = check_gh_cli_setup()
    if not success:
        console.error(error_msg, args=args)
        return exit_codes.SETUP_ERROR

    # Initialize GitHub client and fetcher.
    client = GitHubClient(repo=args.repo)
    fetcher = GitHubFetcher(client, corpus, args=args)

    # Determine fetch strategy.
    if args.run:
        console.note(f"Fetching run {args.run}...", args=args)
        result = fetcher.fetch_run(args.run)
    elif args.pr:
        console.note(f"Fetching PR #{args.pr}...", args=args)
        result = fetcher.fetch_pr(args.pr)
    else:
        # Fetch since last run or specified date.
        since = _get_validated_fetch_since_time(args, corpus)
        if since is None:
            return exit_codes.ERROR

        console.note(f"Fetching failures since {since.isoformat()}...", args=args)
        result = fetcher.fetch_since(
            since=since, limit=args.limit, branch=args.branch, status="failure"
        )

    # Check for critical errors (timeouts, gh CLI failures).
    has_critical_errors = any(
        "timed out" in err.lower() or "failed to fetch" in err.lower()
        for err in result.errors
    )

    # Update last fetch time in config (even if all duplicates).
    config = json.loads(corpus.config_path.read_text())
    config["last_fetch_at"] = datetime.now(timezone.utc).isoformat()
    corpus.config_path.write_text(json.dumps(config, indent=2))

    # Display results.
    if not args.quiet:
        _display_fetch_results(result, has_critical_errors, corpus, args)

    # Return error code if critical errors occurred.
    if has_critical_errors:
        return exit_codes.ERROR

    return exit_codes.SUCCESS


def cmd_classify(args: argparse.Namespace, corpus: Corpus) -> int:
    """Execute classify command (incremental classification).

    Args:
        args: Parsed arguments
        corpus: Corpus instance

    Returns:
        Exit code
    """
    classifier = Classifier(corpus)

    if args.run:
        console.note(f"Classifying run {args.run}...", args=args)
        results = classifier.classify_run(args.run, force_reclassify=False)
        recognized = sum(1 for r in results if r.recognized)

        if not args.quiet:
            console.out(f"\nClassified {len(results)} logs: {recognized} recognized")
    else:
        console.note("Classifying unclassified logs...", args=args)
        report = classifier.classify_all_unclassified()

        # Display report.
        if not args.quiet:
            console.out("\nResults:")
            console.out(f"  Classified: {report.classified} logs")
            console.out(
                f"  Recognized: {report.recognized}/{report.total_logs} "
                f"({report.recognition_rate:.1%})"
            )

            if report.categories:
                console.out("\n  Categories found:")
                for category, count in sorted(
                    report.categories.items(), key=lambda x: x[1], reverse=True
                )[:10]:
                    console.out(f"    {category}: {count}")

            if report.unrecognized > 0:
                console.out(f"\n  Unrecognized: {report.unrecognized} logs")
                console.out(f"  See {corpus.unrecognized_dir}/TODO.md for details")

        # Generate TODO list.
        gardener = Gardener(corpus, classifier)
        gardener.save_todo_list()

    return exit_codes.SUCCESS


def cmd_reclassify(args: argparse.Namespace, corpus: Corpus) -> int:
    """Execute reclassify command (full corpus reclassification).

    Args:
        args: Parsed arguments
        corpus: Corpus instance

    Returns:
        Exit code
    """
    classifier = Classifier(corpus)

    if args.run:
        console.note(f"Reclassifying run {args.run}...", args=args)
        results = classifier.classify_run(args.run, force_reclassify=True)
        recognized = sum(1 for r in results if r.recognized)

        if not args.quiet:
            console.out(f"\nReclassified {len(results)} logs: {recognized} recognized")
    else:
        console.note("Reclassifying entire corpus...", args=args)
        report = classifier.reclassify_all()

        # Display report.
        if not args.quiet:
            console.out("\nResults:")
            console.out(f"  Reclassified: {report.classified} logs")
            console.out(
                f"  Recognized: {report.recognized}/{report.total_logs} "
                f"({report.recognition_rate:.1%})"
            )

            if report.categories:
                console.out("\n  Categories found:")
                for category, count in sorted(
                    report.categories.items(), key=lambda x: x[1], reverse=True
                )[:10]:
                    console.out(f"    {category}: {count}")

            if report.unrecognized > 0:
                console.out(f"\n  Unrecognized: {report.unrecognized} logs")
                console.out(f"  See {corpus.unrecognized_dir}/TODO.md for details")

        # Generate TODO list.
        gardener = Gardener(corpus, classifier)
        gardener.save_todo_list()

    return exit_codes.SUCCESS


def cmd_status(args: argparse.Namespace, corpus: Corpus) -> int:
    """Execute status command.

    Args:
        args: Parsed arguments
        corpus: Corpus instance

    Returns:
        Exit code
    """
    classifier = Classifier(corpus)
    gardener = Gardener(corpus, classifier)
    health = gardener.get_corpus_health()

    if args.json:
        # JSON output.
        output = {
            "corpus_dir": str(corpus.corpus_dir),
            "last_updated": datetime.now().isoformat(),
            "statistics": health["stats"],
            "recognition": {
                "overall_rate": health["recognition_rate"],
                "categories": health["categories"],
            },
        }
        console.out(json.dumps(output, indent=2))
    else:
        # Human-readable output.
        stats = health["stats"]
        console.out("IREE CI Corpus Status")
        console.out("=" * 50)
        console.out(f"Location: {corpus.corpus_dir}")
        console.out(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        console.out("Corpus Statistics:")
        console.out(f"  Total runs: {stats['total_runs']}")
        console.out(f"  Total logs: {stats['total_logs']}")
        size_mb = stats["size_bytes"] / (1024 * 1024)
        console.out(f"  Total size: {size_mb:.1f} MB\n")

        if stats["time_span"]["start"]:
            console.out("  Time span:")
            console.out(f"    Start: {stats['time_span']['start']}")
            console.out(f"    End: {stats['time_span']['end']}\n")

        console.out("Recognition Health:")
        rate = health["recognition_rate"]
        console.out(f"  Overall rate: {rate:.1%}")

        if health["categories"]:
            console.out("\nTop Failure Categories:")
            for i, (category, count) in enumerate(
                sorted(health["categories"].items(), key=lambda x: x[1], reverse=True)[
                    :5
                ],
                1,
            ):
                pct = (
                    count / stats["total_logs"] * 100 if stats["total_logs"] > 0 else 0
                )
                console.out(f"  {i}. {category}: {count} ({pct:.1f}%)")

    return exit_codes.SUCCESS


def cmd_search(args: argparse.Namespace, corpus: Corpus) -> int:
    """Execute search command.

    Args:
        args: Parsed arguments
        corpus: Corpus instance

    Returns:
        Exit code
    """
    console.note(f"Searching corpus for '{args.pattern}'...", args=args)

    pattern = re.compile(args.pattern, re.IGNORECASE)
    matches = []

    # Search all logs.
    for run_dir in corpus.logs_dir.iterdir():
        if not run_dir.is_dir():
            continue

        for log_file in run_dir.glob("*.log"):
            try:
                with open(log_file) as f:
                    for line_num, line in enumerate(f, 1):
                        if pattern.search(line):
                            matches.append((log_file, line_num, line.strip()))
                            if len(matches) >= 100:  # Limit to first 100 matches.
                                break
            except (OSError, UnicodeDecodeError):
                continue

            if len(matches) >= 100:
                break

    # Display matches.
    if not args.quiet:
        console.out(f"\nFound {len(matches)} matches:\n")
        for log_path, line_num, line in matches[:50]:  # Show first 50.
            console.out(f"{log_path.relative_to(corpus.corpus_dir)}:{line_num}: {line}")

        if len(matches) > 50:
            console.out(f"\n... and {len(matches) - 50} more matches")

    return exit_codes.SUCCESS


def cmd_garden(args: argparse.Namespace, corpus: Corpus) -> int:
    """Execute garden command (interactive).

    Args:
        args: Parsed arguments
        corpus: Corpus instance

    Returns:
        Exit code
    """
    console.out("Interactive gardening mode coming soon!")
    console.out("For now, use: iree-ci-garden status && iree-ci-garden classify")
    return exit_codes.SUCCESS


def main() -> int:
    """Main entry point."""
    args = parse_arguments()

    # Ensure a command was specified.
    if not args.command:
        console.error("No command specified. Use --help for usage.")
        return exit_codes.ERROR

    # Initialize corpus.
    corpus_dir = args.corpus_dir or Path.home() / ".iree-ci-corpus"
    corpus = Corpus(corpus_dir)

    # Dispatch to command.
    if args.command == "fetch":
        return cmd_fetch(args, corpus)
    if args.command == "classify":
        return cmd_classify(args, corpus)
    if args.command == "reclassify":
        return cmd_reclassify(args, corpus)
    if args.command == "status":
        return cmd_status(args, corpus)
    if args.command == "search":
        return cmd_search(args, corpus)
    if args.command == "garden":
        return cmd_garden(args, corpus)
    console.error(f"Unknown command: {args.command}")
    return exit_codes.ERROR


if __name__ == "__main__":
    sys.exit(main())
