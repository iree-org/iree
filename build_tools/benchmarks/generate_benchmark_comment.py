#!/usr/bin/env python3
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates benchmark results as pull request comments.

This script is meant to be used by CI and uses pip package "markdown_strings".
"""

import sys
import pathlib

# Add build_tools python dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parent.with_name("python")))

import argparse
import collections
import dataclasses
import json
from typing import Any, Dict, Optional, Set, Tuple

import markdown_strings as md
import requests

from common import benchmark_definition, benchmark_presentation, common_arguments
from reporting import benchmark_comment

GITHUB_IREE_REPO_PREFIX = "https://github.com/openxla/iree"
IREE_DASHBOARD_URL = "https://perf.iree.dev/apis/v2"
IREE_PROJECT_ID = "IREE"
# The maximal numbers of trials when querying base commit benchmark results.
MAX_BASE_COMMIT_QUERY_COUNT = 10
# The max number of rows to show per table.
TABLE_SIZE_CUT = 3
THIS_DIRECTORY = pathlib.Path(__file__).resolve().parent


@dataclasses.dataclass(frozen=True)
class CommentDef(object):
    title: str
    type_id: str


# Map from comment type to comment definition.
COMMENT_DEF_MAP = {
    "android-benchmark-summary": CommentDef(
        title="Abbreviated Android Benchmark Summary",
        type_id="bf8cdf94-a992-466d-b11c-778cbd805a22",
    ),
    "linux-benchmark-summary": CommentDef(
        title="Abbreviated Linux Benchmark Summary",
        type_id="37549014-3c67-4e74-8d88-8e929231abe3",
    ),
    "benchmark-summary": CommentDef(
        title="Abbreviated Benchmark Summary",
        type_id="5b42cbfe-26a0-4164-a51c-07f06762e2dc",
    ),
}


def get_git_total_commit_count(commit: str, verbose: bool = False) -> int:
    """Gets the total commit count in history ending with the given commit."""
    # TODO(#11703): Should use --first-parent here. See issue for the required
    # work.
    count = benchmark_definition.execute_cmd_and_get_stdout(
        ["git", "rev-list", "--count", commit], cwd=THIS_DIRECTORY, verbose=verbose
    )
    return int(count)


def get_from_dashboard(
    url: str, payload: Dict[str, Any], verbose: bool = False
) -> Dict[str, Dict[str, Any]]:
    headers = {"Content-type": "application/json"}
    data = json.dumps(payload)

    if verbose:
        print(f"API request payload: {data}")

    response = requests.get(url, data=data, headers=headers)
    code = response.status_code
    if code != 200:
        raise requests.RequestException(
            f"Failed to get from dashboard server with status code {code}"
        )

    data = response.json()
    if verbose:
        print(f"Queried base benchmark data: {data}")
    return data


BenchmarkQueryResults = Dict[str, Dict[str, Any]]


def query_base_benchmark_results(
    commit: str, verbose: bool = False
) -> BenchmarkQueryResults:
    """Queries the benchmark results for the given commit."""
    build_id = get_git_total_commit_count(commit, verbose)
    payload = {"projectId": IREE_PROJECT_ID, "buildId": build_id}
    return get_from_dashboard(
        f"{IREE_DASHBOARD_URL}/getBuild", payload, verbose=verbose
    )


@dataclasses.dataclass(frozen=True)
class ComparableBenchmarkResults(object):
    commit_sha: str
    benchmark_results: BenchmarkQueryResults


def _find_comparable_benchmark_results(
    start_commit: str, required_benchmark_keys: Set[str], verbose: bool = False
) -> Optional[ComparableBenchmarkResults]:
    cmds = [
        "git",
        "rev-list",
        "--first-parent",
        f"--max-count={MAX_BASE_COMMIT_QUERY_COUNT}",
        start_commit,
    ]
    output = benchmark_definition.execute_cmd_and_get_stdout(
        cmds, cwd=THIS_DIRECTORY, verbose=verbose
    )
    previous_commits = output.splitlines()
    # Try to query some base benchmark to diff against, from the top of the
    # tree. Bail out if the maximal trial number is exceeded.
    for base_commit in previous_commits:
        base_benchmarks = query_base_benchmark_results(
            commit=base_commit, verbose=verbose
        )
        base_benchmark_keys = set(base_benchmarks.keys())
        if required_benchmark_keys <= base_benchmark_keys:
            return ComparableBenchmarkResults(
                commit_sha=base_commit, benchmark_results=base_benchmarks
            )

    return None


def _get_git_commit_hash(ref: str, verbose: bool = False) -> str:
    """Gets the commit hash for the given commit."""
    return benchmark_definition.execute_cmd_and_get_stdout(
        ["git", "rev-parse", ref], cwd=THIS_DIRECTORY, verbose=verbose
    )


def _get_git_merge_base_commit(
    pr_commit: str, target_branch: str, verbose: bool = False
) -> str:
    return benchmark_definition.execute_cmd_and_get_stdout(
        args=["git", "merge-base", target_branch, pr_commit],
        cwd=THIS_DIRECTORY,
        verbose=verbose,
    )


def _get_experimental_dt_comparison_markdown(
    execution_benchmarks: Dict[str, benchmark_presentation.AggregateBenchmarkLatency],
) -> Optional[str]:
    """Get the comparison table to compare different data-tiling options."""

    dt_tags = {"no-dt": "No-DT (baseline)", "dt-only": "DT-Only", "dt-uk": "DT-UK"}
    latency_map = collections.defaultdict(dict)
    for bench_id, latency in execution_benchmarks.items():
        dt_tag = next((tag for tag in dt_tags if tag in latency.name), None)
        if dt_tag is None:
            continue
        # See build_tools/python/e2e_test_framework/definitions/iree_definitions.py
        # for how benchmark name are constructed.
        # Format: model_name gen_tags exec_tags ...
        model, gen_tags, remaining = latency.name.split(" ", maxsplit=2)
        # Format: [compile targets][tags]
        compile_targets = gen_tags.split("][")[0] + "]"
        key_name = " ".join([model, compile_targets, remaining])
        latency_map[key_name][dt_tag] = (bench_id, latency.mean_time / 1e6)

    if len(latency_map) == 0:
        return None

    # Compute speedup vs. the baseline.
    table = {}
    for key_name, data in latency_map.items():
        baseline = data.get("no-dt")
        baseline = None if baseline is None else baseline[1]
        row = {}
        for dt_tag in dt_tags:
            pair = data.get(dt_tag)
            if pair is None:
                continue
            bench_id, mean_time = pair
            text = f"{mean_time:.03f}"
            if baseline is not None:
                text += f" ({(baseline / mean_time):.01f}X)"
            row[dt_tag] = (bench_id, text)
        table[key_name] = row

    table_columns = [["Name"] + list(table.keys())]
    for dt_tag, dt_name in dt_tags.items():
        column = [dt_name]
        for key_name, data in table.items():
            pair = data.get(dt_tag)
            if pair is None:
                column.append("N/A")
                continue
            bench_id, text = pair
            column.append(benchmark_presentation.make_series_link(text, bench_id))
        table_columns.append(column)

    return md.table(table_columns)


def _get_benchmark_result_markdown(
    execution_benchmarks: Dict[str, benchmark_presentation.AggregateBenchmarkLatency],
    compilation_metrics: Dict[str, benchmark_presentation.CompilationMetrics],
    pr_url: str,
    build_url: str,
    comment_def: CommentDef,
    commit_info_md: str,
) -> Tuple[str, str]:
    """Gets the full/abbreviated markdown summary of all benchmarks in files."""

    pr_info = md.link("Pull request", pr_url)
    build_info = md.link("Build", build_url)

    # Compose the full benchmark tables.
    full_table = [md.header("Full Benchmark Summary", 2)]
    full_table.append(md.unordered_list([commit_info_md, pr_info, build_info]))

    # Compose the abbreviated benchmark tables.
    abbr_table = [md.header(comment_def.title, 2)]
    abbr_table.append(commit_info_md)

    # The temporary table to help compare different data-tiling options.
    dt_cmp_table = _get_experimental_dt_comparison_markdown(
        execution_benchmarks=execution_benchmarks
    )
    if dt_cmp_table is not None:
        dt_cmp_header = md.header("Data-Tiling Comparison Table", 3)
        full_table += [dt_cmp_header, dt_cmp_table]
        abbr_table += [
            dt_cmp_header,
            "<details>",
            "<summary>Click to show</summary>",
            dt_cmp_table,
            "</details>",
        ]

    if len(execution_benchmarks) > 0:
        full_table.append(
            benchmark_presentation.categorize_benchmarks_into_tables(
                execution_benchmarks
            )
        )

        abbr_benchmarks_tables = (
            benchmark_presentation.categorize_benchmarks_into_tables(
                execution_benchmarks, TABLE_SIZE_CUT
            )
        )
        if len(abbr_benchmarks_tables) == 0:
            abbr_table.append("No improved or regressed benchmarks 🏖️")
        else:
            abbr_table.append(abbr_benchmarks_tables)

    # Compose the full compilation metrics tables.
    if len(compilation_metrics) > 0:
        full_table.append(
            benchmark_presentation.categorize_compilation_metrics_into_tables(
                compilation_metrics
            )
        )

        abbr_compilation_metrics_tables = (
            benchmark_presentation.categorize_compilation_metrics_into_tables(
                compilation_metrics, TABLE_SIZE_CUT
            )
        )
        if len(abbr_compilation_metrics_tables) == 0:
            abbr_table.append("No improved or regressed compilation metrics 🏖️")
        else:
            abbr_table.append(abbr_compilation_metrics_tables)

    abbr_table.append("For more information:")
    # We don't know until a Gist is really created. Use a placeholder for now and
    # replace later.
    full_result_info = md.link(
        "Full benchmark result tables", benchmark_comment.GIST_LINK_PLACEHORDER
    )
    abbr_table.append(md.unordered_list([full_result_info, build_info]))

    # Append the unique comment type id to help identify and update the existing
    # comment.
    abbr_table.append(f"<!--Comment type id: {comment_def.type_id}-->")

    return "\n\n".join(full_table), "\n\n".join(abbr_table)


def parse_arguments():
    """Parses command-line options."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark_files",
        metavar="<benchmark-json-files>",
        default=[],
        action="append",
        help=(
            "Paths to the JSON files containing benchmark results, " "accepts wildcards"
        ),
    )
    parser.add_argument(
        "--compile_stats_files",
        metavar="<compile-stats-json-files>",
        default=[],
        action="append",
        help=(
            "Paths to the JSON files containing compilation statistics, "
            "accepts wildcards"
        ),
    )
    parser.add_argument("--pr_number", required=True, type=int, help="PR number")
    parser.add_argument(
        "--pr_committish", type=str, default="HEAD", help="PR commit hash or ref"
    )
    parser.add_argument(
        "--pr_base_branch", type=str, default=None, help="Base branch to merge the PR."
    )
    parser.add_argument(
        "--comment_type",
        required=True,
        choices=COMMENT_DEF_MAP.keys(),
        help="Type of summary comment",
    )
    parser.add_argument(
        "--build_url",
        required=True,
        type=str,
        help="CI build page url to show in the report",
    )
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print internal information during execution",
    )

    return parser.parse_args()


def main(args):
    benchmark_files = common_arguments.expand_and_check_file_paths(args.benchmark_files)
    compile_stats_files = common_arguments.expand_and_check_file_paths(
        args.compile_stats_files
    )

    pr_commit = _get_git_commit_hash(ref=args.pr_committish, verbose=args.verbose)
    execution_benchmarks = benchmark_presentation.aggregate_all_benchmarks(
        benchmark_files=benchmark_files, expected_pr_commit=pr_commit
    )
    compilation_metrics = benchmark_presentation.collect_all_compilation_metrics(
        compile_stats_files=compile_stats_files, expected_pr_commit=pr_commit
    )

    if args.pr_base_branch is None:
        pr_base_commit = None
    else:
        pr_base_commit = _get_git_merge_base_commit(
            pr_commit=pr_commit, target_branch=args.pr_base_branch, verbose=args.verbose
        )

    if pr_base_commit is None:
        comparable_results = None
    else:
        required_benchmark_keys = set(execution_benchmarks.keys())
        for target_id in compilation_metrics:
            for mapper in benchmark_presentation.COMPILATION_METRICS_TO_TABLE_MAPPERS:
                required_benchmark_keys.add(mapper.get_series_id(target_id))

        comparable_results = _find_comparable_benchmark_results(
            start_commit=pr_base_commit,
            required_benchmark_keys=required_benchmark_keys,
            verbose=args.verbose,
        )

    if comparable_results is None:
        comparable_commit = None
    else:
        comparable_commit = comparable_results.commit_sha
        # Update the execution benchmarks with base numbers.
        for bench in execution_benchmarks:
            base_benchmark = comparable_results.benchmark_results[bench]
            if base_benchmark["sampleUnit"] != "ns":
                raise ValueError("Only support nanoseconds for latency sample.")
            execution_benchmarks[bench].base_mean_time = base_benchmark["sample"]

        # Update the compilation metrics with base numbers.
        for target_id, metrics in compilation_metrics.items():
            updated_metrics = metrics
            for mapper in benchmark_presentation.COMPILATION_METRICS_TO_TABLE_MAPPERS:
                base_benchmark = comparable_results.benchmark_results[
                    mapper.get_series_id(target_id)
                ]
                if base_benchmark["sampleUnit"] != mapper.get_unit():
                    raise ValueError("Unit of the queried sample is mismatched.")
                updated_metrics = mapper.update_base_value(
                    updated_metrics, base_benchmark["sample"]
                )
            compilation_metrics[target_id] = updated_metrics

    pr_commit_link = md.link(pr_commit, f"{GITHUB_IREE_REPO_PREFIX}/commit/{pr_commit}")
    commit_info_md = f"@ commit {pr_commit_link}"
    if comparable_commit is not None:
        baseline_commit_link = md.link(
            comparable_commit, f"{GITHUB_IREE_REPO_PREFIX}/commit/{comparable_commit}"
        )
        commit_info_md += f" (vs. base {baseline_commit_link})"
    elif pr_base_commit is not None:
        commit_info_md += " (no previous benchmark results to compare)"

    comment_def = COMMENT_DEF_MAP[args.comment_type]
    full_md, abbr_md = _get_benchmark_result_markdown(
        execution_benchmarks=execution_benchmarks,
        compilation_metrics=compilation_metrics,
        pr_url=f"{GITHUB_IREE_REPO_PREFIX}/pull/{args.pr_number}",
        build_url=args.build_url,
        comment_def=comment_def,
        commit_info_md=commit_info_md,
    )

    comment_data = benchmark_comment.CommentData(
        type_id=comment_def.type_id,
        abbr_md=abbr_md,
        full_md=full_md,
        unverified_pr_number=args.pr_number,
    )
    comment_json_data = json.dumps(dataclasses.asdict(comment_data), indent=2)
    if args.output is None:
        print(comment_json_data)
    else:
        args.output.write_text(comment_json_data)


if __name__ == "__main__":
    main(parse_arguments())
