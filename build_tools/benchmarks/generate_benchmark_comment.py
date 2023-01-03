#!/usr/bin/env python3
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates benchmark results as pull request comments.

This script is meant to be used by CI. It requires the following environment
variables to be set:

- IREE_DASHBOARD_URL: the url to IREE's performance dashboard.

This script uses pip package "markdown_strings".
"""

import sys
import pathlib

# Add build_tools python dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parent.with_name("python")))

from typing import Any, Dict, Optional, Set, Tuple
import argparse
import dataclasses
import json
import markdown_strings as md
import os
import requests

from common import benchmark_definition, benchmark_presentation, common_arguments
from reporting import benchmark_comment

GITHUB_IREE_REPO_PREFIX = "https://github.com/iree-org/iree"
IREE_PROJECT_ID = 'IREE'
# The maximal numbers of trials when querying base commit benchmark results.
MAX_BASE_COMMIT_QUERY_COUNT = 10
# The max number of rows to show per table.
TABLE_SIZE_CUT = 3
THIS_DIRECTORY = pathlib.Path(__file__).resolve().parent


def get_required_env_var(var: str) -> str:
  """Gets the value for a required environment variable."""
  value = os.getenv(var, None)
  if value is None:
    raise RuntimeError(f'Missing environment variable "{var}"')
  return value


def get_git_total_commit_count(commit: str, verbose: bool = False) -> int:
  """Gets the total commit count in history ending with the given commit."""
  count = benchmark_definition.execute_cmd_and_get_output(
      ['git', 'rev-list', '--count', commit],
      cwd=THIS_DIRECTORY,
      verbose=verbose)
  return int(count)


def get_from_dashboard(url: str,
                       payload: Dict[str, Any],
                       verbose: bool = False) -> Dict[str, Dict[str, Any]]:
  headers = {'Content-type': 'application/json'}
  data = json.dumps(payload)

  if verbose:
    print(f'API request payload: {data}')

  response = requests.get(url, data=data, headers=headers)
  code = response.status_code
  if code != 200:
    raise requests.RequestException(
        f'Failed to get from dashboard server with status code {code}')

  data = response.json()
  if verbose:
    print(f'Queried base benchmark data: {data}')
  return data


def query_base_benchmark_results(
    commit: str,
    dashboard_api_url: str,
    verbose: bool = False) -> Dict[str, Dict[str, Any]]:
  """Queries the benchmark results for the given commit."""
  build_id = get_git_total_commit_count(commit, verbose)
  payload = {'projectId': IREE_PROJECT_ID, 'buildId': build_id}
  return get_from_dashboard(f'{dashboard_api_url}/getBuild',
                            payload,
                            verbose=verbose)


def _find_comparable_benchmark_results(
    start_commit: str,
    required_benchmark_keys: Set[str],
    dashboard_api_url: str,
    verbose: bool = False) -> Optional[Tuple[str, Dict[str, Dict[str, Any]]]]:
  cmds = [
      "git", "rev-list", f"--max-count={MAX_BASE_COMMIT_QUERY_COUNT}",
      start_commit
  ]
  output = benchmark_definition.execute_cmd_and_get_output(cmds,
                                                           cwd=THIS_DIRECTORY,
                                                           verbose=verbose)
  previous_commits = output.splitlines()
  # Try to query some base benchmark to diff against, from the top of the
  # tree. Bail out if the maximal trial number is exceeded.
  for base_commit in previous_commits:
    base_benchmarks = query_base_benchmark_results(
        commit=base_commit,
        dashboard_api_url=dashboard_api_url,
        verbose=verbose)
    base_benchmark_keys = set(base_benchmarks.keys())
    if required_benchmark_keys <= base_benchmark_keys:
      return base_commit, base_benchmarks

  return None


def _get_benchmark_result_markdown(
    execution_benchmarks: Dict[
        str, benchmark_presentation.AggregateBenchmarkLatency],
    compilation_metrics: Dict[str, benchmark_presentation.CompilationMetrics],
    pr_url: str, build_url: str, comment_title: str, comment_type_id: str,
    commit_info_md: str) -> Tuple[str, str]:
  """Gets the full/abbreviated markdown summary of all benchmarks in files."""

  pr_info = md.link("Pull request", pr_url)
  build_info = md.link("Build", build_url)

  # Compose the full benchmark tables.
  full_table = [md.header("Full Benchmark Summary", 2)]
  full_table.append(md.unordered_list([commit_info_md, pr_info, build_info]))
  full_table.append(
      benchmark_presentation.categorize_benchmarks_into_tables(
          execution_benchmarks))

  # Compose the full compilation metrics tables.
  full_table.append(
      benchmark_presentation.categorize_compilation_metrics_into_tables(
          compilation_metrics))

  # Compose the abbreviated benchmark tables.
  abbr_table = [md.header(comment_title, 2)]
  abbr_table.append(commit_info_md)

  abbr_benchmarks_tables = benchmark_presentation.categorize_benchmarks_into_tables(
      execution_benchmarks, TABLE_SIZE_CUT)
  if len(abbr_benchmarks_tables) == 0:
    abbr_table.append("No improved or regressed benchmarks 🏖️")
  else:
    abbr_table.append(abbr_benchmarks_tables)

  abbr_compilation_metrics_tables = benchmark_presentation.categorize_compilation_metrics_into_tables(
      compilation_metrics, TABLE_SIZE_CUT)
  if len(abbr_compilation_metrics_tables) == 0:
    abbr_table.append("No improved or regressed compilation metrics 🏖️")
  else:
    abbr_table.append(abbr_compilation_metrics_tables)

  abbr_table.append("For more information:")
  # We don't know until a Gist is really created. Use a placeholder for now and
  # replace later.
  full_result_info = md.link("Full benchmark result tables",
                             benchmark_comment.GIST_LINK_PLACEHORDER)
  abbr_table.append(md.unordered_list([full_result_info, build_info]))

  # Append the unique comment type id to help identify and update the existing
  # comment.
  abbr_table.append(f"<!--Comment type id: {comment_type_id}-->")

  return "\n\n".join(full_table), "\n\n".join(abbr_table)


def parse_arguments():
  """Parses command-line options."""

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--benchmark_files",
      metavar="<benchmark-json-files>",
      default=[],
      nargs="+",
      help=("Paths to the JSON files containing benchmark results, "
            "accepts wildcards"))
  parser.add_argument(
      "--compile_stats_files",
      metavar="<compile-stats-json-files>",
      default=[],
      nargs="+",
      help=("Paths to the JSON files containing compilation statistics, "
            "accepts wildcards"))
  parser.add_argument("--pr_number", required=True, type=int, help="PR number")
  parser.add_argument("--pr_commit",
                      required=True,
                      type=str,
                      help="PR commit hash")
  parser.add_argument(
      "--pr_base_commit",
      type=str,
      default=None,
      help="Start commit to find the benchmark results to compare in reverse. "
      "Requires repo to have the full ancestor history of the start commit")
  parser.add_argument("--comment_title",
                      default="Abbreviated Benchmark Summary",
                      help="Title of the comment")
  parser.add_argument("--comment_type_id",
                      default="f6919a4c-7bb3-4fd6-af89-97980ce49f95",
                      help="Unique id to identify previous comment")
  parser.add_argument("--build_url",
                      required=True,
                      type=str,
                      help="CI build page url to show in the report")
  parser.add_argument("--output", type=pathlib.Path, default=None)
  parser.add_argument("--verbose",
                      action="store_true",
                      help="Print internal information during execution")

  return parser.parse_args()


def main(args):
  dashboard_api_url = f'{get_required_env_var("IREE_DASHBOARD_URL")}/apis/v2'

  benchmark_files = common_arguments.expand_and_check_file_paths(
      args.benchmark_files)
  compile_stats_files = common_arguments.expand_and_check_file_paths(
      args.compile_stats_files)

  pr_commit = args.pr_commit
  execution_benchmarks = benchmark_presentation.aggregate_all_benchmarks(
      benchmark_files=benchmark_files, expected_pr_commit=pr_commit)
  compilation_metrics = benchmark_presentation.collect_all_compilation_metrics(
      compile_stats_files=compile_stats_files, expected_pr_commit=pr_commit)

  pr_base_commit = args.pr_base_commit
  if pr_base_commit is None:
    baseline_results = None
  else:
    required_benchmark_keys = set(execution_benchmarks.keys())
    for target_name in compilation_metrics:
      for mapper in benchmark_presentation.COMPILATION_METRICS_TO_TABLE_MAPPERS:
        required_benchmark_keys.add(mapper.get_series_name(target_name))

    baseline_results = _find_comparable_benchmark_results(
        start_commit=pr_base_commit,
        required_benchmark_keys=required_benchmark_keys,
        dashboard_api_url=dashboard_api_url,
        verbose=args.verbose)

  if baseline_results is None:
    baseline_commit = None
  else:
    baseline_commit, base_benchmarks = baseline_results
    # Update the aggregate benchmarks with base numbers.
    for bench in execution_benchmarks:
      base_benchmark = base_benchmarks[bench]
      if base_benchmark["sampleUnit"] != "ns":
        raise ValueError("Only support nanoseconds for latency sample.")
      execution_benchmarks[bench].base_mean_time = base_benchmark["sample"]

    # Update the compilation metrics with base numbers.
    for target_name, metrics in compilation_metrics.items():
      updated_metrics = metrics
      for mapper in benchmark_presentation.COMPILATION_METRICS_TO_TABLE_MAPPERS:
        metric_key = mapper.get_series_name(target_name)
        base_benchmark = base_benchmarks[metric_key]
        if base_benchmark["sampleUnit"] != mapper.get_unit():
          raise ValueError("Unit of the queried sample is mismatched.")
        updated_metrics = mapper.update_base_value(updated_metrics,
                                                   base_benchmark["sample"])
      compilation_metrics[target_name] = updated_metrics

  pr_commit_link = md.link(pr_commit,
                           f"{GITHUB_IREE_REPO_PREFIX}/commit/{pr_commit}")
  commit_info_md = f"@ commit {pr_commit_link}"
  if baseline_commit is not None:
    baseline_commit_link = md.link(
        baseline_commit, f"{GITHUB_IREE_REPO_PREFIX}/commit/{baseline_commit}")
    commit_info_md += f" (vs. base {baseline_commit_link})"
  elif pr_base_commit is not None:
    commit_info_md += " (no previous benchmark results to compare)"

  comment_type_id = args.comment_type_id
  full_md, abbr_md = _get_benchmark_result_markdown(
      execution_benchmarks=execution_benchmarks,
      compilation_metrics=compilation_metrics,
      pr_url=f"{GITHUB_IREE_REPO_PREFIX}/pull/{args.pr_number}",
      build_url=args.build_url,
      comment_title=args.comment_title,
      comment_type_id=comment_type_id,
      commit_info_md=commit_info_md)

  comment_data = benchmark_comment.CommentData(type_id=comment_type_id,
                                               abbr_md=abbr_md,
                                               full_md=full_md)
  comment_json_data = json.dumps(dataclasses.asdict(comment_data), indent=2)
  if args.output is None:
    print(comment_json_data)
  else:
    args.output.write_text(comment_json_data)


if __name__ == "__main__":
  main(parse_arguments())
