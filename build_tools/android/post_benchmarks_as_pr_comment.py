#!/usr/bin/env python3
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Posts benchmark results to GitHub as pull request comments.

This script is meant to be used by Buildkite for automation. It requires the
following environment to be set:

- BUILDKITE_BUILD_URL: the link to the current Buildkite build.
- BUILDKITE_COMMIT: the pull request HEAD commit.
- BUILDKITE_PULL_REQUEST: the current pull request number.
- GITHUB_TOKEN: personal access token to authenticate against GitHub API.

if --query-base in toggled on, then it additionally requires:

- BUILDKITE_PULL_REQUEST_BASE_BRANCH: the targeting base branch.
- IREE_DASHBOARD_URL: the url to IREE's performance dashboard.

This script uses pip package "markdown_strings".

Example usage:
  # Export necessary environment variables:
  export ...
  # Then run the script:
  python3 post_benchmarks_as_pr_comment.py <benchmark-json-file>...
  #   where each <benchmark-json-file> is expected to be of format expected
  #   by BenchmarkResults objects.
"""

import argparse
import json
import os
import requests
import markdown_strings as md

from typing import Any, Dict, Sequence, Tuple, Union

from common.benchmark_description import BenchmarkResults, get_output

GITHUB_IREE_API_PREFIX = "https://api.github.com/repos/google/iree"
IREE_PROJECT_ID = 'IREE'
THIS_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
RESULT_EMPHASIS_THRESHOLD = 0.05


def get_git_commit_hash(commit: str, verbose: bool = False) -> str:
  """Gets the commit hash for the given commit."""
  return get_output(['git', 'rev-parse', commit],
                    cwd=THIS_DIRECTORY,
                    verbose=verbose)


def get_git_total_commit_count(commit: str, verbose: bool = False) -> int:
  """Gets the total commit count in history ending with the given commit."""
  count = get_output(['git', 'rev-list', '--count', commit],
                     cwd=THIS_DIRECTORY,
                     verbose=verbose)
  return int(count)


def get_required_env_var(var: str) -> str:
  """Gets the value for a required environment variable."""
  value = os.getenv(var, None)
  if value is None:
    raise RuntimeError(f'Missing environment variable "{var}"')
  return value


def get_from_dashboard(url: str,
                       payload: Dict[str, Any],
                       verbose: bool = False) -> Dict[str, int]:
  headers = {'Content-type': 'application/json'}
  data = json.dumps(payload)

  if verbose:
    print(f'API request payload: {data}')

  response = requests.get(url, data=data, headers=headers)
  code = response.status_code
  if code != 200:
    raise requests.RequestException(
        f'Failed to get from dashboard server with status code {code}')

  return response.json()


def aggregate_all_benchmarks(
    benchmark_files: Sequence[str]) -> Sequence[Tuple[Union[str, int]]]:
  """Aggregates all benchmarks in the given files.

  Args:
  - benchmark_files: A list of JSON files, each can be decoded as a
    BenchmarkResults.

  Returns:
  - A list of (name, mean-latency, median-latency, stddev-latency) tuples.
  """

  pr_commit = get_required_env_var("BUILDKITE_COMMIT")
  aggregate_results = {}

  for benchmark_file in benchmark_files:
    with open(benchmark_file) as f:
      content = f.read()
    file_results = BenchmarkResults.from_json_str(content)

    if file_results.commit != pr_commit:
      raise ValueError("Inconsistent pull request commit")

    for benchmark_index in range(len(file_results.benchmarks)):
      benchmark_case = file_results.benchmarks[benchmark_index]

      # Make sure each benchmark has a unique name.
      name = str(benchmark_case["benchmark"])
      if name in aggregate_results:
        raise ValueError(f"Duplicated benchmarks: {name}")

      # Now scan all benchmark iterations and find the aggregate results.
      mean_time = file_results.get_aggregate_time(benchmark_index, "mean")
      median_time = file_results.get_aggregate_time(benchmark_index, "median")
      stddev_time = file_results.get_aggregate_time(benchmark_index, "stddev")

      aggregate_results[name] = (mean_time, median_time, stddev_time)

  return sorted([(k,) + v for k, v in aggregate_results.items()])


def query_base_benchmark_results(commit,
                                 verbose: bool = False) -> Dict[str, int]:
  """Queries the benchmark results for the given commit."""
  build_id = get_git_total_commit_count(commit, verbose)

  url = get_required_env_var('IREE_DASHBOARD_URL')
  payload = {'projectId': IREE_PROJECT_ID, 'buildId': build_id}
  return get_from_dashboard(f'{url}/apis/getBuild', payload, verbose=verbose)


def get_comparsion_against_base(pr_means: Sequence[int],
                                base_means: Sequence[int]) -> Sequence[str]:
  """Returns a tuple of strings comparsing mean latency numbers."""
  comparisions = []

  for pr, base in zip(pr_means, base_means):
    if base is None:
      comparisions.append(str(pr))
      continue

    diff = abs(pr - base) / base
    if pr > base:
      percent = "{:.2%}".format(diff)
      direction = "↑"
      if diff > RESULT_EMPHASIS_THRESHOLD:
        direction += ", 🚩"
    elif pr < base:
      percent = "{:.2%}".format(diff)
      direction = "↓"
      if diff > RESULT_EMPHASIS_THRESHOLD:
        direction += ", 🎉"
    else:
      percent = "{:.0%}".format(diff)
      direction = ""

    comparisions.append(f"{pr} (vs. {base}, {percent}{direction})")

  return tuple(comparisions)


def get_benchmark_result_markdown(benchmark_files: Sequence[str],
                                  query_base: bool,
                                  verbose: bool = False) -> str:
  """Gets markdown summary of all benchmarks in the given files."""
  all_benchmarks = aggregate_all_benchmarks(benchmark_files)
  names, means, medians, stddevs = zip(*all_benchmarks)

  build_url = get_required_env_var("BUILDKITE_BUILD_URL")
  pr_commit = get_required_env_var("BUILDKITE_COMMIT")
  if query_base:
    base_branch = get_required_env_var("BUILDKITE_PULL_REQUEST_BASE_BRANCH")
    commit = get_git_commit_hash(base_branch, verbose)
    base_benchmarks = query_base_benchmark_results(commit, verbose)
    base_means = [base_benchmarks.get(v) for v in names]
    means = get_comparsion_against_base(means, base_means)
    commit_info = f"@ commit {pr_commit} (vs. base {commit})"
  else:
    commit_info = f"@ commit {pr_commit}"

  names = ("Benchmark Name",) + names
  means = ("Average Latency (ms)",) + means
  medians = ("Median Latency (ms)",) + medians
  stddevs = ("Latency Standard Deviation (ms)",) + stddevs

  header = md.header("Benchmark results", 3)
  benchmark_table = md.table([names, means, medians, stddevs])
  link = "See more details on " + md.link("Buildkite", build_url)

  return "\n\n".join([header, commit_info, benchmark_table, link])


def comment_on_pr(content):
  """Posts the given content as comments to the current pull request."""
  pr_number = get_required_env_var("BUILDKITE_PULL_REQUEST")
  # Buildkite sets this to "false" if not running on a PR:
  # https://buildkite.com/docs/pipelines/environment-variables#bk-env-vars-buildkite-pull-request
  if pr_number == "false":
    raise ValueError("Not a pull request")

  api_token = get_required_env_var('GITHUB_TOKEN')
  headers = {
      "Accept": "application/vnd.github.v3+json",
      "Authorization": f"token {api_token}",
  }
  payload = json.dumps({"event": "COMMENT", "body": content})

  api_endpoint = f"{GITHUB_IREE_API_PREFIX}/pulls/{pr_number}/reviews"
  request = requests.post(api_endpoint, data=payload, headers=headers)
  if request.status_code != 200:
    raise requests.RequestException(
        f"Failed to comment on GitHub; error code: {request.status_code}")


def parse_arguments():
  """Parses command-line options."""

  def check_file_path(path):
    if os.path.isfile(path):
      return path
    else:
      raise ValueError(path)

  parser = argparse.ArgumentParser()
  parser.add_argument("benchmark_files",
                      metavar="<benchmark-json-file>",
                      type=check_file_path,
                      nargs="+",
                      help="Path to the JSON file containing benchmark results")
  parser.add_argument("--dry-run",
                      action="store_true",
                      help="Print the comment instead of posting to GitHub")
  parser.add_argument(
      "--query-base",
      action="store_true",
      help=
      "Query the dashboard for the benchmark results of the targeting base branch"
  )
  parser.add_argument("--verbose",
                      action="store_true",
                      help="Print internal information during execution")
  args = parser.parse_args()

  return args


def main(args):
  benchmarks_md = get_benchmark_result_markdown(args.benchmark_files,
                                                query_base=args.query_base,
                                                verbose=args.verbose)

  if args.dry_run:
    print(benchmarks_md)
  else:
    comment_on_pr(benchmarks_md)


if __name__ == "__main__":
  main(parse_arguments())
