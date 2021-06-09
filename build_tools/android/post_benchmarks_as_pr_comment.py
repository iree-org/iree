#!/usr/bin/env python3
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Posts benchmark results to GitHub as pull request comments.

This script is meant to be used by Buildkite for automation. It requires the
following environment to be set:

- BUILDKITE_BUILD_NUMBER: the build number of current Buildkite build.
- BUILDKITE_BUILD_URL: the link to the current Buildkite build.
- BUILDKITE_COMMIT: the pull request HEAD commit.
- BUILDKITE_PULL_REQUEST: the current pull request number.
- GITHUB_TOKEN: personal access token to authenticate against GitHub API;
    it should have "public_repo" and "gist" scope.

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

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from common.benchmark_description import BenchmarkResults, get_output

ABBR_PR_COMMENT_TITLE = "Abbreviated Benchmark Summary"
GITHUB_GIST_API_PREFIX = "https://api.github.com/gists"
GITHUB_IREE_API_PREFIX = "https://api.github.com/repos/google/iree"
GITHUB_IREE_REPO_PREFIX = "https://github.com/google/iree"
GITHUB_USER = "iree-github-actions-bot"
IREE_PROJECT_ID = 'IREE'
# The ratio below which benchmarks will be considered as similar with base.
SIMILAR_BECNHMARK_THRESHOLD = 0.05
# The max number of rows to show per table.
TABLE_SIZE_CUT = 3
THIS_DIRECTORY = os.path.dirname(os.path.realpath(__file__))


def get_required_env_var(var: str) -> str:
  """Gets the value for a required environment variable."""
  value = os.getenv(var, None)
  if value is None:
    raise RuntimeError(f'Missing environment variable "{var}"')
  return value


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


def get_origin_tree_top_commit(verbose: bool = False) -> str:
  """Returns the top of the tree commit for the origin base branch."""
  base_branch = get_required_env_var("BUILDKITE_PULL_REQUEST_BASE_BRANCH")
  get_output(['git', 'fetch', '--prune', '--', 'origin', base_branch],
             cwd=THIS_DIRECTORY,
             verbose=verbose)
  return get_git_commit_hash(f'origin/{base_branch}', verbose)


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

  data = response.json()
  if verbose:
    print(f'Queried base benchmark data: {data}')
  return data


@dataclass
class AggregateBenchmarkLatency:
  """An object for describing aggregate latency numbers for a benchmark."""
  mean_time: int
  median_time: int
  stddev_time: int
  # The average latency time for the base commit to compare against.
  base_mean_time: Optional[int] = None


def aggregate_all_benchmarks(
    benchmark_files: Sequence[str]) -> Dict[str, AggregateBenchmarkLatency]:
  """Aggregates all benchmarks in the given files.

  Args:
  - benchmark_files: A list of JSON files, each can be decoded as a
    BenchmarkResults.

  Returns:
  - A dict of benchmark names to AggregateBenchmarkLatency numbers.
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

      aggregate_results[name] = AggregateBenchmarkLatency(
          mean_time, median_time, stddev_time)

  return aggregate_results


def query_base_benchmark_results(commit,
                                 verbose: bool = False) -> Dict[str, int]:
  """Queries the benchmark results for the given commit."""
  build_id = get_git_total_commit_count(commit, verbose)

  url = get_required_env_var('IREE_DASHBOARD_URL')
  payload = {'projectId': IREE_PROJECT_ID, 'buildId': build_id}
  return get_from_dashboard(f'{url}/apis/getBuild', payload, verbose=verbose)


def add_header_and_get_markdown_table(names: Tuple[str],
                                      means: Tuple[Any],
                                      medians: Tuple[int],
                                      stddevs: Tuple[int],
                                      size_cut: Optional[int] = None) -> str:
  """Generates a markdown table with proper headers for benchmarks.

  Args:
  - size_cut: If not None, only show the top N results for each table.
  """
  total_size = len(names)
  if size_cut is not None:
    names = names[0:size_cut]
    means = means[0:size_cut]
    medians = medians[0:size_cut]
    stddevs = stddevs[0:size_cut]

  names = ("Benchmark Name",) + names
  means = ("Average Latency (ms)",) + means
  medians = ("Median Latency (ms)",) + medians
  stddevs = ("Latency Standard Deviation (ms)",) + stddevs

  table_str = md.table([names, means, medians, stddevs])
  if size_cut is not None and size_cut < total_size:
    table_str += "\n\n"
    table_str += md.italics(
        f"[Top {size_cut} out of {total_size} benchmark results showed]")
  return table_str


def sort_benchmarks_and_get_table(benchmarks: Dict[str,
                                                   AggregateBenchmarkLatency],
                                  size_cut: Optional[int] = None):
  """Sorts all benchmarks according to the improvement/regression ratio and
  returns a markdown table for it.

  Args:
  - size_cut: If not None, only show the top N results for each table.
  """
  sorted_benchmarks = []
  for k, v in benchmarks.items():
    ratio = abs(v.mean_time - v.base_mean_time) / v.base_mean_time
    sorted_benchmarks.append((k, (v.mean_time, v.base_mean_time, ratio),
                              v.median_time, v.stddev_time))
  # Sort according to ratio in the reverse order.
  sorted_benchmarks.sort(key=lambda benchmark: benchmark[1][2], reverse=True)

  # Split each field into its own tuple in prepration for markdown table.
  names, means, medians, stddevs = zip(*sorted_benchmarks)

  # Turn the tuple about means into a string representation.
  str_means = []
  for pr, base, ratio in means:
    direction = "↑" if pr > base else ("↓" if pr < base else "")
    str_means.append(f"{pr} (vs. {base}, {ratio:.2%}{direction})")
  str_means = tuple(str_means)

  return add_header_and_get_markdown_table(names, str_means, medians, stddevs,
                                           size_cut)


def categorize_benchmarks_into_tables(benchmarks: Dict[
    str, AggregateBenchmarkLatency],
                                      similar_threshold: float,
                                      size_cut: Optional[int] = None) -> str:
  """Splits benchmarks into regressed/improved/similar/raw categories and
  returns their markdown tables.

    Args:
    - similar_threshold: the threshold under which a benchmark will be
        considered as similar to its base commit.
    - size_cut: If not None, only show the top N results for each table.
    """
  regressed, improved, similar, raw = {}, {}, {}, {}

  for name, results in benchmarks.items():
    # If no informatio about the base result. Then we cannot analyze.
    if results.base_mean_time is None:
      raw[name] = results
      continue

    current = results.mean_time
    base = results.base_mean_time
    ratio = abs(current - base) / base
    if ratio <= similar_threshold:
      similar[name] = results
    elif current > base:
      regressed[name] = results
    else:
      improved[name] = results

  tables = []
  if regressed:
    tables.append(md.header("Regressed Benchmarks 🚩", 3))
    tables.append(sort_benchmarks_and_get_table(regressed, size_cut))
  if improved:
    tables.append(md.header("Improved Benchmarks 🎉", 3))
    tables.append(sort_benchmarks_and_get_table(improved, size_cut))
  # If we want to abbreviate, similar results won't be interesting.
  if similar and size_cut is None:
    tables.append(md.header("Similar Benchmarks", 3))
    tables.append(sort_benchmarks_and_get_table(similar, size_cut))
  if raw:
    tables.append(md.header("Similar Benchmarks", 3))
    raw_list = [
        (k, v.mean_time, v.median_time, v.stddev_time) for k, v in raw.items()
    ]
    names, means, medians, stddevs = zip(*raw_list)
    tables.append(
        add_header_and_get_markdown_table(names=names,
                                          means=means,
                                          medians=medians,
                                          stddevs=stddevs,
                                          size_cut=size_cut))
  return "\n\n".join(tables)


def get_benchmark_result_markdown(benchmark_files: Sequence[str],
                                  query_base: bool,
                                  verbose: bool = False) -> Tuple[str, str]:
  """Gets the full/abbreviated markdown summary of all benchmarks in files."""
  all_benchmarks = aggregate_all_benchmarks(benchmark_files)

  build_url = get_required_env_var("BUILDKITE_BUILD_URL")
  pr_number = get_required_env_var("BUILDKITE_PULL_REQUEST")
  pr_commit = get_required_env_var("BUILDKITE_COMMIT")
  pr_commit = md.link(pr_commit,
                      f"{GITHUB_IREE_REPO_PREFIX}/commit/{pr_commit}")
  if query_base:
    # Update the aggregate benchmarks with base numbers.
    base_commit = get_origin_tree_top_commit(verbose)
    base_benchmarks = query_base_benchmark_results(base_commit, verbose)
    for bench in base_benchmarks:
      if bench in all_benchmarks:
        all_benchmarks[bench].base_mean_time = base_benchmarks[bench]
    base_commit = md.link(base_commit,
                          f"{GITHUB_IREE_REPO_PREFIX}/commit/{base_commit}")
    commit_info = f"@ commit {pr_commit} (vs. base {base_commit})"
  else:
    commit_info = f"@ commit {pr_commit}"

  pr_info = md.link("Pull request",
                    f"{GITHUB_IREE_REPO_PREFIX}/pull/{pr_number}")
  buildkite_info = md.link("Buildkite build", build_url)

  # Compose the full benchmark tables.
  full_table = [md.header("Full Benchmark Summary", 2)]
  full_table.append(md.unordered_list([commit_info, pr_info, buildkite_info]))
  full_table.append(
      categorize_benchmarks_into_tables(all_benchmarks,
                                        SIMILAR_BECNHMARK_THRESHOLD))

  # Compose the abbreviated benchmark tables.
  abbr_table = [md.header(ABBR_PR_COMMENT_TITLE, 2)]
  abbr_table.append(commit_info)
  abbr_table.append(
      categorize_benchmarks_into_tables(all_benchmarks,
                                        SIMILAR_BECNHMARK_THRESHOLD,
                                        TABLE_SIZE_CUT))
  abbr_table.append("For more information:")
  # We don't know until a Gist is really created. Use a placeholder for now
  # and replace later.
  full_result_info = md.link("Full benchmark result tables",
                             "<<placeholder-link>>")
  abbr_table.append(md.unordered_list([full_result_info, buildkite_info]))

  return "\n\n".join(full_table), "\n\n".join(abbr_table)


def post_to_gist(filename: str, content: str, verbose: bool = False):
  """Posts the given content to a new GitHub Gist and returns the URL to it."""
  api_token = get_required_env_var('GITHUB_TOKEN')
  headers = {
      "Accept": "application/vnd.github.v3+json",
      "Authorization": f"token {api_token}",
  }
  payload = json.dumps({
      "public": True,
      "files": {
          filename: {
              "content": content
          }
      }
  })

  api_endpoint = GITHUB_GIST_API_PREFIX
  response = requests.post(api_endpoint, data=payload, headers=headers)
  if response.status_code != 201:
    raise requests.RequestException(
        f"Failed to comment on GitHub; error code: {response.status_code}")

  response = response.json()
  if verbose:
    print(f"Gist posting response: {response}")

  if response["truncated"]:
    raise requests.RequestException(f"Content too large and gotten truncated")

  gist_id = response["id"]
  return f"https://gist.github.com/{GITHUB_USER}/{gist_id}"


def get_previous_comment_on_pr(pr_number: str,
                               verbose: bool = False) -> Optional[int]:
  """Gets the previous comment's ID from GitHub."""
  # Increasing per_page limit requires user authentication.
  api_token = get_required_env_var('GITHUB_TOKEN')
  headers = {
      "Accept": "application/vnd.github.v3+json",
      "Authorization": f"token {api_token}",
  }
  payload = json.dumps({"per_page": 100})

  api_endpoint = f"{GITHUB_IREE_API_PREFIX}/issues/{pr_number}/comments"
  response = requests.get(api_endpoint, data=payload, headers=headers)
  if response.status_code != 200:
    raise requests.RequestException(
        f"Failed to get PR comments from GitHub; error code: {response.status_code}"
    )

  response = response.json()
  if verbose:
    print(f"Previous comment query response: {response}")

  # Find the last comment from GITHUB_USER and has the ABBR_PR_COMMENT_TITILE
  # keyword.
  for comment in reversed(response):
    if (comment["user"]["login"] == GITHUB_USER) and (ABBR_PR_COMMENT_TITLE
                                                      in comment["body"]):
      return comment["id"]
  return None


def create_comment_on_pr(pr_number: str, content: str, verbose: bool = False):
  """Posts the given content as comments to the current pull request."""
  api_token = get_required_env_var('GITHUB_TOKEN')
  headers = {
      "Accept": "application/vnd.github.v3+json",
      "Authorization": f"token {api_token}",
  }
  payload = json.dumps({"body": content})

  api_endpoint = f"{GITHUB_IREE_API_PREFIX}/issues/{pr_number}/comments"
  response = requests.post(api_endpoint, data=payload, headers=headers)
  if response.status_code != 201:
    raise requests.RequestException(
        f"Failed to comment on GitHub; error code: {response.status_code}")


def update_comment_on_pr(comment_id: int, content: str, verbose: bool = False):
  """Updates the content of the given comment."""
  api_token = get_required_env_var('GITHUB_TOKEN')
  headers = {
      "Accept": "application/vnd.github.v3+json",
      "Authorization": f"token {api_token}",
  }
  payload = json.dumps({"body": content})

  api_endpoint = f"{GITHUB_IREE_API_PREFIX}/issues/comments/{comment_id}"
  response = requests.patch(api_endpoint, data=payload, headers=headers)
  if response.status_code != 200:
    raise requests.RequestException(
        f"Failed to comment on GitHub; error code: {response.status_code}")


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
  full_md, abbr_md = get_benchmark_result_markdown(args.benchmark_files,
                                                   query_base=args.query_base,
                                                   verbose=args.verbose)

  if args.dry_run:
    print(full_md, "\n\n", abbr_md)
    return

  pr_number = get_required_env_var("BUILDKITE_PULL_REQUEST")
  # Buildkite sets this to "false" if not running on a PR:
  # https://buildkite.com/docs/pipelines/environment-variables#bk-env-vars-buildkite-pull-request
  if pr_number == "false":
    raise ValueError("Not a pull request")

  build_number = get_required_env_var("BUILDKITE_BUILD_NUMBER")
  filename = f"iree-full-benchmark-result-{build_number}.md"
  gist_url = post_to_gist(filename, full_md, args.verbose)
  abbr_md = abbr_md.replace("<<placeholder-link>>", gist_url)

  previous_comment = get_previous_comment_on_pr(pr_number, args.verbose)
  if previous_comment is not None:
    update_comment_on_pr(previous_comment, abbr_md, args.verbose)
  else:
    create_comment_on_pr(pr_number, abbr_md, args.verbose)


if __name__ == "__main__":
  main(parse_arguments())
