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

from common.benchmark_definition import execute_cmd_and_get_output
from common.benchmark_presentation import *

ABBR_PR_COMMENT_TITLE = "Abbreviated Benchmark Summary"
GITHUB_GIST_API_PREFIX = "https://api.github.com/gists"
GITHUB_IREE_API_PREFIX = "https://api.github.com/repos/iree-org/iree"
GITHUB_IREE_REPO_PREFIX = "https://github.com/iree-org/iree"
GITHUB_USER = "iree-github-actions-bot"
IREE_PROJECT_ID = 'IREE'
# The maximal numbers of trials when querying base commit benchmark results.
MAX_BASE_COMMIT_QUERY_COUNT = 10
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
  return execute_cmd_and_get_output(['git', 'rev-parse', commit],
                                    cwd=THIS_DIRECTORY,
                                    verbose=verbose)


def get_git_total_commit_count(commit: str, verbose: bool = False) -> int:
  """Gets the total commit count in history ending with the given commit."""
  count = execute_cmd_and_get_output(['git', 'rev-list', '--count', commit],
                                     cwd=THIS_DIRECTORY,
                                     verbose=verbose)
  return int(count)


def get_origin_tree_commit(distance: int, verbose: bool = False) -> str:
  """Returns the hash for the commit with the given distance from top of the
  tree for the origin base branch."""
  base_branch = get_required_env_var("BUILDKITE_PULL_REQUEST_BASE_BRANCH")
  execute_cmd_and_get_output(
      ['git', 'fetch', '--prune', '--', 'origin', base_branch],
      cwd=THIS_DIRECTORY,
      verbose=verbose)
  return get_git_commit_hash(f'origin/{base_branch}~{distance}', verbose)


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


def query_base_benchmark_results(commit,
                                 verbose: bool = False) -> Dict[str, int]:
  """Queries the benchmark results for the given commit."""
  build_id = get_git_total_commit_count(commit, verbose)

  url = get_required_env_var('IREE_DASHBOARD_URL')
  payload = {'projectId': IREE_PROJECT_ID, 'buildId': build_id}
  return get_from_dashboard(f'{url}/apis/getBuild', payload, verbose=verbose)


def get_benchmark_result_markdown(benchmark_files: Sequence[str],
                                  query_base: bool,
                                  comment_title: str,
                                  verbose: bool = False) -> Tuple[str, str]:
  """Gets the full/abbreviated markdown summary of all benchmarks in files."""
  pr_commit = get_required_env_var("BUILDKITE_COMMIT")
  all_benchmarks = aggregate_all_benchmarks(benchmark_files,
                                            pr_commit,
                                            verbose=verbose)

  build_url = get_required_env_var("BUILDKITE_BUILD_URL")
  pr_number = get_required_env_var("BUILDKITE_PULL_REQUEST")
  pr_commit = md.link(pr_commit,
                      f"{GITHUB_IREE_REPO_PREFIX}/commit/{pr_commit}")

  commit_info = f"@ commit {pr_commit}"
  if query_base:
    # Try to query some base benchmark to diff against, from the top of the
    # tree. Bail out if the maximal trial number is exceeded.
    for i in range(MAX_BASE_COMMIT_QUERY_COUNT):
      base_commit = get_origin_tree_commit(i, verbose)
      base_benchmarks = query_base_benchmark_results(base_commit, verbose)
      base_commit = md.link(base_commit,
                            f"{GITHUB_IREE_REPO_PREFIX}/commit/{base_commit}")

      # Skip if the base doesn't contain all benchmarks to be compared.
      if not (set(all_benchmarks.keys()) <= set(base_benchmarks.keys())):
        commit_info = (f"@ commit {pr_commit} (no previous benchmark results to"
                       f" compare against since {base_commit})")
        continue

      # Update the aggregate benchmarks with base numbers.
      for bench in all_benchmarks:
        all_benchmarks[bench].base_mean_time = base_benchmarks[bench]

      commit_info = f"@ commit {pr_commit} (vs. base {base_commit})"
      break

  pr_info = md.link("Pull request",
                    f"{GITHUB_IREE_REPO_PREFIX}/pull/{pr_number}")
  buildkite_info = md.link("Buildkite build", build_url)

  # Compose the full benchmark tables.
  full_table = [md.header("Full Benchmark Summary", 2)]
  full_table.append(md.unordered_list([commit_info, pr_info, buildkite_info]))
  full_table.append(categorize_benchmarks_into_tables(all_benchmarks))

  # Compose the abbreviated benchmark tables.
  abbr_table = [md.header(comment_title, 2)]
  abbr_table.append(commit_info)
  tables = categorize_benchmarks_into_tables(all_benchmarks, TABLE_SIZE_CUT)
  if len(tables) == 0:
    abbr_table.append("No improved or regressed benchmarks 🏖️")
  else:
    abbr_table.append(tables)
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
                               comment_title: str,
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
    escaped_title = md.esc_format(comment_title)
    if (comment["user"]["login"] == GITHUB_USER) and (escaped_title
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
  parser.add_argument("--comment-title",
                      default=ABBR_PR_COMMENT_TITLE,
                      help="Title of the comment")
  parser.add_argument("--verbose",
                      action="store_true",
                      help="Print internal information during execution")
  args = parser.parse_args()

  return args


def main(args):
  full_md, abbr_md = get_benchmark_result_markdown(
      args.benchmark_files,
      query_base=args.query_base,
      comment_title=args.comment_title,
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

  previous_comment = get_previous_comment_on_pr(
      pr_number, comment_title=args.comment_title, verbose=args.verbose)
  if previous_comment is not None:
    update_comment_on_pr(previous_comment, abbr_md, args.verbose)
  else:
    create_comment_on_pr(pr_number, abbr_md, args.verbose)


if __name__ == "__main__":
  main(parse_arguments())
