#!/usr/bin/env python3

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Determines whether CI should run on a given PR.

The following environment variables are required:
- GITHUB_EVENT_NAME: GitHub event name, e.g. pull_request.
- GITHUB_OUTPUT: path to write workflow output variables.
- GITHUB_STEP_SUMMARY: path to write workflow summary output.

When GITHUB_EVENT_NAME is "pull_request", there are additional environment
variables to be set:
- PR_BRANCH (required): PR source branch.
- PR_TITLE (required): PR title.
- PR_BODY (optional): PR description.
- PR_LABELS (optional): JSON list of PR label names.
- BASE_REF (required): base commit SHA of the PR.
- ORIGINAL_PR_TITLE (optional): PR title from the original PR event, showing a
    notice if PR_TITLE is different.
- ORIGINAL_PR_BODY (optional): PR description from the original PR event,
    showing a notice if PR_BODY is different. ORIGINAL_PR_TITLE must also be
    set.
- ORIGINAL_PR_LABELS (optional): PR labels from the original PR event, showing a
    notice if PR_LABELS is different. ORIGINAL_PR_TITLE must also be set.

Exit code 0 indicates that it should and exit code 2 indicates that it should
not.
"""

import difflib
import fnmatch
import json
import os
import re
import subprocess
import textwrap
from typing import Iterable, List, Mapping, Sequence, Tuple

SKIP_CI_KEY = "skip-ci"
RUNNER_ENV_KEY = "runner-env"
BENCHMARK_PRESET_KEY = "benchmarks"
# Trailer to prevent benchmarks from always running on LLVM integration PRs.
SKIP_LLVM_INTEGRATE_BENCHMARK_KEY = "skip-llvm-integrate-benchmark"

# Note that these are fnmatch patterns, which are not the same as gitignore
# patterns because they don't treat '/' specially. The standard library doesn't
# contain a function for gitignore style "wildmatch". There's a third-party
# library pathspec (https://pypi.org/project/pathspec/), but it doesn't seem
# worth the dependency.
SKIP_PATH_PATTERNS = [
    "docs/*",
    "third_party/mkdocs-material/*",
    "experimental/*",
    "build_tools/buildkite/*",
    # These configure the runners themselves and don't affect presubmit.
    "build_tools/github_actions/runner/*",
    ".github/ISSUE_TEMPLATE/*",
    "*.cff",
    "*.clang-format",
    "*.git-ignore",
    "*.md",
    "*.natvis",
    "*.pylintrc",
    "*.rst",
    "*.toml",
    "*.yamllint.yml",
    "*.yapf",
    "*CODEOWNERS",
    "*AUTHORS",
    "*LICENSE",
]

RUNNER_ENV_DEFAULT = "prod"
RUNNER_ENV_OPTIONS = [RUNNER_ENV_DEFAULT, "testing"]

BENCHMARK_PRESET_OPTIONS = ["all", "cuda", "x86_64", "comp-stats"]

PR_DESCRIPTION_TEMPLATE = "{title}" "\n\n" "{body}"

# Patterns to detect "LLVM integration" PRs, i.e. changes that update the
# third_party/llvm-project submodule. This should only include PRs
# intended to be merged and should exclude test/draft PRs as well as
# PRs that include temporary patches to the submodule during review.
# See also: https://github.com/openxla/iree/issues/12268
LLVM_INTEGRATE_TITLE_PATTERN = re.compile("^integrate.+llvm-project",
                                          re.IGNORECASE)
LLVM_INTEGRATE_BRANCH_PATTERN = re.compile("bump-llvm|llvm-bump", re.IGNORECASE)
LLVM_INTEGRATE_LABEL = "llvm-integrate"


def skip_path(path: str) -> bool:
  return any(fnmatch.fnmatch(path, pattern) for pattern in SKIP_PATH_PATTERNS)


def set_output(d: Mapping[str, str]):
  print(f"Setting outputs: {d}")
  step_output_file = os.environ["GITHUB_OUTPUT"]
  with open(step_output_file, "a") as f:
    f.writelines(f"{k}={v}" + "\n" for k, v in d.items())


def write_job_summary(summary: str):
  """Write markdown messages on Github workflow UI.
  See https://docs.github.com/en/actions/using-workflows/workflow-commands-for-github-actions#adding-a-job-summary
  """
  step_summary_file = os.environ["GITHUB_STEP_SUMMARY"]
  with open(step_summary_file, "a") as f:
    # Use double newlines to split sections in markdown.
    f.write(summary + "\n\n")


def check_description_and_show_diff(original_description: str,
                                    original_labels: Sequence[str],
                                    current_description: str,
                                    current_labels: Sequence[str]):
  original_labels = sorted(original_labels)
  current_labels = sorted(current_labels)
  if (original_description == current_description and
      original_labels == current_labels):
    return

  description_diffs = difflib.unified_diff(
      original_description.splitlines(keepends=True),
      current_description.splitlines(keepends=True))
  description_diffs = "".join(description_diffs)

  if description_diffs != "":
    description_diffs = textwrap.dedent("""\
    ```diff
    {}
    ```
    """).format(description_diffs)

  if original_labels == current_labels:
    label_diffs = ""
  else:
    label_diffs = textwrap.dedent("""\
    ```
    Original labels: {original_labels}
    Current labels: {current_labels}
    ```
    """).format(original_labels=original_labels, current_labels=current_labels)

  write_job_summary(
      textwrap.dedent("""\
  :pushpin: Using the PR description and labels different from the original PR event that started this workflow.

  <details>
  <summary>Click to show diff (original vs. current)</summary>

  {description_diffs}

  {label_diffs}
  </details>""").format(description_diffs=description_diffs,
                        label_diffs=label_diffs))


def get_trailers_and_labels(is_pr: bool) -> Tuple[Mapping[str, str], List[str]]:
  if not is_pr:
    return ({}, [])

  title = os.environ["PR_TITLE"]
  body = os.environ.get("PR_BODY", "")
  labels = json.loads(os.environ.get("PR_LABELS", "[]"))
  original_title = os.environ.get("ORIGINAL_PR_TITLE")
  original_body = os.environ.get("ORIGINAL_PR_BODY", "")
  original_labels = json.loads(os.environ.get("ORIGINAL_PR_LABELS", "[]"))

  description = PR_DESCRIPTION_TEMPLATE.format(title=title, body=body)

  # PR information can be fetched from API for the latest updates. If
  # ORIGINAL_PR_TITLE is set, compare the current and original description and
  # show a notice if they are different. This is mostly to inform users that the
  # workflow might not parse the PR description they expect.
  if original_title is not None:
    original_description = PR_DESCRIPTION_TEMPLATE.format(title=original_title,
                                                          body=original_body)
    print("Original PR description and labels:",
          original_description,
          original_labels,
          sep="\n")
    check_description_and_show_diff(original_description=original_description,
                                    original_labels=original_labels,
                                    current_description=description,
                                    current_labels=labels)

  print("Parsing PR description and labels:", description, labels, sep="\n")

  trailer_lines = subprocess.run(
      ["git", "interpret-trailers", "--parse", "--no-divider"],
      input=description,
      stdout=subprocess.PIPE,
      check=True,
      text=True,
      timeout=60).stdout.splitlines()
  trailer_map = {
      k.lower().strip(): v.strip()
      for k, v in (line.split(":", maxsplit=1) for line in trailer_lines)
  }
  return (trailer_map, labels)


def get_modified_paths(base_ref: str) -> Iterable[str]:
  return subprocess.run(["git", "diff", "--name-only", base_ref],
                        stdout=subprocess.PIPE,
                        check=True,
                        text=True,
                        timeout=60).stdout.splitlines()


def modifies_included_path(base_ref: str) -> bool:
  return any(not skip_path(p) for p in get_modified_paths(base_ref))


def should_run_ci(is_pr: bool, trailers: Mapping[str, str]) -> bool:
  if not is_pr:
    print("Running CI independent of diff because run was not triggered by a"
          " pull request event.")
    return True

  if SKIP_CI_KEY in trailers:
    print(f"Not running CI because PR description has '{SKIP_CI_KEY}' trailer.")
    return False

  base_ref = os.environ["BASE_REF"]
  try:
    modifies = modifies_included_path(base_ref)
  except TimeoutError as e:
    print("Computing modified files timed out. Running the CI")
    return True

  if not modifies:
    print("Skipping CI because all modified files are marked as excluded.")
    return False

  print("CI should run")
  return True


def get_runner_env(trailers: Mapping[str, str]) -> str:
  runner_env = trailers.get(RUNNER_ENV_KEY)
  if runner_env is None:
    print(f"Using '{RUNNER_ENV_DEFAULT}' runners because '{RUNNER_ENV_KEY}'"
          f" not found in {trailers}")
    runner_env = RUNNER_ENV_DEFAULT
  else:
    print(
        f"Using runner environment '{runner_env}' from PR description trailers")
  return runner_env


def get_benchmark_presets(trailers: Mapping[str, str], labels: Sequence[str],
                          is_pr: bool, is_llvm_integrate_pr: bool) -> str:
  """Parses and validates the benchmark presets from trailers.

  Args:
    trailers: trailers from PR description.
    labels: list of PR labels.
    is_pr: is pull request event.
    is_llvm_integrate_pr: is LLVM integration PR.

  Returns:
    A comma separated preset string, which later will be parsed by
    build_tools/benchmarks/export_benchmark_config.py.
  """

  skip_llvm_integrate_benchmark = SKIP_LLVM_INTEGRATE_BENCHMARK_KEY in trailers
  if skip_llvm_integrate_benchmark:
    print("Skipping default benchmarking on LLVM integration because PR "
          f"description has '{SKIP_LLVM_INTEGRATE_BENCHMARK_KEY}' trailer.")

  if not is_pr:
    preset_options = ["all"]
    print(f"Using benchmark preset 'all' for non-PR run")
  elif is_llvm_integrate_pr and not skip_llvm_integrate_benchmark:
    # Run all benchmark presets for LLVM integration PRs.
    preset_options = ["all"]
    print("Using benchmark preset 'all' for LLVM integration PR")
  else:
    preset_options = set(
        label.split(":", maxsplit=1)[1]
        for label in labels
        if label.startswith(BENCHMARK_PRESET_KEY + ":"))
    trailer = trailers.get(BENCHMARK_PRESET_KEY)
    if trailer is not None:
      preset_options = preset_options.union(
          option.strip() for option in trailer.split(","))
    preset_options = sorted(preset_options)
    print(f"Using benchmark preset '{preset_options}' from trailers and labels")

  for preset_option in preset_options:
    if preset_option not in BENCHMARK_PRESET_OPTIONS:
      raise ValueError(f"Unknown benchmark preset option: '{preset_option}'.\n"
                       f"Available options: '{BENCHMARK_PRESET_OPTIONS}'.")

  if "all" in preset_options:
    preset_options = list(
        option for option in BENCHMARK_PRESET_OPTIONS if option != "all")

  return ",".join(preset_options)


def main():
  is_pr = os.environ["GITHUB_EVENT_NAME"] == "pull_request"
  trailers, labels = get_trailers_and_labels(is_pr)
  is_llvm_integrate_pr = bool(
      LLVM_INTEGRATE_TITLE_PATTERN.search(os.environ.get("PR_TITLE", "")) or
      LLVM_INTEGRATE_BRANCH_PATTERN.search(os.environ.get("PR_BRANCH", "")) or
      LLVM_INTEGRATE_LABEL in labels)
  output = {
      "should-run":
          json.dumps(should_run_ci(is_pr, trailers)),
      "is-pr":
          json.dumps(is_pr),
      "runner-env":
          get_runner_env(trailers),
      "runner-group":
          "presubmit" if is_pr else "postsubmit",
      "write-caches":
          "0" if is_pr else "1",
      "benchmark-presets":
          get_benchmark_presets(trailers, labels, is_pr, is_llvm_integrate_pr),
  }

  set_output(output)


if __name__ == "__main__":
  main()
