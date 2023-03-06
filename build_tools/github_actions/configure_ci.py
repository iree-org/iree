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
- PR_TITLE (required): PR title.
- PR_BODY (optional): PR description.
- BASE_REF (required): base commit SHA of the PR.
- ORIGINAL_PR_TITLE (optional): PR title from the original PR event, showing a
    notice if PR_TITLE is different.
- ORIGINAL_PR_BODY (optional): PR description from the original PR event,
    showing a notice if PR_BODY is different. ORIGINAL_PR_TITLE must also be
    set.

Exit code 0 indicates that it should and exit code 2 indicates that it should
not.
"""

import difflib
import fnmatch
import os
import subprocess
import textwrap
from typing import Iterable, Mapping, MutableMapping

PULL_REQUEST_EVENT_NAME = "pull_request"
PUSH_EVENT_NAME = "push"
SCHEDULE_EVENT_NAME = "schedule"
WORKFLOW_DISPATCH_EVENT_NAME = "workflow_dispatch"
SKIP_CI_KEY = "skip-ci"
RUNNER_ENV_KEY = "runner-env"
BENCHMARK_PRESET_KEY = "benchmarks"

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


def skip_path(path: str) -> bool:
  return any(fnmatch.fnmatch(path, pattern) for pattern in SKIP_PATH_PATTERNS)


def set_output(d: Mapping[str, str]):
  print(f"Setting outputs: {d}")
  step_output_file = os.environ["GITHUB_OUTPUT"]
  with open(step_output_file, "a") as f:
    f.writelines(f"{k}={v}" "\n" for k, v in d.items())


def write_job_summary(summary: str):
  """Write markdown messages on Github workflow UI.
  See https://docs.github.com/en/actions/using-workflows/workflow-commands-for-github-actions#adding-a-job-summary
  """
  step_summary_file = os.environ["GITHUB_STEP_SUMMARY"]
  with open(step_summary_file, "a") as f:
    # Use double newlines to split sections in markdown.
    f.write(summary + "\n\n")


def check_description_and_show_diff(original_description: str,
                                    current_description: str):
  if original_description == current_description:
    return

  diffs = difflib.unified_diff(original_description.splitlines(keepends=True),
                               current_description.splitlines(keepends=True))

  write_job_summary(
      textwrap.dedent("""\
  :pushpin: Using a PR description different from the original PR event \
  started this workflow.

  <details>
  <summary>Click to show diff (original vs. current)</summary>

  ```diff
  {}
  ```
  </details>""").format("".join(diffs)))


def get_trailers() -> Mapping[str, str]:
  title = os.environ["PR_TITLE"]
  body = os.environ.get("PR_BODY", "")
  original_title = os.environ.get("ORIGINAL_PR_TITLE")
  original_body = os.environ.get("ORIGINAL_PR_BODY", "")

  description = PR_DESCRIPTION_TEMPLATE.format(title=title, body=body)

  # PR_TITLE and PR_BODY can be fetched from API for the latest updates. If
  # ORIGINAL_PR_TITLE is set, compare the current and original description and
  # show a notice if they are different. This is mostly to inform users that the
  # workflow might not parse the PR description they expect.
  if original_title is not None:
    original_description = PR_DESCRIPTION_TEMPLATE.format(title=original_title,
                                                          body=original_body)
    print("Original PR description:", original_description, sep="\n")
    check_description_and_show_diff(original_description=original_description,
                                    current_description=description)

  print("Parsing PR description:", description, sep="\n")

  trailer_lines = subprocess.run(
      ["git", "interpret-trailers", "--parse", "--no-divider"],
      input=description,
      stdout=subprocess.PIPE,
      check=True,
      text=True,
      timeout=60).stdout.splitlines()
  return {
      k.lower().strip(): v.strip()
      for k, v in (line.split(":", maxsplit=1) for line in trailer_lines)
  }


def get_modified_paths(base_ref: str) -> Iterable[str]:
  return subprocess.run(["git", "diff", "--name-only", base_ref],
                        stdout=subprocess.PIPE,
                        check=True,
                        text=True,
                        timeout=60).stdout.splitlines()


def modifies_included_path(base_ref: str) -> bool:
  return any(not skip_path(p) for p in get_modified_paths(base_ref))


def should_run_ci(event_name, trailers) -> bool:
  if event_name != PULL_REQUEST_EVENT_NAME:
    print(f"Running CI independent of diff because run was not triggered by a"
          f" pull request event (event name is '{event_name}')")
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


def get_ci_stage(event_name):
  if event_name == PULL_REQUEST_EVENT_NAME:
    return "presubmit"
  elif event_name == PUSH_EVENT_NAME:
    return "postsubmit"
  elif event_name == SCHEDULE_EVENT_NAME:
    return "postsubmit"
  elif event_name == WORKFLOW_DISPATCH_EVENT_NAME:
    return "unknown"
  raise ValueError(f"Unrecognized event name '{event_name}'")


def get_benchmark_presets(ci_stage: str, trailers: Mapping[str, str]) -> str:
  """Parses and validates the benchmark presets from trailers.

  Args:
    trailers: trailers from PR description.

  Returns:
    A comma separated preset string, which later will be parsed by
    build_tools/benchmarks/export_benchmark_config.py.
  """
  if ci_stage == "postsubmit":
    preset_options = ["all"]
  else:
    trailer = trailers.get(BENCHMARK_PRESET_KEY)
    if trailer is None:
      return ""
    print(f"Using benchmark preset '{trailer}' from PR description trailers")
    preset_options = [option.strip() for option in trailer.split(",")]

  for preset_option in preset_options:
    if preset_option not in BENCHMARK_PRESET_OPTIONS:
      raise ValueError(f"Unknown benchmark preset option: '{preset_option}'.\n"
                       f"Available options: '{BENCHMARK_PRESET_OPTIONS}'.")

  if "all" in preset_options:
    preset_options = list(
        option for option in BENCHMARK_PRESET_OPTIONS if option != "all")

  return ",".join(preset_options)


def main():
  output: MutableMapping[str, str] = {}
  event_name = os.environ["GITHUB_EVENT_NAME"]
  trailers = get_trailers() if event_name == PULL_REQUEST_EVENT_NAME else {}
  if should_run_ci(event_name, trailers):
    output["should-run"] = "true"
  else:
    output["should-run"] = "false"
  output[RUNNER_ENV_KEY] = get_runner_env(trailers)
  ci_stage = get_ci_stage(event_name)
  output["ci-stage"] = ci_stage
  output["runner-group"] = ci_stage
  write_caches = "0"
  if ci_stage == "postsubmit":
    write_caches = "1"
  output["write-caches"] = write_caches
  output["benchmark-presets"] = get_benchmark_presets(ci_stage, trailers)

  set_output(output)


if __name__ == "__main__":
  main()
