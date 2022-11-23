#!/usr/bin/env python3

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Determines whether CI should run on a given PR.

Exit code 0 indicates that it should and exit code 2 indicates that it should
not.
"""

import fnmatch
import os
import subprocess
import sys
from typing import Iterable, Mapping, MutableMapping

SKIP_CI_KEY = "skip-ci"
RUNNER_ENV_KEY = "runner-env"

# Note that these are fnmatch patterns, which are not the same as gitignore
# patterns because they don't treat '/' specially. The standard library doesn't
# contain a function for gitignore style "wildmatch". There's a third-party
# library pathspec (https://pypi.org/project/pathspec/), but it doesn't seem
# worth the dependency.
SKIP_PATH_PATTERNS = [
    "docs/*",
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


def skip_path(path: str) -> bool:
  return any(fnmatch.fnmatch(path, pattern) for pattern in SKIP_PATH_PATTERNS)


def set_output(d: Mapping[str, str]):
  print(f"Setting outputs: {d}")
  for k, v in d.items():
    print(f"::set-output name={k}::{v}")


def get_trailers() -> Mapping[str, str]:
  title = os.environ["PR_TITLE"]
  body = os.environ.get("PR_BODY", "")

  description = f"{title}" "\n\n" f"{body}"

  print("Parsing PR description:", description, sep="\n")

  trailer_lines = subprocess.run(["git", "interpret-trailers", "--parse"],
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
  base_ref = os.environ["BASE_REF"]

  if event_name != "pull_request":
    print(f"Running CI independent of diff because run was not triggered by a"
          f" pull request event (event name is '{event_name}')")
    return True

  if SKIP_CI_KEY in trailers:
    print(f"Not running CI because PR description has '{SKIP_CI_KEY}' trailer.")
    return False

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
  return 'presubmit' if event_name == 'pull_request' else 'postsubmit'


def main():
  output: MutableMapping[str, str] = {}
  trailers = get_trailers()
  event_name = os.environ["GITHUB_EVENT_NAME"]
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

  set_output(output)


if __name__ == "__main__":
  main()
