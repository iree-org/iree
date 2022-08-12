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

SKIP_CI_LABEL = "skip-ci"

# Note that these are fnmatch patterns, which are not the same as gitignore
# patterns because they don't treat '/' specially. The standard library doesn't
# contain a function for gitignore style "wildmatch". There's a third-party
# library pathspec (https://pypi.org/project/pathspec/), but it doesn't seem
# worth the dependency.
SKIP_PATH_PATTERNS = [
    "docs/*",
    "experimental/*",
    "build_tools/kokoro/*",
    "build_tools/buildkite/*",
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


def skip_path(path):
  return any(fnmatch.fnmatch(path, pattern) for pattern in SKIP_PATH_PATTERNS)


def get_modified_paths(base_ref):
  return subprocess.run(["git", "diff", "--name-only", base_ref],
                        stdout=subprocess.PIPE,
                        check=True,
                        text=True,
                        timeout=60).stdout.splitlines()


def modifies_included_path(base_ref):
  return any(not skip_path(p) for p in get_modified_paths(base_ref))


def should_run_ci():
  event_name = os.environ["GITHUB_EVENT_NAME"]
  base_ref = os.environ["BASE_REF"]
  labels = os.environ["LABELS"].split(",")

  if event_name != "pull_request":
    print("Running CI independent of diff because run was not triggered by a"
          "pull request event.")
    return True

  if SKIP_CI_LABEL in labels:
    print(f"Not running CI because PR has label '{SKIP_CI_LABEL}'.")
    return False

  try:
    modifies = modifies_included_path(base_ref)
  except TimeoutError as e:
    print("Computing modified files timed out. Running the CI")
    return True

  if not modifies:
    print("Skipping CI because all modified files are marked as excluded.")
    return False

  return True


def main():
  if should_run_ci():
    print("CI should run")
    sys.exit(0)
  print("CI should not run")
  sys.exit(2)


if __name__ == "__main__":
  main()
