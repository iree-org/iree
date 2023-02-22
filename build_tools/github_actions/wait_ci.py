#!/usr/bin/env python3

# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""TODO"""

import os
import requests

GITHUB_IREE_API_PREFIX = "https://api.github.com/repos/openxla/iree"
GITHUB_API_VERSION = "2022-11-28"

NEEDS_CI_JOB = {"build_all", "build_e2e_test_artifacts"}


def main():
  github_token = os.environ["GITHUB_TOKEN"]
  commit_sha = os.environ["GITHUB_SHA"]

  session = requests.session()
  api_headers = {
      "Accept": "application/vnd.github+json",
      "Authorization": f"token {github_token}",
      "X-GitHub-Api-Version": GITHUB_API_VERSION,
  }
  resp = session.get(
      f"{GITHUB_IREE_API_PREFIX}/commits/{commit_sha}/check-runs",
      headers=api_headers)
  needs_runs = [
      run for run in resp.json()["check_runs"] if run["name"] in NEEDS_CI_JOB
  ]
  print(needs_runs)
  print("Done")


if __name__ == "__main__":
  main()
