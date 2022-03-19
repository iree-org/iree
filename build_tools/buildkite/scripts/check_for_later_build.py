#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Checks if any later builds of the current pipeline exist.

Exits successfully if no such build exist. Exit code 2 indicates that such a
build exists. Exit code 1 indicates another error.
"""

import os
from pybuildkite import buildkite
import requests
import sys


def main():
  # A token for the Buildkite API. Needs read/write privileges on builds to
  # watch and create builds. Within our pipelines we fetch this from secret
  # manager: https://cloud.google.com/secret-manager. Users can create a
  # personal token for running this script locally:
  # https://buildkite.com/docs/apis/managing-api-tokens
  access_token = os.environ["BUILDKITE_ACCESS_TOKEN"]

  # Buildkite sets these environment variables. See
  # https://buildkite.com/docs/pipelines/environment-variables. If running
  # locally you can set locally or use the simulate_buildkite.sh script.
  organization = os.environ["BUILDKITE_ORGANIZATION_SLUG"]
  pipeline = os.environ["BUILDKITE_PIPELINE_SLUG"]
  build_number = int(os.environ["BUILDKITE_BUILD_NUMBER"])

  bk = buildkite.Buildkite()
  bk.set_access_token(access_token)

  try:
    bk.builds().get_build_by_number(organization, pipeline, build_number + 1)
  except requests.exceptions.HTTPError as e:
    if e.response.status_code == 404:
      sys.exit(0)
    raise e
  print(f"Found higher-numbered build for pipeline '{pipeline}'")
  sys.exit(2)


if __name__ == "__main__":
  main()
