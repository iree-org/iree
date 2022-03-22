#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Checks if a later build of the current pipeline has completed successfully.

Exits successfully if no such build exist. Exit code 2 indicates that such a
build exists. Any other exit code indicates an error. A exit code of 3 indicates
a concurrency bug. The default exit code used by Python for uncaught exceptions
is 1.

This is to avoid situations where an earlier build is for some reason running a
step after a later build due to race conditions or a retry. Concurrency groups
should be used to prevent builds from running this check simultaneously.
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
  step_key = os.environ["BUILDKITE_STEP_KEY"]

  bk = buildkite.Buildkite()
  bk.set_access_token(access_token)

  next_build_number = build_number + 1

  try:
    build = bk.builds().get_build_by_number(organization, pipeline,
                                            next_build_number)
  except requests.exceptions.HTTPError as e:
    if e.response.status_code == 404:
      sys.exit(0)
    raise e
  jobs = build["jobs"]
  matching_job = next((j for j in jobs if j["key"] == step_key), None)
  if matching_job is None:
    sys.exit(0)

  # See https://buildkite.com/docs/pipelines/notifications#job-states
  # If the other job is "pending", "limiting", or "limited", it will be blocked
  # from racing against this one by the Buildkite concurrency control. If it's
  # in an active state, *this* build shouldn't be running due to Buildkite
  # concurrency control.
  # https://buildkite.com/docs/pipelines/controlling-concurrency#concurrency-groups
  # If it completed unsuccessfully then we assume for now (based on current use
  # cases) that it's ok for this build to proceed as if it was never scheduled.
  state = matching_job["state"]
  print(f"Found higher-numbered build ({next_build_number}) for pipeline"
        f" '{pipeline}' in state '{state}'")

  if state in ["passed"]:
    print("Later build passed. Exiting with code 2.")
    sys.exit(2)

  if state in ["scheduled", "assigned", "accepted", "running"]:
    print("Later build is running at the same time as this one."
          " You have a concurrency bug.")
    sys.exit(3)

  print("Later build did not finish. Exiting successfully.")


if __name__ == "__main__":
  main()
