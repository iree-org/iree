#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Updates the gievn pipelines with the Buildkite API.

This overwrites the YAML steps of all the Buildkite pipeline. Pipelines should
be updated this way, not in the UI. By default before updating, this checks if a
later job with the same step key has already run successfully, in which case it
skips the update. This is to avoid situations where an earlier build is for some
reason running a step after a later build due to race conditions or a retry.
Buildkite concurrency groups should be used to prevent builds from running this
check simultaneously.
"""

import argparse
import os
from pybuildkite import buildkite
import sys

PIPELINE_ROOT_PATH = "build_tools/buildkite/pipelines"


def parse_args():
  parser = argparse.ArgumentParser(
      description="Updates the YAML steps of a Buildkite pipeline.")
  parser.add_argument(
      "pipeline",
      help=(f"The slug of the pipeline to update. The steps are pulled from the"
            f" file at {os.path.join(PIPELINE_ROOT_PATH, '<pipeline>.yml')}"))
  return parser.parse_args()

def should_update(pipeline_to_update):
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

  organization = os.environ["BUILDKITE_ORGANIZATION_SLUG"]
  running_pipeline = os.environ["BUILDKITE_PIPELINE_SLUG"]
  build_number = int(os.environ["BUILDKITE_BUILD_NUMBER"])
  step_key = os.environ["BUILDKITE_STEP_KEY"]

  bk = buildkite.Buildkite()
  bk.set_access_token(access_token)

  next_build_number = build_number + 1
  try:
    build = bk.builds().get_build_by_number(organization, running_pipeline,
                                            next_build_number)
  except requests.exceptions.HTTPError as e:
    if e.response.status_code == 404:
      print(f"Did not find later run of '{pipeline_to_update}' with build number"
            f" '{next_build_number}'. Proceeding with update.")
      return True
    else:
      raise e


  jobs = build["jobs"]
  matching_job = next((j for j in jobs if j["step_key"] == step_key), None)
  if matching_job is None:
    return True

  # See https://buildkite.com/docs/pipelines/notifications#job-states
  # If the other job is "pending", "limiting", or "limited", it will be blocked
  # from racing against this one by the Buildkite concurrency control. If it's
  # in an active state, *this* build shouldn't be running due to Buildkite
  # concurrency control.
  # https://buildkite.com/docs/pipelines/controlling-concurrency#concurrency-groups
  # If it completed unsuccessfully then we assume for now (based on current use
  # cases) that it's ok for this build to proceed as if the other build was
  # never scheduled because presumably if it failed, it also failed to complete
  # the update, since that's the last thing it does.
  state = matching_job["state"]
  print(f"Found higher-numbered build ({next_build_number}) for pipeline"
        f" '{pipeline_to_update}' in state '{state}'")

  if state in ["passed"]:
    print("Later build passed. Skipping update.")
    return False

  if state in ["scheduled", "assigned", "accepted", "running"]:
    print("Later build is running at the same time as this one."
          " You have a concurrency bug.")
    sys.exit(2)

  return True


def main(args):
  if should_update(args.pipeline):
    pipeline_file = os.path.join(PIPELINE_ROOT_PATH, f"{args.pipeline}.yml")
    with open(pipeline_file) as f:
      current_pipeline_yaml = f.read()
    bk.pipelines().update_pipeline(organization=organization,
                                   pipeline=args.pipeline,
                                   configuration=current_pipeline_yaml)


if __name__ == "__main__":
  main(parse_args())
