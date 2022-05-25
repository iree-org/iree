#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Potentially triggers and then waits for a Buildkite pipeline.

Checks if the given pipeline is already running for the given commit, starts
it if not, and then waits for it to complete. Exits successfully if the
triggered (or pre-existing) build succeeds, otherwise fails.
"""

import argparse
import json
import sys
from typing import Optional

from pybuildkite import buildkite

from common.buildkite_pipeline_manager import BuildkitePipelineManager
from common.buildkite_utils import (BuildObject, get_build_number,
                                    get_build_state, linkify)


def should_create_new_build(bk: BuildkitePipelineManager,
                            build: Optional[BuildObject], rebuild_option: str):
  if not build:
    print("Didn't find previous build for pipeline. Creating a new one.")
    return True
  state = get_build_state(build)
  build_number = get_build_number(build)
  url = bk.get_url_for_build(build_number)
  print(f"Found previous build with state '{state.name}': {url}")

  if rebuild_option == "force":
    print(f"Received `--rebuild=force`, so creating a new build")
    return True
  elif rebuild_option == "failed" and state == buildkite.BuildState.FAILED:
    print(f"Previous build failed and received `--rebuild=failed`, so"
          f" creating a new one.")
    return True
  elif rebuild_option == "bad" and state in (
      buildkite.BuildState.FAILED,
      buildkite.BuildState.CANCELED,
      buildkite.BuildState.SKIPPED,
      buildkite.BuildState.NOT_RUN,
  ):
    print(f"Previous build completed with state '{state.name}' and received"
          f" `--rebuild=bad`, so creating a new one.")
    return True

  return False


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Waits on the status of the last Buildkite build for a given"
      " commit or creates such a build if none exists")
  parser.add_argument(
      "pipeline", help="The pipeline for which to create and wait for builds")
  parser.add_argument(
      "--rebuild",
      help="Behavior for triggering a new build even if there is an existing"
      " one. `force`: always rebuild without checking for existing build,"
      " `failed`: rebuild on build finished in 'failed' state, `bad`: rebuild"
      " on build finished in state other than 'passed'",
      choices=["force", "failed", "bad"],
  )
  parser.add_argument(
      "--output-build-json",
      help="Path to which to dump the JSON describing the finished build")
  return parser.parse_args()


def main(args):
  bk = BuildkitePipelineManager.from_environ(args.pipeline)
  build = bk.get_latest_build()

  if should_create_new_build(bk, build, args.rebuild):
    build = bk.create_build()
  build_number = get_build_number(build)
  url = bk.get_url_for_build(build_number)
  print(f"Waiting on {linkify(url)}")
  build = bk.wait_for_build(build_number)

  if args.output_build_json:
    with open(args.output_build_json, "w") as f:
      json.dump(build, f)

  if get_build_state(build) != buildkite.BuildState.PASSED:
    print(f"Build was not successful: {linkify(url)}")
    sys.exit(1)
  print(f"Build completed successfully: {linkify(url)}")


if __name__ == "__main__":
  main(parse_args())
