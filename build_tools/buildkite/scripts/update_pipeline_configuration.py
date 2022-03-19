#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Updates the YAML steps of a Buildkite pipeline."""

import argparse
import os
from pybuildkite import buildkite
import sys


def parse_args():
  parser = argparse.ArgumentParser(
      description="Updates the YAML steps of a Buildkite pipeline.")
  parser.add_argument("pipeline", help="The slug of the pipeline to update")
  parser.add_argument("--file",
                      help="File to read from. Default stdin",
                      type=argparse.FileType("r", encoding="UTF-8"),
                      default=sys.stdin)
  return parser.parse_args()


def main(args):
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

  bk = buildkite.Buildkite()
  bk.set_access_token(access_token)

  current_pipeline_yaml = args.file.read()
  bk.pipelines().update_pipeline(configuration=current_pipeline_yaml,
                                 pipeline=args.pipeline,
                                 organization=organization)


if __name__ == "__main__":
  main(parse_args())
