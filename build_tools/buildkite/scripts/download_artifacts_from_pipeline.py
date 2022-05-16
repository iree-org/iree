#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import json
import os
import re
from common.buildkite_pipeline_manager import BuildkitePipelineManager
from common.buildkite_utils import get_build_number

from pybuildkite import buildkite


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Downloads the artifacts from a Buildkite build.")
  parser.add_argument("--filter",
                      default=".*",
                      help="Filter on the paths of artifacts.")
  parser.add_argument("--download-dir",
                      default=".",
                      help="Directory path to store downloaded artifacts.")
  parser.add_argument("build_json",
                      help="JSON file of the target build object.")
  return parser.parse_args()


def main(args: argparse.Namespace):
  build = json.load(open(args.build_json, "r"))
  pipeline_name = build["pipeline"]["name"]
  build_number = get_build_number(build)

  bk = BuildkitePipelineManager.from_environ(pipeline_name)
  artifacts = bk.list_artifacts_for_build(build_number)

  downloaded_files = set()
  for artifact in artifacts:
    state = buildkite.BuildState(artifact["state"])
    if state != buildkite.BuildState.FINISHED:
      continue
    if not re.match(args.filter, artifact["path"]):
      continue

    output_filename = f"{artifact['sha1sum']}_{artifact['filename']}"
    if output_filename in downloaded_files:
      continue

    print(f"Downloading {output_filename}")
    stream = bk.download_artifact(build_number, artifact, as_stream=True)
    with open(os.path.join(args.download_dir, output_filename), "wb") as f:
      for chunk in stream:
        f.write(chunk)

    downloaded_files.add(output_filename)


if __name__ == "__main__":
  main(parse_args())
