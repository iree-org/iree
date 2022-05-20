#!/usr/bin/env python3

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
import sys

ALLOWED_PIPELINES = ["presubmit", "postsubmit"]
ALLOWED_PLUGINS = [
    "github.com/GMNGeoffrey/smooth-checkout-buildkite-plugin#24e54e7729",
]


def main():
  # See https://buildkite.com/docs/agent/v3/hooks#agent-lifecycle-hooks
  buildkite_env_file = os.environ["BUILDKITE_ENV_FILE"]
  buildkite_env = {}
  with open(buildkite_env_file) as f:
    for line in f:
      key, val = line.split("=", maxsplit=1)
      buildkite_env[key] = val

  pipeline = buildkite_env["BUILDKITE_PIPELINE_SLUG"]
  if pipeline not in ALLOWED_PIPELINES:
    print(f"Pipeline '{pipeline}' is not allowed to run on this agent.",
          file=sys.stderr)
    sys.exit(2)

  plugins = json.loads(buildkite_env["BUILDKITE_PLUGINS"])

  for plugin in plugins:
    if plugin not in ALLOWED_PLUGINS:
      print(f"Plugin '{plugin}' is not allowed to run on this agent.",
            file=sys.stderr)
      sys.exit(2)


if __name__ == "__main__":
  main()
