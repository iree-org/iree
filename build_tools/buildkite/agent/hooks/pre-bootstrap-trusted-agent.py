#!/usr/bin/env python3

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
import sys

# An external install because environment file parsing apparently doesn't exist
# in a built-in package.
from dotenv import dotenv_values

ALLOWED_PIPELINES = ["presubmit", "postsubmit"]
ALLOWED_PLUGINS = [
    "https://github.com/GMNGeoffrey/smooth-checkout-buildkite-plugin#24e54e7729",
]


def main():
  # See https://buildkite.com/docs/agent/v3/hooks#agent-lifecycle-hooks
  buildkite_env_file = os.environ["BUILDKITE_ENV_FILE"]
  buildkite_env = {}

  buildkite_env = dotenv_values(buildkite_env_file)

  print("Buildkite environment:", file=sys.stderr)
  for k, v in buildkite_env.items():
    print(f"{k}: {v}", file=sys.stderr)

  pipeline = buildkite_env["BUILDKITE_PIPELINE_SLUG"]
  if pipeline not in ALLOWED_PIPELINES:
    print(f"Pipeline '{pipeline}' is not allowed to run on this agent.",
          file=sys.stderr)
    sys.exit(2)

  plugins_var = buildkite_env.get("BUILDKITE_PLUGINS")
  if plugins_var is None:
    return

  plugins = json.loads(buildkite_env["BUILDKITE_PLUGINS"])

  for plugin in plugins:
    plugin_keys = list(plugin.keys())
    if len(plugin_keys) != 1:
      print(f"Got plugin in unexpected format: '{plugin}'", file=sys.stderr)
      sys.exit(3)
    plugin_key = plugin_keys[0]
    if plugin_key not in ALLOWED_PLUGINS:
      print(
          f"Plugin with key '{plugin_key}' is not allowed to run on this agent:"
          f" '{plugin}'",
          file=sys.stderr)
      sys.exit(2)


if __name__ == "__main__":
  main()
