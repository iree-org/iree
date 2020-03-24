#!/bin/bash

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Build the IREE project with bazel. Designed for CI, but can be run manually.

set -e

set -x

# CI-friendly defaults that control availability of certain platform tests.
if ! [[ -v IREE_VULKAN_DISABLE ]]; then
  IREE_VULKAN_DISABLE=1
fi
test_env_args=(
  --test_env=IREE_VULKAN_DISABLE=$IREE_VULKAN_DISABLE
)
echo "Running with test env args: ${test_env_args[@]}"

# Build and test everything in supported directories not explicitly marked as
# excluded from CI (using the tag "nokokoro").
# Note that somewhat contrary to its name `bazel test` will also build
# any non-test targets specified.
# We use `bazel query //...` piped to `bazel test` rather than the simpler
# `bazel test //...` because the latter excludes targets tagged "manual". The
# "manual" tag allows targets to be excluded from human wildcard builds, but we
# want them built by CI unless they are excluded with "nokokoro".
bazel query '//iree/... + //bindings/... except attr("tags", "nokokoro", //...)' | \
  xargs bazel test ${test_env_args[@]} --keep_going --test_output=errors --config=rs

# Disable RBE until compatibility issues with the experimental_repo_remote_exec
# flag are fixed.
#   --config=rbe
