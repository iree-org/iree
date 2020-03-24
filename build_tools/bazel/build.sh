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

# Use user-environment variables if set, otherwise use CI-friendly defaults.
if ! [[ -v IREE_VULKAN_DISABLE ]]; then
  IREE_VULKAN_DISABLE=1
fi
declare -a test_env_args=(
  --test_env=IREE_VULKAN_DISABLE=$IREE_VULKAN_DISABLE
)

declare -a default_build_tag_filters=("-nokokoro")
declare -a default_test_tag_filters=("-nokokoro")

# We can still build things that use vulkan. Only add to test tag filters.
if [[ "${IREE_VULKAN_DISABLE?}" == 1 ]]; then
  default_test_tag_filters+=("-driver=vulkan")
fi
# Use user-environment variables if set, otherwise use CI-friendly defaults.
if ! [[ -v BUILD_TAG_FILTERS ]]; then
  # String join on comma
  BUILD_TAG_FILTERS="$(IFS="," ; echo "${default_build_tag_filters[*]?}")"
fi
if ! [[ -v TEST_TAG_FILTERS ]]; then
  # String join on comma
  TEST_TAG_FILTERS="$(IFS="," ; echo "${default_test_tag_filters[*]?}")"
fi

# Build and test everything in supported directories not explicitly marked as
# excluded from CI (using the tag "nokokoro").
# Note that somewhat contrary to its name `bazel test` will also build
# any non-test targets specified.
# We use `bazel query //...` piped to `bazel test` rather than the simpler
# `bazel test //...` because the latter excludes targets tagged "manual". The
# "manual" tag allows targets to be excluded from human wildcard builds, but we
# want them built by CI unless they are excluded with "nokokoro".
bazel query "//iree/... + //bindings/..." | \
  xargs bazel test ${test_env_args[@]} \
    --build_tag_filters="${BUILD_TAG_FILTERS?}" \
    --test_tag_filters="${TEST_TAG_FILTERS?}" \
    --keep_going \
    --test_output=errors \
    --config=rs

# Disable RBE until compatibility issues with the experimental_repo_remote_exec
# flag are fixed.
#   --config=rbe
