#!/bin/bash

# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build IREE's core (//iree/... and //build_tools/...) with bazel.
# Designed for CI, but can be run manually.

# Looks at environment variables and uses CI-friendly defaults if they are not
# set.
# IREE_LLVMAOT_DISABLE: Do not run tests that require LLVM-AOT. Default: 0
# IREE_VULKAN_DISABLE: Do not run tests that require Vulkan. Default: 1
# BUILD_TAG_FILTERS: Passed to bazel to filter targets to build.
#   See https://docs.bazel.build/versions/master/command-line-reference.html#flag--build_tag_filters)
#   Default: "-nokokoro"
# TEST_TAG_FILTERS: Passed to bazel to filter targets to test. Note that test
#   targets excluded this way will also not be built.
#   See https://docs.bazel.build/versions/master/command-line-reference.html#flag--test_tag_filters)
#   Default: If IREE_VULKAN_DISABLE=1, "-nokokoro,-driver=vulkan". Else "-nokokoro".

set -e
set -x

# Use user-environment variables if set, otherwise use CI-friendly defaults.
if ! [[ -v IREE_LLVMAOT_DISABLE ]]; then
  IREE_LLVMAOT_DISABLE=0
fi
if ! [[ -v IREE_VULKAN_DISABLE ]]; then
  IREE_VULKAN_DISABLE=0
fi
declare -a test_env_args=(
  --test_env=IREE_LLVMAOT_DISABLE=$IREE_LLVMAOT_DISABLE
  --test_env=IREE_VULKAN_DISABLE=$IREE_VULKAN_DISABLE
  --action_env=IREE_LLVMAOT_LINKER_PATH=$IREE_LLVMAOT_LINKER_PATH
)

declare -a default_build_tag_filters=("-nokokoro")
declare -a default_test_tag_filters=("-nokokoro" "-driver=metal")

# The VK_KHR_shader_float16_int8 extension is optional prior to Vulkan 1.2.
# We test on SwiftShader, which does not support this extension.
default_test_tag_filters+=("-vulkan_uses_vk_khr_shader_float16_int8")
# CUDA CI testing disabled until we setup a target for it.
default_test_tag_filters+=("-uses_cuda_runtime")
default_test_tag_filters+=("-driver=cuda")

if [[ "${IREE_VULKAN_DISABLE?}" == 1 ]]; then
  default_test_tag_filters+=("-driver=vulkan")
fi
if [[ "${IREE_LLVMAOT_DISABLE?}" == 1 ]]; then
  default_test_tag_filters+=("-driver=dylib")
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

# Build and test everything in supported directories not excluded by the tag
# filters.
# Note that somewhat contrary to its name `bazel test` will also build
# any non-test targets specified.
# We use `bazel query //...` piped to `bazel test` rather than the simpler
# `bazel test //...` because the latter excludes targets tagged "manual". The
# "manual" tag allows targets to be excluded from human wildcard builds, but we
# want them built by CI unless they are excluded with "nokokoro".
# Explicitly list bazelrc so that builds are reproducible and get cache hits
# when this script is invoked locally.
bazel \
  --nosystem_rc --nohome_rc --noworkspace_rc \
  --bazelrc=build_tools/bazel/iree.bazelrc \
  query \
    --config=non_darwin \
    //iree/... + //build_tools/... | \
      xargs bazel \
        --nosystem_rc --nohome_rc --noworkspace_rc \
        --bazelrc=build_tools/bazel/iree.bazelrc \
          test \
            --color=yes \
            ${test_env_args[@]} \
            --config=generic_clang \
            --config=non_darwin \
            --build_tag_filters="${BUILD_TAG_FILTERS?}" \
            --test_tag_filters="${TEST_TAG_FILTERS?}" \
            --keep_going \
            --test_output=errors \
            --config=rs \
            --config=rbe
