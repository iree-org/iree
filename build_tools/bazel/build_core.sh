#!/bin/bash

# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build IREE's core workspace (excluding integrations/tensorflow/) with bazel.
# Designed for CI, but can be run manually.

# Looks at environment variables and uses CI-friendly defaults if they are not
# set.
# IREE_VULKAN_DISABLE: Do not run tests that require Vulkan. Default: 0
# BUILD_TAG_FILTERS: Passed to bazel to filter targets to build.
#   See https://docs.bazel.build/versions/master/command-line-reference.html#flag--build_tag_filters)
#   Default: "-nokokoro"
# TEST_TAG_FILTERS: Passed to bazel to filter targets to test. Note that test
#   targets excluded this way will also not be built.
#   See https://docs.bazel.build/versions/master/command-line-reference.html#flag--test_tag_filters)
#   Default: If IREE_VULKAN_DISABLE=1, "-nokokoro,-driver=vulkan". Else "-nokokoro".

set -xeuo pipefail


IREE_READ_REMOTE_BAZEL_CACHE="${IREE_READ_REMOTE_BAZEL_CACHE:-1}"
IREE_WRITE_REMOTE_BAZEL_CACHE="${IREE_WRITE_REMOTE_BAZEL_CACHE:-0}"
BAZEL_BIN="${BAZEL_BIN:-$(which bazel)}"

if (( ${IREE_WRITE_REMOTE_BAZEL_CACHE} == 1 && ${IREE_READ_REMOTE_BAZEL_CACHE} != 1 )); then
  echo "Can't have 'IREE_WRITE_REMOTE_BAZEL_CACHE' (${IREE_WRITE_REMOTE_BAZEL_CACHE}) set without 'IREE_READ_REMOTE_BAZEL_CACHE' (${IREE_READ_REMOTE_BAZEL_CACHE})"
fi

# Use user-environment variables if set, otherwise use CI-friendly defaults.
if ! [[ -v IREE_VULKAN_DISABLE ]]; then
  IREE_VULKAN_DISABLE=0
fi
if ! [[ -v IREE_NVIDIA_GPU_TESTS_DISABLE ]]; then
  IREE_NVIDIA_GPU_TESTS_DISABLE=1
fi

declare -a test_env_args=(
  --test_env="LD_PRELOAD=libvulkan.so.1"
  --test_env="VK_ICD_FILENAMES=${VK_ICD_FILENAMES}"
  --test_env=IREE_VULKAN_DISABLE="${IREE_VULKAN_DISABLE}"
  --test_env=IREE_NVIDIA_GPU_TESTS_DISABLE="${IREE_NVIDIA_GPU_TESTS_DISABLE}"
)

if ! [[ -n IREE_LLVM_SYSTEM_LINKER_PATH ]]; then
  test_env_args+=(--action_env=IREE_LLVM_SYSTEM_LINKER_PATH="${IREE_LLVM_SYSTEM_LINKER_PATH}")
fi

if ! [[ -n IREE_LLVM_EMBEDDED_LINKER_PATH ]]; then
  test_env_args+=(--action_env=IREE_LLVM_EMBEDDED_LINKER_PATH="${IREE_LLVM_EMBEDDED_LINKER_PATH}")
fi

declare -a default_build_tag_filters=("-nokokoro")
declare -a default_test_tag_filters=("-nokokoro" "-driver=metal")

# The VK_KHR_shader_float16_int8 extension is optional prior to Vulkan 1.2.
# We test on SwiftShader, which does not support this extension.
default_test_tag_filters+=("-vulkan_uses_vk_khr_shader_float16_int8")
# CUDA CI testing disabled until we setup a target for it.
default_test_tag_filters+=("-driver=cuda")

if [[ "${IREE_VULKAN_DISABLE?}" == 1 ]]; then
  default_test_tag_filters+=("-driver=vulkan")
fi
if [[ "${IREE_NVIDIA_GPU_TESTS_DISABLE?}" == 1 ]]; then
  default_test_tag_filters+=("-requires-gpu-nvidia")
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

# Build and test everything in the main workspace (not integrations).
# Note that somewhat contrary to its name `bazel test` will also build
# any non-test targets specified.
# We use `bazel query //...` piped to `bazel test` rather than the simpler
# `bazel test //...` because the latter excludes targets tagged "manual". The
# "manual" tag allows targets to be excluded from human wildcard builds, but we
# want them built by CI unless they are excluded with "nokokoro".
# Explicitly list bazelrc so that builds are reproducible and get cache hits
# when this script is invoked locally.
# xargs is set to high arg limits to avoid multiple Bazel invocations and will
# hard fail if the limits are exceeded.
# See https://github.com/bazelbuild/bazel/issues/12479

declare -a BAZEL_STARTUP_CMD=(
  "${BAZEL_BIN}"
  --noworkspace_rc
  --bazelrc=build_tools/bazel/iree.bazelrc
)

declare -a BAZEL_TEST_CMD=(
  "${BAZEL_STARTUP_CMD[@]}"
  test
)

if [[ ! -z "${SANDBOX_BASE}" ]]; then
  BAZEL_TEST_CMD+=(--sandbox_base="${SANDBOX_BASE}")
fi

if (( IREE_READ_REMOTE_BAZEL_CACHE == 1 )); then
  BAZEL_TEST_CMD+=(--config=remote_cache_bazel_ci)
fi

if (( IREE_WRITE_REMOTE_BAZEL_CACHE != 1 )); then
  BAZEL_TEST_CMD+=(--noremote_upload_local_results)
fi

BAZEL_TEST_CMD+=(
  --color=yes
  "${test_env_args[@]}"
  --build_tag_filters="${BUILD_TAG_FILTERS?}"
  --test_tag_filters="${TEST_TAG_FILTERS?}"
  --keep_going
  --test_output=errors
  --config=rs
  --config=generic_clang
)

"${BAZEL_STARTUP_CMD[@]}" query //... | \
  xargs --max-args 1000000 --max-chars 1000000 --exit \
    "${BAZEL_TEST_CMD[@]}"
