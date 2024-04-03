#!/bin/bash

# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build IREE's core workspace with Bazel and runs tests.
# Designed for CI, but can be run manually.
#
# Looks at environment variables and uses CI-friendly defaults for unset vars.
#
# GPU / hardware-dependent tests can be enabled by setting any of these to 0:
#   * IREE_CUDA_DISABLE
#   * IREE_HIP_DISABLE
#   * IREE_METAL_DISABLE
#   * IREE_VULKAN_DISABLE
#   * IREE_NVIDIA_GPU_TESTS_DISABLE
#   * IREE_AMD_RDNA3_GPU_TESTS_DISABLE
#
# Freeform filters can be appended using:
#   * BUILD_TAG_FILTERS: Passed to bazel to filter targets to build:
#     https://bazel.build/reference/command-line-reference.html#flag--build_tag_filters)
#   * TEST_TAG_FILTERS: Passed to bazel to filter targets to test. Note that
#     test targets excluded this way will also not be built:
#     https://bazel.build/reference/command-line-reference.html#flag--test_tag_filters)

set -xeuo pipefail

IREE_READ_REMOTE_BAZEL_CACHE="${IREE_READ_REMOTE_BAZEL_CACHE:-1}"
IREE_WRITE_REMOTE_BAZEL_CACHE="${IREE_WRITE_REMOTE_BAZEL_CACHE:-0}"
BAZEL_BIN="${BAZEL_BIN:-$(which bazel)}"
SANDBOX_BASE="${SANDBOX_BASE:-}"

if (( ${IREE_WRITE_REMOTE_BAZEL_CACHE} == 1 && ${IREE_READ_REMOTE_BAZEL_CACHE} != 1 )); then
  echo "Can't have 'IREE_WRITE_REMOTE_BAZEL_CACHE' (${IREE_WRITE_REMOTE_BAZEL_CACHE}) set without 'IREE_READ_REMOTE_BAZEL_CACHE' (${IREE_READ_REMOTE_BAZEL_CACHE})"
fi

# Use user-environment variables if set, otherwise use CI-friendly defaults.
if ! [[ -v IREE_CUDA_DISABLE ]]; then
  IREE_CUDA_DISABLE=1
fi
if ! [[ -v IREE_HIP_DISABLE ]]; then
  IREE_HIP_DISABLE=1
fi
if ! [[ -v IREE_METAL_DISABLE ]]; then
  IREE_METAL_DISABLE=1
fi
if ! [[ -v IREE_VULKAN_DISABLE ]]; then
  IREE_VULKAN_DISABLE=1
fi
if ! [[ -v IREE_NVIDIA_GPU_TESTS_DISABLE ]]; then
  IREE_NVIDIA_GPU_TESTS_DISABLE=1
fi
if ! [[ -v IREE_AMD_RDNA3_GPU_TESTS_DISABLE ]]; then
  IREE_AMD_RDNA3_GPU_TESTS_DISABLE=1
fi

declare -a test_env_args=(
  --test_env="LD_PRELOAD=libvulkan.so.1"
  --test_env=IREE_CUDA_DISABLE="${IREE_CUDA_DISABLE}"
  --test_env=IREE_HIP_DISABLE="${IREE_HIP_DISABLE}"
  --test_env=IREE_METAL_DISABLE="${IREE_METAL_DISABLE}"
  --test_env=IREE_VULKAN_DISABLE="${IREE_VULKAN_DISABLE}"
  --test_env=IREE_NVIDIA_GPU_TESTS_DISABLE="${IREE_NVIDIA_GPU_TESTS_DISABLE}"
  --test_env=IREE_AMD_RDNA3_GPU_TESTS_DISABLE="${IREE_AMD_RDNA3_GPU_TESTS_DISABLE}"
)

if ! [[ -n IREE_LLVM_SYSTEM_LINKER_PATH ]]; then
  test_env_args+=(--action_env=IREE_LLVM_SYSTEM_LINKER_PATH="${IREE_LLVM_SYSTEM_LINKER_PATH}")
fi

if ! [[ -n IREE_LLVM_EMBEDDED_LINKER_PATH ]]; then
  test_env_args+=(--action_env=IREE_LLVM_EMBEDDED_LINKER_PATH="${IREE_LLVM_EMBEDDED_LINKER_PATH}")
fi

declare -a default_build_tag_filters=("-nodocker")
declare -a default_test_tag_filters=("-nodocker")

if (( IREE_CUDA_DISABLE == 1 )); then
  default_test_tag_filters+=("-driver=cuda")
fi
if (( IREE_HIP_DISABLE == 1 )); then
  default_test_tag_filters+=("-driver=hip")
fi
if (( IREE_METAL_DISABLE == 1 )); then
  default_test_tag_filters+=("-driver=metal")
fi
if (( IREE_VULKAN_DISABLE == 1 )); then
  default_test_tag_filters+=("-driver=vulkan")
fi
if (( IREE_NVIDIA_GPU_TESTS_DISABLE == 1 )); then
  default_test_tag_filters+=("-requires-gpu-nvidia" "-requires-gpu-sm80")
fi
if (( IREE_AMD_RDNA3_GPU_TESTS_DISABLE == 1 )); then
  default_test_tag_filters+=("-requires-gpu-rdna3")
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
#
# Note that somewhat contrary to its name `bazel test` will also build
# any non-test targets specified.
# We use `bazel query //...` piped to `bazel test` rather than the simpler
# `bazel test //...` because the latter excludes targets tagged "manual". The
# "manual" tag allows targets to be excluded from human wildcard builds, but we
# want them built by CI unless they are excluded with tags.
#
# Explicitly list bazelrc so that builds are reproducible and get cache hits
# when this script is invoked locally.
#
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
