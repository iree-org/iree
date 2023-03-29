#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build the IREE TF integration binaries. Despite the name, also runs the few
# lit tests for these that are enabled through Bazel. These take seconds to run
# all of them and it's easiest to just run them here rather than trying to
# figure out how to pass prebuilt binaries to Bazel tests.

set -xeuo pipefail

ROOT_DIR="${ROOT_DIR:-$(git rev-parse --show-toplevel)}"
IREE_USE_WORKSPACE_RC="${IREE_USE_WORKSPACE_RC:-0}"
IREE_READ_REMOTE_BAZEL_CACHE="${IREE_READ_REMOTE_BAZEL_CACHE:-1}"
IREE_WRITE_REMOTE_BAZEL_CACHE="${IREE_WRITE_REMOTE_BAZEL_CACHE:-0}"
IREE_TF_BINARIES_OUTPUT_DIR="${IREE_TF_BINARIES_OUTPUT_DIR:-}"
INTEGRATIONS_DIR="${ROOT_DIR}/integrations/tensorflow"

if (( ${IREE_WRITE_REMOTE_BAZEL_CACHE} == 1 && ${IREE_READ_REMOTE_BAZEL_CACHE} != 1 )); then
  echo "Can't have 'IREE_WRITE_REMOTE_BAZEL_CACHE' (${IREE_WRITE_REMOTE_BAZEL_CACHE}) set without 'IREE_READ_REMOTE_BAZEL_CACHE' (${IREE_READ_REMOTE_BAZEL_CACHE})"
fi

# We want to get back to wherever we were called from and output to the output
# directory relative to that.
pushd "${INTEGRATIONS_DIR}" > /dev/null

BAZEL_BIN=${BAZEL_BIN:-$(which bazel)}

BAZEL_STARTUP_CMD=("${BAZEL_BIN}")

if [[ "${IREE_USE_WORKSPACE_RC}" == 0 ]]; then
  BAZEL_STARTUP_CMD+=(--noworkspace_rc --bazelrc=build_tools/bazel/iree-tf.bazelrc)
fi

if [[ "${IREE_BAZEL_OUTPUT_BASE:-}" != "" ]]; then
  BAZEL_STARTUP_CMD+=("--output_base=${IREE_BAZEL_OUTPUT_BASE}")
  mkdir -p "${IREE_BAZEL_OUTPUT_BASE}"
fi

BAZEL_TEST_CMD=("${BAZEL_STARTUP_CMD[@]}" test)

if (( IREE_READ_REMOTE_BAZEL_CACHE == 1 )); then
  BAZEL_TEST_CMD+=(--config=remote_cache_bazel_tf_ci)
fi

if (( IREE_WRITE_REMOTE_BAZEL_CACHE != 1 )); then
  BAZEL_TEST_CMD+=(--noremote_upload_local_results)
fi

BAZEL_TEST_CMD+=(
  --config=generic_clang
  --test_tag_filters="-nokokoro"
  --build_tag_filters="-nokokoro"
)

# xargs is set to high arg limits to avoid multiple Bazel invocations and will
# hard fail if the limits are exceeded.
# See https://github.com/bazelbuild/bazel/issues/12479
"${BAZEL_STARTUP_CMD[@]}" query //iree_tf_compiler/... | \
   xargs --max-args 1000000 --max-chars 1000000 --exit \
    "${BAZEL_TEST_CMD[@]}"

popd > /dev/null

if [[ "${IREE_TF_BINARIES_OUTPUT_DIR}" != "" ]]; then
  mkdir -p "${IREE_TF_BINARIES_OUTPUT_DIR}"
  cp \
    "${INTEGRATIONS_DIR}/bazel-bin/iree_tf_compiler/iree-import-tflite" \
    "${INTEGRATIONS_DIR}/bazel-bin/iree_tf_compiler/iree-import-xla" \
    "${IREE_TF_BINARIES_OUTPUT_DIR}"
fi
