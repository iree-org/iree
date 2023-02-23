#!/bin/bash
# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

PR_JSON="${1:-${IREE_PR_JSON}}"
GITHUB_OUTPUT="${2:-${GITHUB_OUTPUT}}"

TRAILER_PRESETS=$(jq --raw-output '.body' "${PR_JSON}" | \
  git interpret-trailers --parse --no-divider | \
  jq --raw-input 'match("^benchmarks:.+$", "g").string | [ scan("[-\\w]+") ][1:]')

echo "Presets in trailers: ${TRAILER_PRESETS}"

LABELS_PRESETS=$(jq --raw-output \
  '.labels | map(.name) | map(select(. | startswith("wip-benchmarks:")) | split(":")[1])' \
  "${PR_JSON}")

echo "Presets in labels: ${LABELS_PRESETS}"

BENCHMARK_PRESETS=$(echo "${TRAILER_PRESETS}${LABELS_PRESETS}" | \
  jq --slurp --raw-output '. | flatten | unique | join(",")')

echo "benchmark-presets=${BENCHMARK_PRESETS}" >> "${GITHUB_OUTPUT}"
