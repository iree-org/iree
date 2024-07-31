#!/bin/bash

# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -xeuo pipefail

CPU_PLATFORM="${CPU_PLATFORM}"
GPU_PLATFORM="${GPU_PLATFORM}"
MACHINE_TYPE="${MACHINE_TYPE}"
RUNNER_NAME="${RUNNER_NAME}"
REGISTER_TOKEN="${REGISTER_TOKEN}"

HOSTNAME="$(hostname)"
RUNNER_SCOPE="${RUNNER_SCOPE:-openxla}"
RUNNER_TRUST="${RUNNER_TRUST:-minimal}"
RUNNER_GROUP="${RUNNER_GROUP:-presubmit}"
RUNNER_ENV="${RUNNER_ENV:-prod}"
OS_FAMILY="${OS_FAMILY:-ubuntu}"
RUNNER_DIR="${RUNNER_DIR:-actions-runner}"

sed -i "s/%RUNNER_GROUP%/${RUNNER_GROUP}/" "${RUNNER_DIR}/.env"

declare -a RUNNER_LABELS_ARRAY=(
  "os-family=${OS_FAMILY}"
  # Also as just a raw label, to match GitHub default behavior
  "${OS_FAMILY}"
  "hostname=${HOSTNAME}"
  "trust=${RUNNER_TRUST}"
  "runner-group=${RUNNER_GROUP}"
  "cpu-platform=${CPU_PLATFORM}"
  "gpu-platform=${GPU_PLATFORM}"
  "machine-type=${MACHINE_TYPE}"
  "environment=${RUNNER_ENV}"
)

RUNNER_LABELS="$(IFS="," ; echo "${RUNNER_LABELS_ARRAY[*]}")"

declare -a args=(
  --unattended \
  --url "https://github.com/${RUNNER_SCOPE}" \
  --name "${RUNNER_NAME}" \
  # If we end up with name conflicts, just replace the old entry.
  --replace \
  --runnergroup "${RUNNER_GROUP}" \
  --labels "${RUNNER_LABELS}"
)

"${RUNNER_DIR}/config.sh" --token "${REGISTER_TOKEN}" "${args[@]}"
