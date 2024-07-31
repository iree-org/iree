#!/bin/bash

# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -xeuo pipefail

SCRIPT_DIR="$(dirname -- "$( readlink -f -- "$0"; )")";
GCLOUD_CRED_FILE="$(realpath "${GCLOUD_CRED_FILE}")"
RUNNER_ROOT="${RUNNER_ROOT:-/runner-root}"

sudo apt-get install jq

sudo adduser --system --group --home "${RUNNER_ROOT}" runner

sudo cp -r "${SCRIPT_DIR}" "${RUNNER_ROOT}/config"
sudo chown -R runner:runner "${RUNNER_ROOT}/config"

sudo chown runner:runner "${GCLOUD_CRED_FILE}"

pushd "${RUNNER_ROOT}"
sudo -u runner bash -c \
  "GCLOUD_CRED_FILE=${GCLOUD_CRED_FILE} ./config/lab_runner_setup.sh"
popd

sudo ln -s "${RUNNER_ROOT}/actions-runner/_work/iree/iree" "/work"
