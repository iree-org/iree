#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Deregisters the GitHub actions runner using proxy token.

set -euo pipefail

SCRIPT_DIR="$(dirname -- "$( readlink -f -- "$0"; )")";
source "${SCRIPT_DIR}/functions.sh"

TOKEN_PROXY_URL="$(get_attribute github-token-proxy-url)"
RUNNER_SCOPE="$(get_attribute github-runner-scope)"
DEREGISTER_TOKEN="$(get_token remove ${RUNNER_SCOPE})"

if [ -z "${DEREGISTER_TOKEN}" ]; then
  echo "failed to get remove runner token" >&2
  exit 1
fi

echo "removing github actions runner"

/home/runner/actions-runner/config.sh remove --token "${DEREGISTER_TOKEN}"
