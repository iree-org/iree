#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This script is actually part of the VM image and fetches the rest of the
# configuration. It is invoked on startup through a one-line startup script that
# calls it. Longer term, we may want to have an explicit deployment of new
# scripts instead of fetching them directly from HEAD.

set -euo pipefail

SCRIPT="$( readlink -f -- "$0"; )"
SCRIPT_BASENAME="$(basename ${SCRIPT})"

if [[ "$(whoami)" != "runner" ]]; then
  echo "Current user is not 'runner'. Rerunning script as 'runner'."
  sudo su runner --shell /bin/bash --command "${SCRIPT}"
  exit
fi

GITHUB_CLONE_TARGET=$(curl --silent --fail --show-error "http://metadata.google.internal/computeMetadata/v1/instance/attributes/github-clone-target" -H "Metadata-Flavor: Google")
 
cd "${HOME}"
rm -rf config

rm -rf /tmp/iree
git clone "${GITHUB_CLONE_TARGET}" /tmp/iree

cp -r /tmp/iree/build_tools/github_actions/runner/config/ "${HOME}/config"

rm -rf /tmp/iree

cd "${HOME}/config"

./setup.sh