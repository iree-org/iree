#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This script is actually part of the VM image and fetches the rest of the
# configuration. Longer term, we may want to have an explicit deployment of new
# scripts instead.

set -euo pipefail

runner_registration_token="${1}"

if [[ "$(whoami)" != "runner" ]]; then
  echo "Current user is not 'runner'. Rerunning script as 'runner'."
  SCRIPT="$( readlink -f -- "$0"; )"
  sudo su runner --shell /bin/bash --command "${SCRIPT} ${runner_registration_token}"
  exit
fi

cd "${HOME}"
shopt -s extglob
rm -rf -v !("actions-runner"|"fetch_and_setup.sh")
shopt -u extglob

cd /tmp/
rm -rf /tmp/iree
# TODO: replace with main repo
# git clone https://github.com/iree-org/iree.git
git clone https://github.com/gmngeoffrey/iree.git
cd iree
git fetch origin runner-setup
git checkout runner-setup
cd ..

cp -r iree/build_tools/github_actions/runner/* "${HOME}/"

cd "${HOME}"
rm -rf /tmp/iree

./setup.sh "${runner_registration_token}"
