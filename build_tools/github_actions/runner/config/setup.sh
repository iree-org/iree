#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Installs and sets up the GitHub actions runner, creating services to start and
# tear down the runner.

set -xeuo pipefail

SCRIPT_DIR="$(dirname -- "$( readlink -f -- "$0"; )")";
source "${SCRIPT_DIR}/functions.sh"

mount -t tmpfs -o size=50g tmpfs /home/runner
cp -r "${SCRIPT_DIR}" /home/runner/config
chown -R runner:runner /home/runner/

RUNNER_VERSION="$(get_attribute github-runner-version)"
RUNNER_ARCHIVE="actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
RUNNER_ARCHIVE_DIGEST="$(get_attribute github-runner-archive-digest)"

echo "Fetching the runner archive"
cd /home/runner
mkdir actions-runner
cd actions-runner
curl --silent --fail --show-error --location \
  "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${RUNNER_ARCHIVE}" \
  -o "${RUNNER_ARCHIVE}"

echo "${RUNNER_ARCHIVE_DIGEST} *${RUNNER_ARCHIVE}" | shasum -a 256 -c
tar xzf "${RUNNER_ARCHIVE}"
ln -s ../config/runner.env .env

echo "Registering the runner."
runuser --user runner /home/runner/config/register.sh

echo "Setting up the deregister service."
cp /home/runner/config/github-actions-runner-deregister.service /etc/systemd/system/

echo "Setting up the runner service."
cp /home/runner/config/github-actions-runner-start.service /etc/systemd/system/

echo "Reloading system service files to reflect changes."
systemctl daemon-reload

echo "Enabling the deregister service."
systemctl enable github-actions-runner-deregister

echo "Starting the runner service."
systemctl start github-actions-runner-start
