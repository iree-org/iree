#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Installs and sets up the GitHub actions runner, creating services to start and
# tear down the runner.

set -xeEuo pipefail

# If the startup script fails, shut down the VM.
trap '/usr/sbin/shutdown -P now' ERR

SCRIPT_DIR="$(dirname -- "$( readlink -f -- "$0"; )")";
source "${SCRIPT_DIR}/functions.sh"

RUNNER_TYPE="$(get_attribute github-runner-type)"
# The CPU machines have 360GB of RAM
TMPFS_SIZE=100g
if [[ "${RUNNER_TYPE}" == gpu ]]; then
  # The GPU machines have only 85GB of RAM
  # TODO(gcmn): Switch to using a local ssd. This is probably too much of the
  # RAM.
  TMPFS_SIZE=50g
fi

echo "Creating tmpfs for runner"
mkdir /runner-root
mount -t tmpfs -o size="${TMPFS_SIZE}" tmpfs /runner-root
cp -r "${SCRIPT_DIR}" /runner-root/config
chown -R runner:runner /runner-root/

echo "Fetching the runner archive"
RUNNER_VERSION="$(get_attribute github-runner-version)"
RUNNER_ARCHIVE="actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
RUNNER_ARCHIVE_DIGEST="$(get_attribute github-runner-archive-digest)"

cd /runner-root
mkdir actions-runner
cd actions-runner
curl --silent --fail --show-error --location \
  "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${RUNNER_ARCHIVE}" \
  -o "${RUNNER_ARCHIVE}"

echo "${RUNNER_ARCHIVE_DIGEST} *${RUNNER_ARCHIVE}" | shasum -a 256 -c
tar xzf "${RUNNER_ARCHIVE}"
ln -s ../config/runner.env .env

echo "Registering the runner."
runuser --user runner /runner-root/config/register.sh

echo "Setting up the deregister service."
cp /runner-root/config/github-actions-runner-deregister.service /etc/systemd/system/

echo "Setting up the runner service."
cp /runner-root/config/github-actions-runner-start.service /etc/systemd/system/

echo "Reloading system service files to reflect changes."
systemctl daemon-reload

echo "Enabling the deregister service."
systemctl enable github-actions-runner-deregister

echo "Starting the runner service."
systemctl start github-actions-runner-start
