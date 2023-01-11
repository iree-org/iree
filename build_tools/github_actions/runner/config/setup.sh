#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Installs and sets up the GitHub actions runner, creating services to start and
# tear down the runner.

set -xeEuo pipefail

SCRIPT_DIR="$(dirname -- "$( readlink -f -- "$0"; )")";
source "${SCRIPT_DIR}/functions.sh"

mkdir /runner-root

RUNNER_TYPE="$(get_attribute github-runner-type)"

# On the CPU machines we use a ramdisk because they have 360GB of RAM, but the
# GPU machines only have 80GB. We don't use local SSD on the CPU machines
# because they require a minimum of 16 local SSD if any, which is quite wasteful
# (increases cost by about 20%).
if [[ "${RUNNER_TYPE}" == gpu ]]; then
  echo "Formatting and mounting local SSD for working directory"
  mkfs.ext4 -F /dev/nvme0n1
  # Options suggested from https://cloud.google.com/compute/docs/disks/optimizing-local-ssd-performance#disable_flush
  mount --options discard,defaults,nobarrier /dev/nvme0n1 /runner-root
else
  echo "Mounting tmpfs for working directory"
  mount --types tmpfs --options size=100g tmpfs /runner-root
fi

cp -r "${SCRIPT_DIR}" /runner-root/config
chown -R runner:runner /runner-root/

echo "Installing ops agent and turning on systemd logging"
# TODO(gcmn): This should probably be baked into the image.
curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
chmod +x add-google-cloud-ops-agent-repo.sh
./add-google-cloud-ops-agent-repo.sh --also-install
cp /runner-root/config/google-cloud-ops-agent/config.yaml /etc/google-cloud-ops-agent/config.yaml
service google-cloud-ops-agent restart

echo "Fetching the runner archive"
RUNNER_VERSION="$(get_attribute github-runner-version)"
RUNNER_ARCHIVE="actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
RUNNER_ARCHIVE_DIGEST="$(get_attribute github-runner-archive-digest)"

cd /runner-root
mkdir actions-runner
cd actions-runner
nice_curl \
  "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${RUNNER_ARCHIVE}" \
  -o "${RUNNER_ARCHIVE}"

echo "${RUNNER_ARCHIVE_DIGEST} *${RUNNER_ARCHIVE}" | shasum -a 256 -c
tar xzf "${RUNNER_ARCHIVE}"
ln -s ../config/runner.env .env

echo "Registering the runner."
runuser --user runner /runner-root/config/register.sh

echo "Loading systemd services"
cp /runner-root/config/systemd/system/* /etc/systemd/system/
systemctl daemon-reload

echo "Enabling systemd services."
find /runner-root/config/systemd/system/ -type f -printf "%f\n" \
  | xargs systemctl enable

echo "Starting the runner services"
systemctl start runner-setup.target
