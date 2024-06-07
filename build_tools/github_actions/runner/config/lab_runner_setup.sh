#!/bin/bash

# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -xeuo pipefail

GCLOUD_PLATFORM="${GCLOUD_CLI_PLATFORM:-linux-arm}"
GCLOUD_VERSION="${GCLOUD_CLI_VERSION:-414.0.0}"
GCLOUD_CHECKSUM="6f9186bc2b90b9140b3fbc8db121da71638b4a5c91c05c4f77e188bde20f692c"
GCLOUD_CRED_FILE="${GCLOUD_CRED_FILE:-service-account-key.json}"
RUNNER_PLATFORM="${RUNNER_PLATFORM:-linux-arm64}"
RUNNER_VERSION="${RUNNER_VERSION:-2.317.0}"
RUNNER_CHECKSUM="7e8e2095d2c30bbaa3d2ef03505622b883d9cb985add6596dbe2f234ece308f3"

# Install gcloud cli

curl -L --output "gcloud.tar.gz" \
  "https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-${GCLOUD_VERSION}-${GCLOUD_PLATFORM}.tar.gz"
echo "${GCLOUD_CHECKSUM} gcloud.tar.gz" | sha256sum --check --strict
tar -xf "gcloud.tar.gz"
rm "gcloud.tar.gz"
google-cloud-sdk/install.sh --quiet --path-update=true
# This setting is now enabled by default. It sounds great, but unfortunately
# doing such an upload requires *delete* permissions on the bucket, which we
# deliberately do not give runners. For the life of me, I could not figure out
# how to use `gcloud config set` (the "proper" way to set properties) to work
# on the global properties.
cat <<EOF >> "google-cloud-sdk/properties"
[storage]
parallel_composite_upload_enabled = False
EOF

source ~/.bashrc

gcloud auth login --cred-file="${GCLOUD_CRED_FILE}"
gcloud auth list
gcloud info

# Install action runner

mkdir "actions-runner"
pushd "actions-runner"
curl -L --output "runner.tar.gz" \
  "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-${RUNNER_PLATFORM}-${RUNNER_VERSION}.tar.gz"
echo "${RUNNER_CHECKSUM} runner.tar.gz" | sha256sum --check --strict
tar -xzf "runner.tar.gz"
rm "runner.tar.gz"
ln -s ../config/runner.env .env
popd
