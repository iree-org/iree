#!/bin/bash

# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

GCLOUD_PLATFORM="${GCLOUD_CLI_PLATFORM:-linux-arm}"
GCLOUD_VERSION="${GCLOUD_CLI_VERSION:-414.0.0}"
GCLOUD_CHECKSUM="e0382917353272655959bb650643c5df72c85de326a720b97e562bb6ea4478b1"
GCLOUD_CRED_FILE="${GCLOUD_CRED_FILE:-service_account_key.json}
RUNNER_PLATFORM="${RUNNER_PLATFORM:-linux-arm64}"
RUNNER_VERSION="${RUNNER_VERSION:-2.304.0}"
RUNNER_CHECKSUM="34c49bd0e294abce6e4a073627ed60dc2f31eee970c13d389b704697724b31c6"

# Install gcloud cli

curl -L --output "gcloud.tar.gz" \
  "https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-${GCLOUD_PLATFORM}-${GCLOUD_PLATFORM}.tar.gz"
echo "${GCLOUD_CHECKSUM} gcloud.tar.gz" | sha256sum --check --strict
tar -xf "gcloud.tar.gz"
rm "gcloud.tar.gz"
google-cloud-sdk/install.sh --quiet
# This setting is now enabled by default. It sounds great, but unfortunately
# doing such an upload requires *delete* permissions on the bucket, which we
# deliberately do not give runners. For the life of me, I could not figure out
# how to use `gcloud config set` (the "proper" way to set properties) to work
# on the global properties.
cat <<EOF >> "google-cloud-sdk/properties"
[storage]
parallel_composite_upload_enabled = False
EOF

gcloud auth login --cred_file="${GCLOUD_CRED_FILE}"
gcloud info

# Install action runner

mkdir "actions-runner"
pushd "actions-runner"
curl -L --output "runner.tar.gz" \
  "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-${RUNNER_PLATFORM}-{RUNNER_VERSION}.tar.gz"
echo "${RUNNER_CHECKSUM} runner.tar.gz" | sha256sum --check --strict
tar -xzf "runner.tar.gz"
rm "runner.tar.gz"
popd
