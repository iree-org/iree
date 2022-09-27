#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This script is put into the VM metadata as the startup script. It fetches the
# rest of the configuration from the repo at the specified commit.

set -xeuo pipefail

echo "Running setup script"

# Unfortunately this has to be duplicated from functions.sh because here we
# haven't fetched that file yet.
get_metadata() {
  local url="http://metadata.google.internal/computeMetadata/v1/${1}"
  ret=0
  curl "${url}" \
    --silent --fail --show-error \
    --header "Metadata-Flavor: Google" || ret=$?
  if [[ "${ret}" != 0 ]]; then
    echo "Failed fetching ${url}" >&2
    return "${ret}"
  fi
}

get_attribute() {
  get_metadata "instance/attributes/${1}"
}

REPO="$(get_attribute github-runner-config-repo)"
CONFIG_REF="$(get_attribute github-runner-config-ref)"

echo "Fetching from ${CONFIG_REF}"

cd /tmp/
rm -rf config
curl --silent --fail --show-error --location \
  "https://github.com/${REPO}/archive/${CONFIG_REF}.tar.gz" \
  | tar -zx -f - \
  --strip-components=4  --wildcards \
  */build_tools/github_actions/runner/config/

./config/setup.sh
