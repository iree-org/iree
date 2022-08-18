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

# Change this to a different git reference to fetch from somewhere else.
# For PRs, that would be refs/pull/<pr_number>/merge or for forks, you can
# change the repo. When deployed to the VM, the config reference is substituted
# with an explicit commit digest.
REPO="iree-org/iree"
CONFIG_REF=main

echo "Fetching from ${CONFIG_REF}"

cd /tmp/
rm -rf config
curl --silent --fail --show-error --location \
  "https://github.com/${REPO}/archive/${CONFIG_REF}.tar.gz" \
  | tar -zx -f - \
  --strip-components=4  --wildcards \
  */build_tools/github_actions/runner/config/

chown -R runner:runner config/
./config/setup.sh
