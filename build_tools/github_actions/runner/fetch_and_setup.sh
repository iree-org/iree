#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This script is put into the VM metadata as the startup script. It fetches the 
# rest of the configuration from the repo. Longer term, we may want to have an
# explicit deployment of new scripts instead of fetching them directly from HEAD.
# Note that the startup script runs as root.

set -euo pipefail

echo "Running setup script"

# Change this to a different git reference to fetch from somewhere else.
# For PRs, that would be refs/pull/<pr_number>/merge
CONFIG_REF=main
 
echo "Fetching from ${CONFIG_REF}"

cd ~runner
rm -rf config
curl --silent --fail --show-error --location \
  "https://github.com/iree-org/iree/archive/${CONFIG_REF}.tar.gz" \
  | tar -zx -f - \
  --strip-components=4  --wildcards \
  */build_tools/github_actions/runner/config/

chown -R runner:runner config
~runner/config/setup.sh
