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


# Change config-ref metadata to a different git reference to fetch from somewhere else.
# For PRs, that would be refs/pull/<pr_number>/merge
gihub_config_ref=$(curl --silent --fail "http://metadata.google.internal/computeMetadata/v1/instance/attributes/github-config-ref" -H "Metadata-Flavor: Google")

if [[ -z "${gihub_config_ref}" ]]; then
  echo "Using default 'main' config as no github-config-ref was specified."
  github_config_ref="main"
fi

cd /home/runner
rm -rf config/
mkdir config/
curl --silent --fail --show-error --location \
  "https://github.com/iree-org/iree/archive/${gihub_config_ref}.tar.gz" \
  | tar -zx -f - \
  --strip-components=4  --wildcards \
  */build_tools/github_actions/runner/config/

chown -R runner:runner /home/runner/config/
cd config

./start.sh