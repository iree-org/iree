#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Starts the GitHub actions runner in service mode.
# We don't use runsvc.sh here because it's a shell wrapper around a
# nodejs listener that calls the .NET runner program 😱 in startup mode
# and at the exit codes so it can retry if the runner is busy updating.
# We disable updating, so can avoid those layers of cruft.

set -euo pipefail

echo "Starting runner"

/runner-root/actions-runner/bin/Runner.Listener run --startuptype service
