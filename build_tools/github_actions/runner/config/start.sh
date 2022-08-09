#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Starts the GitHub actions runner as a custom service.

set -euo pipefail


echo "Starting runner"

/home/runner/actions-runner/bin/Runner.Listener run --startuptype service
