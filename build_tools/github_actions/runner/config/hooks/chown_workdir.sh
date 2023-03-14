#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

# Docker has a tendency to make things owned by root unless you do a dance with
# which user you run it as. This can make the workspace unusable. This isn't
# really necessary with ephemeral runners, but it also doesn't hurt and we may
# have some runners that aren't ephemeral
sudo chown -R runner:runner /runner-root
