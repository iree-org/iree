#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -xeuo pipefail

# Docker has a tendency to make things owned by root unless you do a dance with
# which user you run it as. This can make the workspace unusable.
# TODO: switch to ephemeral runners and get rid of this.
sudo chown -R runner:runner /home/runner
