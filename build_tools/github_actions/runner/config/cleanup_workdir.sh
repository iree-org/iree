#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Remove the working directory so we have a fresh version for subsequent
# actions.
# TODO: switch to ephemeral runners and get rid of this.
sudo rm -rf /home/runner/actions-runner/_work/iree/iree/
mkdir -p /home/runner/actions-runner/_work/iree/iree/
