#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Remove the working directory so we have a fresh version for subsequent
# actions. This isn't really necessary with ephemeral runners, but it also
# doesn't hurt and we will have some runners that aren't ephemeral.

rm -rf /runner-root/actions-runner/_work/iree/iree/
mkdir -p /runner-root/actions-runner/_work/iree/iree/
