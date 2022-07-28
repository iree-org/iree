#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -xeuo pipefail

# Update submodules with depth 1 and parallelism. Lots of built-in ways of
# fetching submodules do it with full history and/or no parallelism. This takes
# a significant amount of time on the CI. Intended for CI. May be run locally,
# but it is not advised to do so in your development repository.


git submodule sync
git submodule update --init --force --depth=1 --jobs=8
