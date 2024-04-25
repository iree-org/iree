#!/bin/bash
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -e
set -x

ROOT_DIR=$(git rev-parse --show-toplevel)
cd ${ROOT_DIR?}

./samples/dynamic_shapes/test.sh
./samples/variables_and_state/test.sh
