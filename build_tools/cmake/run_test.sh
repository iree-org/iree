#!/bin/bash

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# A wrapper around a test command that performs setup and teardown. This is
# appranetly not supported natively in ctest/cmake.

set -x
set -e

function cleanup() {
  echo "Cleaning up test environment"
  rm -rf ${TEST_TMPDIR?}
}

echo "Creating test environment"
rm -rf "${TEST_TMPDIR?}" # In case this wasn't cleaned up previously
mkdir -p "${TEST_TMPDIR?}"
trap cleanup EXIT
# Execute whatever we were passed.
"$@"
