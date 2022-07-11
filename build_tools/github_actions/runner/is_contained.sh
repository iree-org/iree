#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# A helper function for checking presence in an array.
# This file should be sourced, not executed.

set -euo pipefail

isContained () {
  local e;
  local match="$1"
  shift
  # for loop with no paramters iterates over arguments
  for e; do
    if [[ "$e" == "$match" ]]; then
      return 0
    fi
  done
  return 1
}
