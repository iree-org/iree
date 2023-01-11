#!/bin/bash

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Runs the liburing configure script to update compat headers here.
# At the time of this writing, this facility is very simple and ok to just
# snapshot (which will need to be done for cross-compilation anyway). If this
# ever changes, something more exotic than a manual update will need to be
# done.

this_dir="$(cd $(dirname $0) && pwd)"
liburing_dir="$this_dir/../../../third_party/liburing"

if ! [ -d "$liburing_dir" ]; then
  echo "ERROR: Could not find directory $liburing_dir"
  exit 1
fi

# The configure script outputs files into the current directory and a
# src/include/liburing directory, matching the source tree.
config_dir="$this_dir/default_config"
mkdir -p "$config_dir/src/include/liburing"
cd "$config_dir"

if ! bash "$liburing_dir/configure"; then
  echo "ERROR: Could not configure"
  exit 2
fi
