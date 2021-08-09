#!/bin/bash
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Sets up launch scripts for containers.
# Typical usage: ./setup_dockcross.sh manylinux2014-x64

set -eu -o errtrace

this_dir="$(cd $(dirname $0) && pwd)"

for container in "$@"; do
  launch_script="$this_dir/$container"
  if ! [ -x "$launch_script" ]; then
    echo "Launch script not found. Generating..."
    docker run --rm dockcross/$container > $launch_script
    chmod u+x $launch_script
  fi
done
