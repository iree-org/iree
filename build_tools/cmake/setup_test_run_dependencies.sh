#!/bin/bash

# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Setup run requirements for tests.

set -euo pipefail

if [[ $OSTYPE =~ ^linux ]]; then
  source /etc/lsb-release
  if [[ $DISTRIB_ID == "Ubuntu" ]]; then
    if command -v sudo &> /dev/null; then
      SUDO=sudo
    else
      # No sudo, probably running inside docker as root
      SUDO=""
    fi
    $SUDO apt update
    $SUDO apt -y install libopenmpi-dev
  fi
fi
if [[ $OSTYPE =~ ^darwin ]]; then
  brew install open-mpi
fi

if [[ -v IREE_PYTHON_VENV_DIR ]]; then
  source "$IREE_PYTHON_VENV_DIR/bin/activate"
fi
pip install -r tests/run_requirements.txt
