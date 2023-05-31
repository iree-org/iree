#!/bin/bash
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Source this to set up the local python environment with in-tree, source
# only packages.

IREE_PYTHON3_EXECUTABLE="${IREE_PYTHON3_EXECUTABLE:-$(which python3)}"
PYTHON_SCRIPTS_DIR="$(python -c "import sysconfig; print(sysconfig.get_path('scripts'))"):$HOME/.local/bin"
export PATH="$PYTHON_SCRIPTS_DIR:$PATH"

# Install local source-only Python packages. These do not have a build step
# but export important binaries onto the path.
"${IREE_PYTHON3_EXECUTABLE}" -m pip install integrations/tensorflow/python_projects/iree_tf integrations/tensorflow/python_projects/iree_tflite

