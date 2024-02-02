#!/bin/bash

# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

PYTHON_SCRIPTS_DIR="$(python3 -c "import sysconfig; print(sysconfig.get_path('scripts'))"):$HOME/.local/bin"
export PATH="$PYTHON_SCRIPTS_DIR:$PATH"

python3 -m pip install lit
export LLVM_EXTERNAL_LIT="$(which lit)"
