# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Symlinks built binaries from the bazel-bin/ directory into the corresponding
# python packages.

set -euo pipefail

cd "$(dirname $0)"

BINARIES_DIR="${1:-${PWD}/bazel-bin/iree_tf_compiler}"

if [ -f "${BINARIES_DIR}/iree-import-tf" ]; then
  ln -sf "${BINARIES_DIR}/iree-import-tf" python_projects/iree_tf/iree/tools/tf/
fi
if [ -f "${BINARIES_DIR}/iree-import-tflite" ]; then
  ln -sf "${BINARIES_DIR}/iree-import-tflite" python_projects/iree_tflite/iree/tools/tflite/
fi
if [ -f "${BINARIES_DIR}/iree-import-xla" ]; then
  ln -sf "${BINARIES_DIR}/iree-import-xla" python_projects/iree_xla/iree/tools/xla/
fi
