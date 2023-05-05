# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Symlinks built TF import binaries from the specified directory (defaults to
# the appropriate bazel-bin/ subdirectory) into the corresponding Python
# packages. If the binary directory is contained within the root directory, it
# uses a relative symlink, which makes this work when the repository is copied
# or mounted in a Docker container under some other path.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(git rev-parse --show-toplevel)}"
SCRIPT_DIR="$(dirname -- "$( readlink -f -- "$0"; )")";

BINARIES_DIR="${1:-${SCRIPT_DIR}/bazel-bin/iree_tf_compiler}"

function symlink_import_binary() {
  local type="$1"
  local import_binary="${BINARIES_DIR}/iree-import-${type}"
  if [ -f "${import_binary}" ]; then
    local to="${SCRIPT_DIR}/python_projects/iree_${type}/iree/tools/${type}"
    local from="$(realpath --no-symlinks --relative-to=${to} --relative-base="${ROOT_DIR}" "${import_binary}")"
    ln --symbolic --verbose --force "${from}" "${to}"
  fi
}

symlink_import_binary tflite
symlink_import_binary xla
