#!/bin/bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Test runner for bundled WebAssembly binaries. Invoked by iree_wasm_cc_test
# targets via sh_test. The argument is the rootpath of the bundled .mjs.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <bundle.mjs> [args...]" >&2
  exit 1
fi

node_bin="${IREE_WASM_NODE:-}"
if [[ -z "${node_bin}" ]]; then
  node_bin="$(command -v node || true)"
fi
if [[ -z "${node_bin}" ]]; then
  node_bin="$(command -v nodejs || true)"
fi
if [[ -z "${node_bin}" ]]; then
  echo "Unable to find node. Set IREE_WASM_NODE to a Node.js executable." >&2
  exit 1
fi

# Ubuntu snap installs expose /snap/bin/node as a launcher through /usr/bin/snap.
# The launcher exits inside Bazel's Linux sandbox before running the test, so
# bypass it when the real snap-packaged node binary is available.
node_realpath="$(readlink -f "${node_bin}" || true)"
if [[ "${node_realpath}" == "/usr/bin/snap" &&
      -x "/snap/node/current/bin/node" ]]; then
  node_bin="/snap/node/current/bin/node"
fi

exec "${node_bin}" "$@"
