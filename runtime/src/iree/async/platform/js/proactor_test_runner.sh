#!/bin/bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Test runner for JS proactor wasm binaries under Node.js Worker threads.
#
# Used as --run_under for proactor wasm tests. Bazel invokes:
#   <this script> <path/to/test.wasm> [test args...]
#
# The wasm binary runs in a Worker thread with the iree_proactor imports
# wired up. The main thread handles timer scheduling and completion delivery.
# Exit code from the wasm program propagates to Bazel.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec node "$SCRIPT_DIR/proactor_test_runner.mjs" "$@"
