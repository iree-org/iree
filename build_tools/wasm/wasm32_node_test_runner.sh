#!/bin/bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Test runner for wasm32 bundled binaries. Invoked by iree_wasm_cc_test
# targets via sh_test. The argument is the rootpath of the bundled .mjs.

exec node "$@"
