# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""wasi-sdk version and integrity constants.

Single source of truth for both Bazel and CMake. CMake parses this file
with regex to extract the values; keep the format machine-friendly:
  IDENTIFIER = "value"
  IDENTIFIER = { "key": "value", ... }
"""

# wasi-sdk release from https://github.com/WebAssembly/wasi-sdk/releases
# Includes: clang 21.1.4, lld, wasi-libc (musl), libc++, compiler-rt.
WASI_SDK_VERSION = "30.0"
WASI_SDK_TAG = "wasi-sdk-30"

# URL template: replace {version}, {arch}, {os}.
WASI_SDK_URL_TEMPLATE = "https://github.com/WebAssembly/wasi-sdk/releases/download/{tag}/wasi-sdk-{version}-{arch}-{os}.tar.gz"

# SHA-256 digests per host platform.
WASI_SDK_SHA256 = {
    "x86_64-linux": "0507679dff16814b74516cd969a9b16d2ced1347388024bc7966264648c78bfb",
    "arm64-linux": "6f2977942308d91b0123978da3c6a0d6fce780994b3b020008c617e26764ea40",
    "arm64-macos": "2c2ed99296857e60fd14c3f40fe226231f296409502491094704089c31a16740",
    "x86_64-macos": "1594a0791309781bf0d0224431c3556ec4a2326b205687b659f6550d08d8b13e",
}
