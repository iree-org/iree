// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// JS implementation of iree_wasm_csprng_fill for the Web platform.
// Provides the Wasm import backing the extern declaration in csprng.c.

export function register(env, memory) {
  env.iree_wasm_csprng_fill = (ptr, length) => {
    try {
      // crypto.getRandomValues() has a 65536-byte per-call limit.
      for (let offset = 0; offset < length; offset += 65536) {
        const chunk = Math.min(length - offset, 65536);
        crypto.getRandomValues(
            new Uint8Array(memory.buffer, ptr + offset, chunk));
      }
      return 0;
    } catch (e) {
      return 1;
    }
  };
}
