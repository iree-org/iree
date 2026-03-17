// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// JS implementation of iree_wasm_logical_core_count for the Web platform.
// Provides the Wasm import backing the extern declaration in topology_wasm.c.

export function register(env, _memory) {
  env.iree_wasm_logical_core_count = () => {
    return navigator.hardwareConcurrency || 1;
  };
}
