// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// errno for wasm32.
//
// Plain global variable for single-mutator wasm. When wasm threads are enabled,
// this should become _Thread_local and the host must initialize wasm TLS for
// each worker.

#include <errno.h>

int errno = 0;
