// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// errno for wasm32.
//
// Plain global variable for single-threaded wasm. When wasm threads are
// enabled (SharedArrayBuffer + -pthread -matomics), this should be changed
// to _Thread_local — the LLVM wasm backend supports TLS via the local-exec
// model with __tls_base per-thread initialization.

#include <errno.h>

int errno = 0;
