// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// <signal.h> for wasm32 (minimal — no signal delivery on wasm).

#ifndef IREE_WASM_LIBC_SIGNAL_H_
#define IREE_WASM_LIBC_SIGNAL_H_

typedef int sig_atomic_t;
typedef void (*sighandler_t)(int);

#define SIG_DFL ((sighandler_t)0)
#define SIG_IGN ((sighandler_t)1)
#define SIG_ERR ((sighandler_t) - 1)

#define SIGABRT 6
#define SIGFPE 8
#define SIGILL 4
#define SIGINT 2
#define SIGSEGV 11
#define SIGTERM 15

sighandler_t signal(int signum, sighandler_t handler);
int raise(int signum);

#endif  // IREE_WASM_LIBC_SIGNAL_H_
