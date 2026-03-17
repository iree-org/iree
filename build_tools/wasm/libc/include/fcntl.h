// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// <fcntl.h> for wasm32 (minimal — no real filesystem).

#ifndef IREE_WASM_LIBC_FCNTL_H_
#define IREE_WASM_LIBC_FCNTL_H_

// Open flags.
#define O_RDONLY 0
#define O_WRONLY 1
#define O_RDWR 2
#define O_CREAT 0100
#define O_EXCL 0200
#define O_NOCTTY 0400
#define O_TRUNC 01000
#define O_APPEND 02000
#define O_NONBLOCK 04000
#define O_CLOEXEC 02000000

int open(const char* pathname, int flags, ...);
int fcntl(int fd, int cmd, ...);

// fcntl commands.
#define F_DUPFD 0
#define F_GETFD 1
#define F_SETFD 2
#define F_GETFL 3
#define F_SETFL 4

#endif  // IREE_WASM_LIBC_FCNTL_H_
