// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// <unistd.h> for wasm32 (minimal — no filesystem, no process control).

#ifndef IREE_WASM_LIBC_UNISTD_H_
#define IREE_WASM_LIBC_UNISTD_H_

#include <stddef.h>
#include <stdint.h>

typedef int32_t ssize_t;
typedef int32_t off_t;
typedef uint32_t useconds_t;

// Standard file descriptors.
#define STDIN_FILENO 0
#define STDOUT_FILENO 1
#define STDERR_FILENO 2

// File operations (stubs — no real filesystem on wasm32 freestanding).
ssize_t read(int fd, void* buffer, size_t count);
ssize_t write(int fd, const void* buffer, size_t count);
int close(int fd);
off_t lseek(int fd, off_t offset, int whence);
int dup(int fd);
int dup2(int oldfd, int newfd);
int ftruncate(int fd, off_t length);
ssize_t pread(int fd, void* buffer, size_t count, off_t offset);
ssize_t pwrite(int fd, const void* buffer, size_t count, off_t offset);
int fsync(int fd);
int unlink(const char* pathname);
int isatty(int fd);
int usleep(useconds_t usec);
unsigned int sleep(unsigned int seconds);
int pipe(int pipefd[2]);

// Seek whence values.
#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2

#endif  // IREE_WASM_LIBC_UNISTD_H_
