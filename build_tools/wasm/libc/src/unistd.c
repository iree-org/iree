// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// <unistd.h> stubs for wasm32.
//
// write() delegates to the JS host via wasm import. All other POSIX I/O
// functions return errors — there is no filesystem on freestanding wasm.

#include <errno.h>
#include <unistd.h>

#include "syscall_imports.h"

ssize_t write(int fd, const void* buffer, size_t count) {
  return (ssize_t)iree_wasm_write(fd, buffer, (int32_t)count);
}

ssize_t read(int fd, void* buffer, size_t count) {
  (void)fd;
  (void)buffer;
  (void)count;
  errno = ENOSYS;
  return -1;
}

int close(int fd) {
  (void)fd;
  return 0;
}

off_t lseek(int fd, off_t offset, int whence) {
  (void)fd;
  (void)offset;
  (void)whence;
  errno = ENOSYS;
  return -1;
}

int dup(int fd) {
  (void)fd;
  errno = ENOSYS;
  return -1;
}

int dup2(int oldfd, int newfd) {
  (void)oldfd;
  (void)newfd;
  errno = ENOSYS;
  return -1;
}

int ftruncate(int fd, off_t length) {
  (void)fd;
  (void)length;
  errno = ENOSYS;
  return -1;
}

ssize_t pread(int fd, void* buffer, size_t count, off_t offset) {
  (void)fd;
  (void)buffer;
  (void)count;
  (void)offset;
  errno = ENOSYS;
  return -1;
}

ssize_t pwrite(int fd, const void* buffer, size_t count, off_t offset) {
  (void)fd;
  (void)buffer;
  (void)count;
  (void)offset;
  errno = ENOSYS;
  return -1;
}

int fsync(int fd) {
  (void)fd;
  return 0;
}

int unlink(const char* pathname) {
  (void)pathname;
  errno = ENOSYS;
  return -1;
}

int isatty(int fd) {
  // stdout and stderr are "terminal-like" (they go to the JS console).
  return (fd == 1 || fd == 2) ? 1 : 0;
}

int usleep(useconds_t usec) {
  (void)usec;
  // No sleep on single-threaded wasm — return immediately.
  return 0;
}

unsigned int sleep(unsigned int seconds) {
  (void)seconds;
  return 0;
}

int pipe(int pipefd[2]) {
  (void)pipefd;
  errno = ENOSYS;
  return -1;
}
