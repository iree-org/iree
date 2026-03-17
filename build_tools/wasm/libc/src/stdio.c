// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// <stdio.h> for wasm32.
//
// IREE uses eyalroz/printf for all formatted output. The snprintf/vsnprintf
// functions here are thin wrappers that defer to the eyalroz library via the
// IREE printf API (iree_snprintf, etc.). The printf/fprintf functions that
// write to streams use the iree_wasm_write syscall import.
//
// FILE is a minimal struct — just enough to carry a file descriptor. No
// buffering, no seeking, no real filesystem. stdout and stderr write to the
// JS console via the write import. stdin always returns EOF.

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "syscall_imports.h"

//===----------------------------------------------------------------------===//
// FILE implementation
//===----------------------------------------------------------------------===//

struct _iree_wasm_file {
  int fd;
  int error;
  int eof;
};

static struct _iree_wasm_file stdin_file = {0, 0, 1};  // Always EOF.
static struct _iree_wasm_file stdout_file = {1, 0, 0};
static struct _iree_wasm_file stderr_file = {2, 0, 0};

FILE* stdin = &stdin_file;
FILE* stdout = &stdout_file;
FILE* stderr = &stderr_file;

//===----------------------------------------------------------------------===//
// Formatted output — delegated to eyalroz/printf via IREE's printf API.
//
// IREE's printf.c provides iree_snprintf/iree_vsnprintf which call
// eyalroz/printf's vsnprintf_. The libc snprintf/vsnprintf below are
// provided for code that calls the standard names directly (e.g., third-party
// code, gtest). They have the same implementation — just call eyalroz.
//
// eyalroz/printf provides snprintf_/vsnprintf_ which are the actual
// implementations. We declare them here to avoid pulling in iree headers
// from the libc (the libc must be self-contained).
//===----------------------------------------------------------------------===//

// eyalroz/printf entry points (defined in third_party/printf/printf.c).
extern int snprintf_(char* buffer, size_t count, const char* format, ...);
extern int vsnprintf_(char* buffer, size_t count, const char* format,
                      va_list args);

int snprintf(char* buffer, size_t count, const char* format, ...) {
  va_list args;
  va_start(args, format);
  int result = vsnprintf_(buffer, count, format, args);
  va_end(args);
  return result;
}

int vsnprintf(char* buffer, size_t count, const char* format, va_list args) {
  return vsnprintf_(buffer, count, format, args);
}

int sprintf(char* buffer, const char* format, ...) {
  va_list args;
  va_start(args, format);
  int result = vsnprintf_(buffer, (size_t)-1, format, args);
  va_end(args);
  return result;
}

int vsprintf(char* buffer, const char* format, va_list args) {
  return vsnprintf_(buffer, (size_t)-1, format, args);
}

//===----------------------------------------------------------------------===//
// Stream output — writes to JS console via wasm import.
//===----------------------------------------------------------------------===//

// Formatting buffer for printf/fprintf. 4KB is generous for single format
// calls. If a single printf produces more than this, the output is truncated.
// IREE's own formatting always uses snprintf with explicit buffers, so this
// limit only affects third-party code calling printf directly.
#define PRINTF_BUFFER_SIZE 4096

int printf(const char* format, ...) {
  char buffer[PRINTF_BUFFER_SIZE];
  va_list args;
  va_start(args, format);
  int length = vsnprintf_(buffer, sizeof(buffer), format, args);
  va_end(args);
  if (length > 0) {
    int write_length =
        length < (int)sizeof(buffer) ? length : (int)sizeof(buffer) - 1;
    iree_wasm_write(1, buffer, write_length);
  }
  return length;
}

int vprintf(const char* format, va_list args) {
  char buffer[PRINTF_BUFFER_SIZE];
  int length = vsnprintf_(buffer, sizeof(buffer), format, args);
  if (length > 0) {
    int write_length =
        length < (int)sizeof(buffer) ? length : (int)sizeof(buffer) - 1;
    iree_wasm_write(1, buffer, write_length);
  }
  return length;
}

int fprintf(FILE* stream, const char* format, ...) {
  char buffer[PRINTF_BUFFER_SIZE];
  va_list args;
  va_start(args, format);
  int length = vsnprintf_(buffer, sizeof(buffer), format, args);
  va_end(args);
  if (length > 0) {
    int write_length =
        length < (int)sizeof(buffer) ? length : (int)sizeof(buffer) - 1;
    iree_wasm_write(stream->fd, buffer, write_length);
  }
  return length;
}

int vfprintf(FILE* stream, const char* format, va_list args) {
  char buffer[PRINTF_BUFFER_SIZE];
  int length = vsnprintf_(buffer, sizeof(buffer), format, args);
  if (length > 0) {
    int write_length =
        length < (int)sizeof(buffer) ? length : (int)sizeof(buffer) - 1;
    iree_wasm_write(stream->fd, buffer, write_length);
  }
  return length;
}

//===----------------------------------------------------------------------===//
// Character and string I/O
//===----------------------------------------------------------------------===//

int fputc(int character, FILE* stream) {
  char c = (char)character;
  int32_t result = iree_wasm_write(stream->fd, &c, 1);
  return result == 1 ? (unsigned char)c : EOF;
}

int fputs(const char* string, FILE* stream) {
  size_t length = strlen(string);
  int32_t result = iree_wasm_write(stream->fd, string, (int32_t)length);
  return result >= 0 ? result : EOF;
}

int fgetc(FILE* stream) {
  (void)stream;
  return EOF;  // No input on freestanding wasm.
}

int getchar(void) { return EOF; }

int ungetc(int character, FILE* stream) {
  (void)character;
  (void)stream;
  return EOF;
}

//===----------------------------------------------------------------------===//
// Block I/O
//===----------------------------------------------------------------------===//

size_t fread(void* buffer, size_t size, size_t count, FILE* stream) {
  (void)buffer;
  (void)size;
  (void)count;
  (void)stream;
  return 0;  // No input on freestanding wasm.
}

size_t fwrite(const void* buffer, size_t size, size_t count, FILE* stream) {
  if (size == 0 || count == 0) return 0;
  size_t total = size * count;

  // Detect size*count overflow and int32_t truncation. The wasm import takes
  // int32_t, so we cannot write more than INT32_MAX bytes in a single call.
  if (total / size != count || total > (size_t)INT32_MAX) {
    stream->error = 1;
    return 0;
  }

  int32_t result = iree_wasm_write(stream->fd, buffer, (int32_t)total);
  if (result < 0) {
    stream->error = 1;
    return 0;
  }
  return (size_t)result / size;
}

int fflush(FILE* stream) {
  (void)stream;
  return 0;  // No buffering.
}

//===----------------------------------------------------------------------===//
// File operations (stubs)
//===----------------------------------------------------------------------===//

FILE* fopen(const char* pathname, const char* mode) {
  (void)pathname;
  (void)mode;
  return NULL;
}

FILE* fdopen(int fd, const char* mode) {
  (void)fd;
  (void)mode;
  return NULL;
}

FILE* freopen(const char* pathname, const char* mode, FILE* stream) {
  (void)pathname;
  (void)mode;
  (void)stream;
  return NULL;
}

int fclose(FILE* stream) {
  (void)stream;
  return 0;
}

int fseek(FILE* stream, long offset, int whence) {
  (void)stream;
  (void)offset;
  (void)whence;
  return -1;
}

long ftell(FILE* stream) {
  (void)stream;
  return -1;
}

void rewind(FILE* stream) { (void)stream; }
int feof(FILE* stream) { return stream->eof; }
int ferror(FILE* stream) { return stream->error; }
void clearerr(FILE* stream) {
  stream->error = 0;
  stream->eof = 0;
}

int fileno(FILE* stream) { return stream->fd; }

int remove(const char* pathname) {
  (void)pathname;
  return -1;
}

int rename(const char* oldpath, const char* newpath) {
  (void)oldpath;
  (void)newpath;
  return -1;
}
