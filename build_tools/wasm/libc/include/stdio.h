// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// <stdio.h> for wasm32.
// IREE uses eyalroz/printf for all formatting; this header provides the
// function signatures that the runtime expects.

#ifndef IREE_WASM_LIBC_STDIO_H_
#define IREE_WASM_LIBC_STDIO_H_

#include <stdarg.h>
#include <stddef.h>

// File handle type (opaque — no real filesystem on wasm32 freestanding).
typedef struct _iree_wasm_file FILE;

// Standard streams (provided by the libc implementation).
extern FILE* stdin;
extern FILE* stdout;
extern FILE* stderr;

#define EOF (-1)

// Formatted output (provided by eyalroz/printf).
int printf(const char* format, ...) __attribute__((format(printf, 1, 2)));
int fprintf(FILE* stream, const char* format, ...)
    __attribute__((format(printf, 2, 3)));
int sprintf(char* buffer, const char* format, ...)
    __attribute__((format(printf, 2, 3)));
int snprintf(char* buffer, size_t count, const char* format, ...)
    __attribute__((format(printf, 3, 4)));
int vprintf(const char* format, va_list args);
int vfprintf(FILE* stream, const char* format, va_list args);
int vsprintf(char* buffer, const char* format, va_list args);
int vsnprintf(char* buffer, size_t count, const char* format, va_list args);

// Stream I/O.
int fputc(int character, FILE* stream);
int fputs(const char* string, FILE* stream);
int fgetc(FILE* stream);
int getchar(void);
int ungetc(int character, FILE* stream);
size_t fread(void* buffer, size_t size, size_t count, FILE* stream);
size_t fwrite(const void* buffer, size_t size, size_t count, FILE* stream);
int fflush(FILE* stream);

// File operations (stubs — no real filesystem on wasm32 freestanding).
FILE* fopen(const char* pathname, const char* mode);
FILE* fdopen(int fd, const char* mode);
FILE* freopen(const char* pathname, const char* mode, FILE* stream);
int fclose(FILE* stream);
int fseek(FILE* stream, long offset, int whence);
long ftell(FILE* stream);
void rewind(FILE* stream);
int feof(FILE* stream);
int ferror(FILE* stream);
void clearerr(FILE* stream);
int fileno(FILE* stream);
int remove(const char* pathname);
int rename(const char* oldpath, const char* newpath);

#define putchar(c) fputc((c), stdout)
#define puts(s) fputs((s), stdout)
#define getc(stream) fgetc(stream)

#endif  // IREE_WASM_LIBC_STDIO_H_
