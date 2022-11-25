// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/API2/Stub/Loader.h"

#include <stdio.h>
#include <stdlib.h>

#include "iree/compiler/API2/Embed.h"

#if (defined(_WIN32) || defined(__CYGWIN__))
// Windows implementation
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
namespace {
using DlHandle = HMODULE;
DlHandle loadLibrary(const char *libraryPath) {
  HMODULE lib = LoadLibraryExA(libraryPath, nullptr, 0);
  if (lib) return lib;
  DWORD errorMessageID = GetLastError();
  LPSTR messageBuffer = nullptr;
  size_t size = FormatMessageA(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      NULL, errorMessageID, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      (LPSTR)&messageBuffer, 0, NULL);

  fprintf(stderr,
          "IREE COMPILER ERROR: Could not open compiler dll %s : %.*s\n",
          libraryPath, static_cast<int>(size), messageBuffer);
  LocalFree(messageBuffer);
  return nullptr;
}
void *lookupLibrarySymbol(DlHandle lib, const char *symbol) {
  return (void *)GetProcAddress(lib, symbol);
}
}  // namespace
#else
// Posix impl
#include <dlfcn.h>
namespace {
using DlHandle = void *;
DlHandle loadLibrary(const char *libraryPath) {
  DlHandle lib = dlopen(libraryPath, RTLD_NOW | RTLD_LOCAL);
  if (!lib) {
    const char *reason = dlerror();
    if (!reason) reason = "";
    fprintf(stderr,
            "IREE COMPILER ERROR: Could not open compiler library %s : %s\n",
            libraryPath, reason);
    return nullptr;
  }
  return lib;
}
void *lookupLibrarySymbol(DlHandle lib, const char *symbol) {
  return dlsym(lib, symbol);
}
}  // namespace
#endif

namespace {
DlHandle libraryHandle = nullptr;

#define HANDLE_SYMBOL(fn_name) decltype(fn_name) *__##fn_name = nullptr;
#include "iree/compiler/API2/Stub/HandleSymbols.inc"
#undef HANDLE_SYMBOL

void assertLoaded() {
  if (libraryHandle) return;
  fprintf(stderr,
          "FATAL ERROR: Attempt to call IREE compiler stub methods before "
          "library loaded\n");
  abort();
}
}  // namespace

bool ireeCompilerLoadLibrary(const char *libraryPath) {
  if (libraryHandle) {
    fprintf(stderr, "ERROR: IREE compiler stub already initialized\n");
    return false;
  }
  DlHandle localLibraryHandle = loadLibrary(libraryPath);
  if (!localLibraryHandle) {
    return false;
  }

#define HANDLE_SYMBOL(fn_name)                                                 \
  __##fn_name = (decltype(__##fn_name))lookupLibrarySymbol(localLibraryHandle, \
                                                           #fn_name);          \
  if (!__##fn_name) {                                                          \
    fprintf(stderr, "IREE COMPILER ERROR: Could not find symbol %s\n",         \
            #fn_name);                                                         \
    return false;                                                              \
  }
#include "iree/compiler/API2/Stub/HandleSymbols.inc"
#undef HANDLE_SYMBOL

  // Mark as loaded.
  libraryHandle = localLibraryHandle;
  return true;
}

////////////////////////////////////////////////////////////////////////////////
// Trampoline functions.
////////////////////////////////////////////////////////////////////////////////

void ireeCompilerErrorDestroy(struct iree_compiler_error_t *error) {
  return __ireeCompilerErrorDestroy(error);
}

const char *ireeCompilerErrorGetMessage(struct iree_compiler_error_t *error) {
  return __ireeCompilerErrorGetMessage(error);
}

int ireeCompilerGetAPIVersion() {
  assertLoaded();
  return __ireeCompilerGetAPIVersion();
}

void ireeCompilerGlobalInitialize(bool initializeCommandLine) {
  assertLoaded();
  __ireeCompilerGlobalInitialize(initializeCommandLine);
}

void ireeCompilerGlobalShutdown() {
  assertLoaded();
  __ireeCompilerGlobalShutdown();
}

void ireeCompilerEnumerateRegisteredHALTargetBackends(
    void (*callback)(const char *backend, void *userData), void *userData) {
  __ireeCompilerEnumerateRegisteredHALTargetBackends(callback, userData);
}

struct iree_compiler_session_t *ireeCompilerSessionCreate() {
  assertLoaded();
  return __ireeCompilerSessionCreate();
}

void ireeCompilerSessionDestroy(struct iree_compiler_session_t *session) {
  assertLoaded();
  __ireeCompilerSessionDestroy(session);
}

struct iree_compiler_run_t *ireeCompilerRunCreate(
    struct iree_compiler_session_t *session) {
  return __ireeCompilerRunCreate(session);
}

void ireeCompilerRunEnableConsoleDiagnostics(struct iree_compiler_run_t *run) {
  __ireeCompilerRunEnableConsoleDiagnostics(run);
}

void ireeCompilerRunDestroy(struct iree_compiler_run_t *run) {
  __ireeCompilerRunDestroy(run);
}

bool ireeCompilerRunParseSource(struct iree_compiler_run_t *run,
                                struct iree_compiler_source_t *source) {
  return __ireeCompilerRunParseSource(run, source);
}

void ireeCompilerRunSetCompileToPhase(struct iree_compiler_run_t *run,
                                      const char *phase) {
  __ireeCompilerRunSetCompileToPhase(run, phase);
}

IREE_EMBED_EXPORTED void ireeCompilerRunSetVerifyIR(
    struct iree_compiler_run_t *run, bool enable) {
  __ireeCompilerRunSetVerifyIR(run, enable);
}

bool ireeCompilerRunPipeline(struct iree_compiler_run_t *run,
                             enum iree_compiler_pipeline_t pipeline) {
  return __ireeCompilerRunPipeline(run, pipeline);
}

struct iree_compiler_error_t *ireeCompilerRunOutputIR(
    struct iree_compiler_run_t *run, struct iree_compiler_output_t *output) {
  return __ireeCompilerRunOutputIR(run, output);
}

struct iree_compiler_error_t *ireeCompilerRunOutputVMBytecode(
    struct iree_compiler_run_t *run, struct iree_compiler_output_t *output) {
  return __ireeCompilerRunOutputVMBytecode(run, output);
}

struct iree_compiler_error_t *ireeCompilerRunOutputVMCSource(
    struct iree_compiler_run_t *run, struct iree_compiler_output_t *output) {
  return __ireeCompilerRunOutputVMCSource(run, output);
}

struct iree_compiler_error_t *ireeCompilerRunOutputHALExecutable(
    struct iree_compiler_run_t *run, struct iree_compiler_output_t *output) {
  return __ireeCompilerRunOutputHALExecutable(run, output);
}

struct iree_compiler_source_t *ireeCompilerSourceCreate(
    struct iree_compiler_session_t *session) {
  return __ireeCompilerSourceCreate(session);
}

void ireeCompilerSourceDestroy(struct iree_compiler_source_t *source) {
  __ireeCompilerSourceDestroy(source);
}

struct iree_compiler_error_t *ireeCompilerSourceOpenFile(
    struct iree_compiler_source_t *source, const char *filePath) {
  return __ireeCompilerSourceOpenFile(source, filePath);
}

struct iree_compiler_error_t *ireeCompilerSourceWrapBuffer(
    struct iree_compiler_source_t *source, const char *bufferName,
    const char *buffer, size_t length) {
  return __ireeCompilerSourceWrapBuffer(source, bufferName, buffer, length);
}

struct iree_compiler_error_t *ireeCompilerSourceSplit(
    struct iree_compiler_source_t *source,
    void (*callback)(struct iree_compiler_source_t *source, void *userData),
    void *userData) {
  return __ireeCompilerSourceSplit(source, callback, userData);
}

struct iree_compiler_output_t *ireeCompilerOutputCreate() {
  return __ireeCompilerOutputCreate();
}

void ireeCompilerOutputDestroy(struct iree_compiler_output_t *output) {
  __ireeCompilerOutputDestroy(output);
}

struct iree_compiler_error_t *ireeCompilerOutputOpenFile(
    struct iree_compiler_output_t *output, const char *filePath) {
  return __ireeCompilerOutputOpenFile(output, filePath);
}

void ireeCompileOutputKeep(struct iree_compiler_output_t *output) {
  __ireeCompileOutputKeep(output);
}

struct iree_compiler_error_t *ireeCompilerOutputWrite(
    struct iree_compiler_output_t *output, const void *data, size_t length) {
  return __ireeCompilerOutputWrite(output, data, length);
}
