// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/loader.h"

#include <stdio.h>
#include <stdlib.h>

#include "iree/compiler/embedding_api.h"

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

// Some operating systems have a prefix for cdecl exported symbols.
#if __APPLE__
#define IREE_CDECL_SYMBOL_PREFIX "_"
#else
#define IREE_CDECL_SYMBOL_PREFIX ""
#endif

namespace {
DlHandle libraryHandle = nullptr;

#define HANDLE_SYMBOL(fn_name) decltype(fn_name) *__##fn_name = nullptr;
#include "./handle_symbols.inc"
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

#define HANDLE_SYMBOL(fn_name)                                           \
  __##fn_name = (decltype(__##fn_name))lookupLibrarySymbol(              \
      localLibraryHandle, IREE_CDECL_SYMBOL_PREFIX #fn_name);            \
  if (!__##fn_name) {                                                    \
    fprintf(stderr, "IREE COMPILER ERROR: Could not find symbol '%s'\n", \
            IREE_CDECL_SYMBOL_PREFIX #fn_name);                          \
    return false;                                                        \
  }
#include "./handle_symbols.inc"
#undef HANDLE_SYMBOL

  // Mark as loaded.
  libraryHandle = localLibraryHandle;
  return true;
}

////////////////////////////////////////////////////////////////////////////////
// Trampoline functions.
////////////////////////////////////////////////////////////////////////////////

void ireeCompilerErrorDestroy(iree_compiler_error_t *error) {
  return __ireeCompilerErrorDestroy(error);
}

const char *ireeCompilerErrorGetMessage(iree_compiler_error_t *error) {
  return __ireeCompilerErrorGetMessage(error);
}

int ireeCompilerGetAPIVersion() {
  assertLoaded();
  return __ireeCompilerGetAPIVersion();
}

void ireeCompilerGlobalInitialize() {
  assertLoaded();
  __ireeCompilerGlobalInitialize();
}

void ireeCompilerGetProcessCLArgs(int *argc, const char ***argv) {
  assertLoaded();
  __ireeCompilerGetProcessCLArgs(argc, argv);
}

void ireeCompilerSetupGlobalCL(int argc, const char **argv, const char *banner,
                               bool installSignalHandlers) {
  assertLoaded();
  __ireeCompilerSetupGlobalCL(argc, argv, banner, installSignalHandlers);
}

void ireeCompilerGlobalShutdown() {
  assertLoaded();
  __ireeCompilerGlobalShutdown();
}

void ireeCompilerEnumerateRegisteredHALTargetBackends(
    void (*callback)(const char *backend, void *userData), void *userData) {
  __ireeCompilerEnumerateRegisteredHALTargetBackends(callback, userData);
}

iree_compiler_session_t *ireeCompilerSessionCreate() {
  assertLoaded();
  return __ireeCompilerSessionCreate();
}

void ireeCompilerSessionDestroy(iree_compiler_session_t *session) {
  assertLoaded();
  __ireeCompilerSessionDestroy(session);
}

iree_compiler_error_t *ireeCompilerSessionSetFlags(
    iree_compiler_session_t *session, int argc, const char *const *argv) {
  assertLoaded();
  return __ireeCompilerSessionSetFlags(session, argc, argv);
}

void ireeCompilerSessionGetFlags(
    iree_compiler_session_t *session, bool nonDefaultOnly,
    void (*onFlag)(const char *flag, size_t length, void *), void *userData) {
  assertLoaded();
  return __ireeCompilerSessionGetFlags(session, nonDefaultOnly, onFlag,
                                       userData);
}

iree_compiler_invocation_t *ireeCompilerInvocationCreate(
    iree_compiler_session_t *session) {
  return __ireeCompilerInvocationCreate(session);
}

void ireeCompilerInvocationEnableCallbackDiagnostics(
    iree_compiler_invocation_t *inv, int flags,
    void (*callback)(enum iree_compiler_diagnostic_severity_t severity,
                     const char *message, size_t messageSize, void *userData),
    void *userData) {
  __ireeCompilerInvocationEnableCallbackDiagnostics(inv, flags, callback,
                                                    userData);
}

void ireeCompilerInvocationEnableConsoleDiagnostics(
    iree_compiler_invocation_t *run) {
  __ireeCompilerInvocationEnableConsoleDiagnostics(run);
}

void ireeCompilerInvocationDestroy(iree_compiler_invocation_t *run) {
  __ireeCompilerInvocationDestroy(run);
}

void ireeCompilerInvocationSetCrashHandler(
    iree_compiler_invocation_t *inv, bool genLocalReproducer,
    iree_compiler_error_t *(*onCrashCallback)(
        iree_compiler_output_t **outOutput, void *userData),
    void *userData) {
  __ireeCompilerInvocationSetCrashHandler(inv, genLocalReproducer,
                                          onCrashCallback, userData);
}

bool ireeCompilerInvocationParseSource(iree_compiler_invocation_t *run,
                                       iree_compiler_source_t *source) {
  return __ireeCompilerInvocationParseSource(run, source);
}

void ireeCompilerInvocationSetCompileToPhase(iree_compiler_invocation_t *run,
                                             const char *phase) {
  __ireeCompilerInvocationSetCompileToPhase(run, phase);
}

IREE_EMBED_EXPORTED void ireeCompilerInvocationSetVerifyIR(
    iree_compiler_invocation_t *run, bool enable) {
  __ireeCompilerInvocationSetVerifyIR(run, enable);
}

bool ireeCompilerInvocationPipeline(iree_compiler_invocation_t *run,
                                    enum iree_compiler_pipeline_t pipeline) {
  return __ireeCompilerInvocationPipeline(run, pipeline);
}

iree_compiler_error_t *ireeCompilerInvocationOutputIR(
    iree_compiler_invocation_t *run, iree_compiler_output_t *output) {
  return __ireeCompilerInvocationOutputIR(run, output);
}

iree_compiler_error_t *ireeCompilerInvocationOutputVMBytecode(
    iree_compiler_invocation_t *run, iree_compiler_output_t *output) {
  return __ireeCompilerInvocationOutputVMBytecode(run, output);
}

iree_compiler_error_t *ireeCompilerInvocationOutputVMCSource(
    iree_compiler_invocation_t *run, iree_compiler_output_t *output) {
  return __ireeCompilerInvocationOutputVMCSource(run, output);
}

iree_compiler_error_t *ireeCompilerInvocationOutputHALExecutable(
    iree_compiler_invocation_t *run, iree_compiler_output_t *output) {
  return __ireeCompilerInvocationOutputHALExecutable(run, output);
}

void ireeCompilerSourceDestroy(iree_compiler_source_t *source) {
  __ireeCompilerSourceDestroy(source);
}

iree_compiler_error_t *ireeCompilerSourceOpenFile(
    iree_compiler_session_t *session, const char *filePath,
    iree_compiler_source_t **out_source) {
  return __ireeCompilerSourceOpenFile(session, filePath, out_source);
}

iree_compiler_error_t *ireeCompilerSourceWrapBuffer(
    iree_compiler_session_t *session, const char *bufferName,
    const char *buffer, size_t length, bool isNullTerminated,
    iree_compiler_source_t **out_source) {
  return __ireeCompilerSourceWrapBuffer(session, bufferName, buffer, length,
                                        isNullTerminated, out_source);
}

iree_compiler_error_t *ireeCompilerSourceSplit(
    iree_compiler_source_t *source,
    void (*callback)(iree_compiler_source_t *source, void *userData),
    void *userData) {
  return __ireeCompilerSourceSplit(source, callback, userData);
}

void ireeCompilerOutputDestroy(iree_compiler_output_t *output) {
  __ireeCompilerOutputDestroy(output);
}

iree_compiler_error_t *ireeCompilerOutputOpenFile(
    const char *filePath, iree_compiler_output_t **out_output) {
  return __ireeCompilerOutputOpenFile(filePath, out_output);
}

iree_compiler_error_t *ireeCompilerOutputOpenFD(
    int fd, iree_compiler_output_t **out_output) {
  return __ireeCompilerOutputOpenFD(fd, out_output);
}

iree_compiler_error_t *ireeCompilerOutputOpenMembuffer(
    iree_compiler_output_t **out_output) {
  return __ireeCompilerOutputOpenMembuffer(out_output);
}

void ireeCompilerOutputKeep(iree_compiler_output_t *output) {
  __ireeCompilerOutputKeep(output);
}

iree_compiler_error_t *ireeCompilerOutputMapMemory(
    iree_compiler_output_t *output, void **contents, uint64_t *size) {
  return __ireeCompilerOutputMapMemory(output, contents, size);
}

iree_compiler_error_t *ireeCompilerOutputWrite(iree_compiler_output_t *output,
                                               const void *data,
                                               size_t length) {
  return __ireeCompilerOutputWrite(output, data, length);
}
