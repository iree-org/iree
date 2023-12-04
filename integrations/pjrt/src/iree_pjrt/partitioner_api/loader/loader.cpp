// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_pjrt/partitioner_api/loader.h"

#include <stdio.h>
#include <stdlib.h>

#include "iree_pjrt/partitioner_api/embedding_api.h"

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
          "OPENXLA PARTITIONER ERROR: Could not open compiler dll %s : %.*s\n",
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
    fprintf(
        stderr,
        "OPENXLA PARTITIONER ERROR: Could not open compiler library %s : %s\n",
        libraryPath, reason);
    return nullptr;
  }
  return lib;
}
void *lookupLibrarySymbol(DlHandle lib, const char *symbol) {
  // note: on macOS, dlsym already prepends CDECL_SYMBOL_PREFIX _
  return dlsym(lib, symbol);
}
}  // namespace
#endif

namespace {
DlHandle libraryHandle = nullptr;

#define HANDLE_SYMBOL(fn_name) decltype(fn_name) *__##fn_name = nullptr;
#define HANDLE_VERSIONED_SYMBOL(fn_name, major, minor) HANDLE_SYMBOL(fn_name)
#include "./handle_symbols.inc"
#undef HANDLE_SYMBOL
#undef HANDLE_VERSIONED_SYMBOL

void assertLoaded() {
  if (libraryHandle) return;
  fprintf(
      stderr,
      "FATAL ERROR: Attempt to call OpenXLA partitioner stub methods before "
      "library loaded\n");
  abort();
}
}  // namespace

bool openxlaPartitionerLoadLibrary(const char *libraryPath) {
  if (libraryHandle) {
    fprintf(stderr, "ERROR: OpenXLA partitioner stub already initialized\n");
    return false;
  }
  DlHandle localLibraryHandle = loadLibrary(libraryPath);
  if (!localLibraryHandle) {
    return false;
  }

  // Resolve the api version separately.
  int (*apiVersionFn)() = (int (*)())lookupLibrarySymbol(
      localLibraryHandle, "openxlaPartitionerGetAPIVersion");
  if (!apiVersionFn) {
    fprintf(stderr,
            "OpenXLA partitioner ERROR: Could not find symbol "
            "'openxlaPartitionerGetAPIVersion'\n");
    return false;
  }
  int packedApiVersion = apiVersionFn();
  int apiMinor = packedApiVersion & 0xffff;
  int apiMajor = packedApiVersion >> 16;
  (void)apiMinor;
  (void)apiMajor;

#define HANDLE_SYMBOL(fn_name)                                                 \
  __##fn_name = (decltype(__##fn_name))lookupLibrarySymbol(localLibraryHandle, \
                                                           #fn_name);          \
  if (!__##fn_name) {                                                          \
    fprintf(stderr, "IREE COMPILER ERROR: Could not find symbol '%s'\n",       \
            #fn_name);                                                         \
    return false;                                                              \
  }
#define HANDLE_VERSIONED_SYMBOL(fn_name, availApiMajor, availApiMinor) \
  if (apiMajor > availApiMajor ||                                      \
      (apiMajor == availApiMajor && apiMinor >= availApiMinor)) {      \
    HANDLE_SYMBOL(fn_name);                                            \
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

void openxlaPartitionerErrorDestroy(openxla_partitioner_error_t *error) {
  assertLoaded();
  return __openxlaPartitionerErrorDestroy(error);
}

const char *openxlaPartitionerErrorGetMessage(
    openxla_partitioner_error_t *error) {
  assertLoaded();
  return __openxlaPartitionerErrorGetMessage(error);
}

int openxlaPartitionerGetAPIVersion() {
  assertLoaded();
  return __openxlaPartitionerGetAPIVersion();
}

void openxlaPartitionerGlobalInitialize() {
  assertLoaded();
  __openxlaPartitionerGlobalInitialize();
}

const char *openxlaPartitionerGetRevision() {
  assertLoaded();
  if (__openxlaPartitionerGetRevision) {
    return __openxlaPartitionerGetRevision();
  } else {
    return "";
  }
}

void openxlaPartitionerSetupGlobalCL(int argc, const char **argv,
                                     const char *banner,
                                     bool installSignalHandlers) {
  assertLoaded();
  __openxlaPartitionerSetupGlobalCL(argc, argv, banner, installSignalHandlers);
}

void openxlaPartitionerGlobalShutdown() {
  assertLoaded();
  __openxlaPartitionerGlobalShutdown();
}

openxla_partitioner_session_t *openxlaPartitionerSessionCreate() {
  assertLoaded();
  return __openxlaPartitionerSessionCreate();
}

void openxlaPartitionerSessionDestroy(openxla_partitioner_session_t *session) {
  assertLoaded();
  __openxlaPartitionerSessionDestroy(session);
}

openxla_partitioner_error_t *openxlaPartitionerSessionSetFlags(
    openxla_partitioner_session_t *session, int argc, const char *const *argv) {
  assertLoaded();
  return __openxlaPartitionerSessionSetFlags(session, argc, argv);
}

void openxlaPartitionerSessionGetFlags(
    openxla_partitioner_session_t *session, bool nonDefaultOnly,
    void (*onFlag)(const char *flag, size_t length, void *), void *userData) {
  assertLoaded();
  return __openxlaPartitionerSessionGetFlags(session, nonDefaultOnly, onFlag,
                                             userData);
}

openxla_partitioner_invocation_t *openxlaPartitionerInvocationCreate(
    openxla_partitioner_session_t *session) {
  assertLoaded();
  return __openxlaPartitionerInvocationCreate(session);
}

// void openxlaPartitionerInvocationEnableCallbackDiagnostics(
//     openxla_partitioner_invocation_t *inv, int flags,
//     void (*callback)(enum openxla_partitioner_diagnostic_severity_t severity,
//                      const char *message, size_t messageSize, void
//                      *userData),
//     void *userData) {
//   __openxlaPartitionerInvocationEnableCallbackDiagnostics(inv, flags,
//   callback,
//                                                           userData);
// }

void openxlaPartitionerInvocationEnableConsoleDiagnostics(
    openxla_partitioner_invocation_t *run) {
  assertLoaded();
  __openxlaPartitionerInvocationEnableConsoleDiagnostics(run);
}

void openxlaPartitionerInvocationDestroy(
    openxla_partitioner_invocation_t *run) {
  assertLoaded();
  __openxlaPartitionerInvocationDestroy(run);
}

void openxlaPartitionerInvocationSetCrashHandler(
    openxla_partitioner_invocation_t *inv, bool genLocalReproducer,
    openxla_partitioner_error_t *(*onCrashCallback)(
        openxla_partitioner_output_t **outOutput, void *userData),
    void *userData) {
  assertLoaded();
  __openxlaPartitionerInvocationSetCrashHandler(inv, genLocalReproducer,
                                                onCrashCallback, userData);
}

bool openxlaPartitionerInvocationParseSource(
    openxla_partitioner_invocation_t *run,
    openxla_partitioner_source_t *source) {
  assertLoaded();
  return __openxlaPartitionerInvocationParseSource(run, source);
}

void openxlaPartitionerInvocationSetVerifyIR(
    openxla_partitioner_invocation_t *run, bool enable) {
  assertLoaded();
  __openxlaPartitionerInvocationSetVerifyIR(run, enable);
}

bool openxlaPartitionerInvocationPipeline(openxla_partitioner_invocation_t *run,
                                          const char *pipeline) {
  assertLoaded();
  return __openxlaPartitionerInvocationPipeline(run, pipeline);
}

openxla_partitioner_error_t *openxlaPartitionerInvocationOutputIR(
    openxla_partitioner_invocation_t *run,
    openxla_partitioner_output_t *output) {
  assertLoaded();
  return __openxlaPartitionerInvocationOutputIR(run, output);
}

void openxlaPartitionerSourceDestroy(openxla_partitioner_source_t *source) {
  assertLoaded();
  __openxlaPartitionerSourceDestroy(source);
}

openxla_partitioner_error_t *openxlaPartitionerSourceOpenFile(
    openxla_partitioner_session_t *session, const char *filePath,
    openxla_partitioner_source_t **out_source) {
  assertLoaded();
  return __openxlaPartitionerSourceOpenFile(session, filePath, out_source);
}

openxla_partitioner_error_t *openxlaPartitionerSourceWrapBuffer(
    openxla_partitioner_session_t *session, const char *bufferName,
    const char *buffer, size_t length, bool isNullTerminated,
    openxla_partitioner_source_t **out_source) {
  assertLoaded();
  return __openxlaPartitionerSourceWrapBuffer(
      session, bufferName, buffer, length, isNullTerminated, out_source);
}

void openxlaPartitionerOutputDestroy(openxla_partitioner_output_t *output) {
  assertLoaded();
  __openxlaPartitionerOutputDestroy(output);
}

openxla_partitioner_error_t *openxlaPartitionerOutputOpenFile(
    const char *filePath, openxla_partitioner_output_t **out_output) {
  assertLoaded();
  return __openxlaPartitionerOutputOpenFile(filePath, out_output);
}

openxla_partitioner_error_t *openxlaPartitionerOutputOpenFD(
    int fd, openxla_partitioner_output_t **out_output) {
  assertLoaded();
  return __openxlaPartitionerOutputOpenFD(fd, out_output);
}

openxla_partitioner_error_t *openxlaPartitionerOutputOpenMembuffer(
    openxla_partitioner_output_t **out_output) {
  assertLoaded();
  return __openxlaPartitionerOutputOpenMembuffer(out_output);
}

void openxlaPartitionerOutputKeep(openxla_partitioner_output_t *output) {
  assertLoaded();
  __openxlaPartitionerOutputKeep(output);
}

openxla_partitioner_error_t *openxlaPartitionerOutputMapMemory(
    openxla_partitioner_output_t *output, void **contents, uint64_t *size) {
  assertLoaded();
  return __openxlaPartitionerOutputMapMemory(output, contents, size);
}

openxla_partitioner_error_t *openxlaPartitionerOutputWrite(
    openxla_partitioner_output_t *output, const void *data, size_t length) {
  assertLoaded();
  return __openxlaPartitionerOutputWrite(output, data, length);
}
