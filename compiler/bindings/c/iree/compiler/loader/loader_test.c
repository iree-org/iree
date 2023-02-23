// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/loader.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/compiler/embedding_api.h"

static int flagCount = 0;

static void printFlag(const char *flag, size_t length, void *userData) {
  printf("  FLAG: %.*s\n", (int)length, flag);
  flagCount += 1;
};

static bool manipulateFlags(iree_compiler_session_t *session) {
  iree_compiler_error_t *error;
  // Flags.
  printf("All flags:\n");
  ireeCompilerSessionGetFlags(session, /*nonDefaultOnly=*/false, printFlag,
                              /*userData=*/NULL);
  if (flagCount <= 1) {
    printf("FLAG GET ERROR. Abort\n");
    return 1;
  }
  const char *flag1 = "--iree-input-type=mhlo";
  const char *badFlag1 = "--iree-non-existing-flag=foobar";
  const char *flagArgv[] = {
      flag1,
  };
  error = ireeCompilerSessionSetFlags(session, 1, flagArgv);
  if (error) {
    printf("FLAG SET ERROR. Abort.\n");
    return 1;
  }

  // Read-back non-default flags.
  flagCount = 0;
  printf("Changed flags:\n");
  ireeCompilerSessionGetFlags(session, /*nonDefaultOnly=*/true, printFlag,
                              /*userData=*/NULL);
  if (flagCount != 1) {
    printf("Mismatched changed flags. Got %d, expected 1.\n", flagCount);
    return 1;
  }

  // Try to set a flag in error.
  flagArgv[0] = badFlag1;
  error = ireeCompilerSessionSetFlags(session, 1, flagArgv);
  if (!error) {
    printf("Expected error. Abort.\n");
    return 1;
  }
  const char *msg = ireeCompilerErrorGetMessage(error);
  printf("EXPECTED ERROR: %s\n", msg);
  ireeCompilerErrorDestroy(error);
  return true;
}

static bool invokeWithConsoleDiagnostics(
    iree_compiler_session_t *session,
    iree_compiler_source_t *sourceWithErrors) {
  bool rc;
  printf(
      "--- INVOKING WITH CONSOLE DIAGNOSTICS (console error expected) ---\n");
  iree_compiler_invocation_t *inv = ireeCompilerInvocationCreate(session);
  ireeCompilerInvocationEnableConsoleDiagnostics(inv);
  // Expected to fail - testing diagnostics.
  rc = !ireeCompilerInvocationParseSource(inv, sourceWithErrors);
  ireeCompilerInvocationDestroy(inv);
  return rc;
}

static char *callbackDiagMessage = NULL;

static void callbackDiag(enum iree_compiler_diagnostic_severity_t severity,
                         const char *message, size_t messageSize,
                         void *userData) {
  char **messageAccum = (char **)userData;
  printf("GOT DIAG CALLBACK: %.*s\n", (int)messageSize, message);
  size_t currentSize = *messageAccum ? strlen(*messageAccum) : 0;
  *messageAccum = realloc(*messageAccum, currentSize + messageSize + 2);
  (*messageAccum)[currentSize] = '\n';
  memcpy(*messageAccum + currentSize + 1, message, messageSize);
  (*messageAccum)[currentSize + 1 + messageSize] = 0;
}

static bool invokeWithCallbackDiagnostics(
    iree_compiler_session_t *session,
    iree_compiler_source_t *sourceWithErrors) {
  bool rc;
  printf(
      "--- INVOKING WITH CALLBACK DIAGNOSTICS (console error expected) ---\n");
  iree_compiler_invocation_t *inv = ireeCompilerInvocationCreate(session);
  ireeCompilerInvocationEnableCallbackDiagnostics(
      inv, /*flags=*/0, callbackDiag, &callbackDiagMessage);
  // Expected to fail - testing diagnostics.
  rc = !ireeCompilerInvocationParseSource(inv, sourceWithErrors);

  if (!callbackDiagMessage) {
    printf("ERROR: Did not produce any callback diagnostics\n");
    rc = false;
  } else {
    printf("CALLBACK MESSAGES:\n%s\n", callbackDiagMessage);
    free(callbackDiagMessage);
    callbackDiagMessage = NULL;
  }

  ireeCompilerInvocationDestroy(inv);
  return rc;
}

int main(int argc, char **argv) {
  iree_compiler_error_t *error;
  if (argc < 2) {
    printf("ERROR: Requires library path\n");
    return 1;
  }

  if (!ireeCompilerLoadLibrary(argv[1])) {
    printf("ERROR: Could not load library\n");
    return 1;
  }
  printf("Library loaded: %s\n", argv[1]);

  int version = ireeCompilerGetAPIVersion();
  printf("Version: %d\n", version);

  ireeCompilerGlobalInitialize(true);
  printf("Initialized\n");

  // Session.
  iree_compiler_session_t *session = ireeCompilerSessionCreate();

  // Define sources that produce errors.
  iree_compiler_source_t *sourceWithErrors;
  const char sourceWithErrorsStr[] = "}}}}FOOBAR\0";
  error = ireeCompilerSourceWrapBuffer(
      session, "foobar.mlir", sourceWithErrorsStr, sizeof(sourceWithErrorsStr),
      /*isNullTerminated=*/true, &sourceWithErrors);
  if (error) {
    printf("ERROR: ireeCompilerSourceWrapBuffer failed: %s\n",
           ireeCompilerErrorGetMessage(error));
    return 1;
  }

  if (!manipulateFlags(session)) {
    return 1;
  }

  if (!invokeWithConsoleDiagnostics(session, sourceWithErrors)) {
    return 1;
  }

  if (!invokeWithCallbackDiagnostics(session, sourceWithErrors)) {
    return 1;
  }

  ireeCompilerSourceDestroy(sourceWithErrors);
  ireeCompilerSessionDestroy(session);

  ireeCompilerGlobalShutdown();
  printf("Shutdown\n");
  return 0;
}
