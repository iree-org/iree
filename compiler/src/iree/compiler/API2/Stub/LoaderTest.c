// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdbool.h>
#include <stdio.h>

#include "iree/compiler/API2/Embed.h"
#include "iree/compiler/API2/Stub/Loader.h"

static int flagCount = 0;

void printFlag(const char *flag, size_t length, void *userData) {
  printf("  FLAG: %.*s\n", (int)length, flag);
  flagCount += 1;
};

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

  // Invocation.
  iree_compiler_invocation_t *inv = ireeCompilerInvocationCreate(session);
  ireeCompilerInvocationEnableConsoleDiagnostics(inv);
  ireeCompilerInvocationDestroy(inv);

  ireeCompilerSessionDestroy(session);

  ireeCompilerGlobalShutdown();
  printf("Shutdown\n");
  return 0;
}
