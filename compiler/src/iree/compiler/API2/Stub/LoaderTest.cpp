// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/compiler/API2/Embed.h"
#include "iree/compiler/API2/Stub/Loader.h"

int main(int argc, char **argv) {
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
  auto *session = ireeCompilerSessionCreate();

  // Invocation.
  auto *inv = ireeCompilerInvocationCreate(session);
  ireeCompilerInvocationEnableConsoleDiagnostics(inv);
  ireeCompilerInvocationDestroy(inv);

  ireeCompilerSessionDestroy(session);

  ireeCompilerGlobalShutdown();
  printf("Shutdown\n");
  return 0;
}
