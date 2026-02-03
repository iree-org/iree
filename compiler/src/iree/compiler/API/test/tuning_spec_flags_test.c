// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test that verifies the tuning spec path flag can be set via C API.
// The tuning spec path is now registered as part of ROCMOptions but uses
// the codegen prefix for consistency. Other tuning spec flags
// (enable-default-tuning-specs, dump-tuning-specs-to) remain CLI-only.

#include <stdio.h>
#include "iree/compiler/embedding_api.h"

int main(int argc, char **argv) {
  ireeCompilerGlobalInitialize();
  iree_compiler_session_t *session = ireeCompilerSessionCreate();

  const char *flags[] = {"--iree-codegen-tuning-spec-path=/tmp/spec.mlir"};

  iree_compiler_error_t *err = ireeCompilerSessionSetFlags(session, 1, flags);
  if (err) {
    fprintf(stderr, "FAIL: Tuning spec path flag not accepted: %s\n",
            ireeCompilerErrorGetMessage(err));
    ireeCompilerErrorDestroy(err);
    ireeCompilerSessionDestroy(session);
    ireeCompilerGlobalShutdown();
    return 1;
  }

  printf("PASS: Tuning spec path flag accepted via C API\n");

  ireeCompilerSessionDestroy(session);
  ireeCompilerGlobalShutdown();
  return 0;
}
