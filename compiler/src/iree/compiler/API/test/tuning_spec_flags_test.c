// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test that verifies tuning spec flags can be set via C API.
// Before the TuningSpecOptions refactoring, these flags would be rejected
// because they were not registered in the Session's local OptionsBinder.

#include <stdio.h>
#include <string.h>

#include "iree/compiler/embedding_api.h"

int main(int argc, char **argv) {
  ireeCompilerGlobalInitialize();
  iree_compiler_session_t *session = ireeCompilerSessionCreate();

  // Test setting all three tuning spec flags together.
  // This is the use case from iree-org/iree#23314.
  const char *flags[] = {"--iree-codegen-tuning-spec-path=/tmp/spec.mlir",
                         "--iree-codegen-enable-default-tuning-specs=true",
                         "--iree-codegen-dump-tuning-specs-to=/tmp/dump"};

  iree_compiler_error_t *err = ireeCompilerSessionSetFlags(session, 3, flags);
  if (err) {
    fprintf(stderr, "FAIL: Tuning spec flags not accepted: %s\n",
            ireeCompilerErrorGetMessage(err));
    ireeCompilerErrorDestroy(err);
    ireeCompilerSessionDestroy(session);
    ireeCompilerGlobalShutdown();
    return 1;
  }

  printf("PASS: All tuning spec flags accepted via C API\n");

  ireeCompilerSessionDestroy(session);
  ireeCompilerGlobalShutdown();
  return 0;
}
