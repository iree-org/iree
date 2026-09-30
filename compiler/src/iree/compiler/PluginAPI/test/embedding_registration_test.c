// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/compiler/embedding_api.h"
#include "iree/compiler/loader.h"

static void check_plugin(const char *id, void *user_data) {
  if (strcmp(id, "registration_failure") == 0) {
    *(bool *)user_data = true;
  }
}

int main(int argc, char **argv) {
  if (argc != 3 || setenv("IREE_LOAD_PLUGINS", argv[2], 1) != 0 ||
      !ireeCompilerLoadLibrary(argv[1])) {
    return EXIT_FAILURE;
  }
  ireeCompilerGlobalInitialize();
  bool found_failure = false;
  ireeCompilerEnumeratePlugins(check_plugin, &found_failure);
  iree_compiler_session_t *session = ireeCompilerSessionCreate();
  ireeCompilerSessionDestroy(session);
  ireeCompilerGlobalShutdown();
  if (found_failure) {
    fprintf(stderr, "Failed plugin was advertised as registered\n");
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
