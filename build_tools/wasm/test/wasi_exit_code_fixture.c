// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
  if (argc != 2) {
    return 2;
  }
  if (strcmp(argv[1], "success") == 0) {
    return 0;
  }
  if (strcmp(argv[1], "failure") == 0) {
    return 7;
  }
  if (strcmp(argv[1], "exit") == 0) {
    exit(7);
  }
  if (strcmp(argv[1], "trap") == 0) {
    __builtin_trap();
  }
  return 2;
}
