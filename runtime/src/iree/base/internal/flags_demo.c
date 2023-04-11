// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"

IREE_FLAG(bool, test_bool, false, "A boolean value.");
IREE_FLAG(int32_t, test_int32, 123, "An int32_t value.");
IREE_FLAG(int64_t, test_int64, 555, "An int64_t value.");
IREE_FLAG(float, test_float, 1.0f, "A float value.");
IREE_FLAG(string, test_string, "some default", "A string\nvalue.");

static iree_status_t parse_callback(iree_string_view_t flag_name, void* storage,
                                    iree_string_view_t value) {
  int* count_ptr = (int*)storage;
  if (strcmp(value.data, "FORCE_FAILURE") == 0) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "callbacks can do verification");
  }
  *count_ptr += atoi(value.data);
  return iree_ok_status();
}
static void print_callback(iree_string_view_t flag_name, void* storage,
                           FILE* file) {
  int* count_ptr = (int*)storage;
  fprintf(file, "--%.*s=%d\n", (int)flag_name.size, flag_name.data, *count_ptr);
}
static int callback_count = 0;
IREE_FLAG_CALLBACK(parse_callback, print_callback, &callback_count,
                   test_callback, "Callback!");

IREE_FLAG_LIST(string, test_strings, "repeated");

int main(int argc, char** argv) {
  // Parse flags, updating argc/argv with position arguments.
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  // Report parsed flag values:
  printf("FLAG[test_bool] = %s\n", FLAG_test_bool ? "true" : "false");
  printf("FLAG[test_int32] = %" PRId32 "\n", FLAG_test_int32);
  printf("FLAG[test_int64] = %" PRId64 "\n", FLAG_test_int64);
  printf("FLAG[test_float] = %g\n", FLAG_test_float);
  printf("FLAG[test_string] = %s\n", FLAG_test_string);
  printf("FLAG[test_callback] = %d\n", callback_count);

  iree_flag_string_list_t strings = FLAG_test_strings_list();
  printf("FLAG[test_strings] = %" PRIhsz ": ", strings.count);
  for (iree_host_size_t i = 0; i < strings.count; ++i) {
    if (i > 0) printf(", ");
    printf("%.*s", (int)strings.values[i].size, strings.values[i].data);
  }
  printf("\n");

  // Report positional arguments:
  for (int i = 0; i < argc; ++i) {
    printf("ARG(%d) = %s\n", i, argv[i]);
  }

  // Dump all flags back out for round-tripping:
  iree_flags_dump(IREE_FLAG_DUMP_MODE_DEFAULT, stdout);

  return 0;
}
