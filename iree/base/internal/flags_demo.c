// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

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

  // Report positional arguments:
  for (int i = 0; i < argc; ++i) {
    printf("ARG(%d) = %s\n", i, argv[i]);
  }

  // Dump all flags back out for round-tripping:
  iree_flags_dump(IREE_FLAG_DUMP_MODE_DEFAULT, stdout);

  return 0;
}
