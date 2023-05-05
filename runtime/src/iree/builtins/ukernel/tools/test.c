// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/tools/test.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/schemas/cpu_data.h"

typedef enum {
  IREE_UK_TEST_STATUS_RUN,
  IREE_UK_TEST_STATUS_OK,
  IREE_UK_TEST_STATUS_FAILED,
  IREE_UK_TEST_STATUS_SKIPPED,
} iree_uk_test_status_t;

struct iree_uk_test_t {
  const char* name;
  const char* cpu_features;
  iree_uk_uint64_t cpu_data[IREE_CPU_DATA_FIELD_COUNT];
  iree_time_t time_start;
  iree_uk_random_engine_t random_engine;
};

iree_uk_random_engine_t* iree_uk_test_random_engine(
    const iree_uk_test_t* test) {
  // Cast constness away, i.e. consider random engine state mutation as not
  // really a test state mutation.
  return (iree_uk_random_engine_t*)&test->random_engine;
}

const iree_uk_uint64_t* iree_uk_test_cpu_data(const iree_uk_test_t* test) {
  return test->cpu_data;
}

const char* iree_uk_test_status_header(iree_uk_test_status_t status) {
  switch (status) {
    case IREE_UK_TEST_STATUS_RUN:
      return "[ RUN      ] 🎲";
    case IREE_UK_TEST_STATUS_OK:
      return "[       OK ] ✅";
    case IREE_UK_TEST_STATUS_FAILED:
      return "[   FAILED ] ❌";
    case IREE_UK_TEST_STATUS_SKIPPED:
      return "[  SKIPPED ] 🙈";
    default:
      IREE_UK_ASSERT(false);
      return "";
  }
}

static void iree_uk_test_log_status(const iree_uk_test_t* test,
                                    iree_uk_test_status_t status) {
  fprintf(stderr, "%s %s", iree_uk_test_status_header(status), test->name);
  if (test->cpu_features) {
    fprintf(stderr, ", cpu_features:%s", test->cpu_features);
  }
  if (status != IREE_UK_TEST_STATUS_RUN) {
    fprintf(stderr, " (%" PRIi64 " ms)",
            (iree_time_now() - test->time_start) / (1000 * 1000));
  }
  fprintf(stderr, "\n");
}

static void iree_uk_test_log_info(const iree_uk_test_t* test, const char* emoji,
                                  const char* msg) {
  fprintf(stderr, "[   INFO   ] %s %s\n", emoji, msg);
}

static void iree_uk_test_log_error(const iree_uk_test_t* test,
                                   const char* msg) {
  fprintf(stderr, "[   ERROR  ] ❌ %s\n", msg);
}

void iree_uk_test(const char* name,
                  void (*test_func)(iree_uk_test_t*, const void*),
                  const void* params, const char* cpu_features) {
  iree_uk_test_t test = {
      .name = name,
      .cpu_features = cpu_features,
      .time_start = iree_time_now(),
      // Letting each test create its own engine makes them independent: a
      // testcase succeeds or fails the same way if we isolate it or reorder it.
      // The potential downside of repeating the same pseudorandom sequence is
      // OK because any pseudorandom sequence should be equally good at
      // coverage, and different testcases tend to use different tile shapes
      // anyway.
      .random_engine = iree_uk_random_engine_init(),
  };
  iree_uk_test_log_status(&test, IREE_UK_TEST_STATUS_RUN);
  // First try without any optional CPU feature (test.cpu_data is still zeros).
  // This matters even when the feature is supported by the CPU because we want
  // to test the fallback to the architecture-default code path.
  test_func(&test, params);
  // We might skip the actual test payload requiring CPU features if these are
  // not supported.
  bool skipped = false;
  if (cpu_features) {
    iree_uk_initialize_cpu_once();
    iree_uk_make_cpu_data_for_features(cpu_features, test.cpu_data);
    if (iree_uk_cpu_supports(test.cpu_data)) {
      iree_uk_test_log_info(&test, "🚀", "CPU supports required features");
      test_func(&test, params);
    } else {
      skipped = true;
      char msg[128];
      snprintf(msg, sizeof msg, "CPU does not support required feature %s",
               iree_uk_cpu_first_unsupported_feature(test.cpu_data));
      iree_uk_test_log_info(&test, "🦕", msg);
    }
  }
  // Since errors are fatal (see iree_uk_test_fail), if we reached this point,
  // we know the test didn't fail.
  iree_uk_test_log_status(
      &test, skipped ? IREE_UK_TEST_STATUS_SKIPPED : IREE_UK_TEST_STATUS_OK);
}

void iree_uk_test_fail(const iree_uk_test_t* test, const char* file, int line) {
  char msg_buf[256];
  snprintf(msg_buf, sizeof msg_buf, "Error occurred at %s:%d", file, line);
  iree_uk_test_log_error(test, msg_buf);
  iree_uk_test_log_status(test, IREE_UK_TEST_STATUS_FAILED);
  // Error are always fatal for now. Works well for debugging (fail at the
  // root cause), and easiest to implement.
  iree_abort();
}
