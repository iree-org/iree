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
  iree_uk_test_status_t status;
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

// Returns the log line header appropriate for the given status.
static const char* iree_uk_test_status_header(iree_uk_test_status_t status) {
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

// Log the test's current status.
static void iree_uk_test_log_status(const iree_uk_test_t* test) {
  fprintf(stderr, "%s %s", iree_uk_test_status_header(test->status),
          test->name);
  if (strlen(test->cpu_features)) {
    fprintf(stderr, " cpu_features:%s", test->cpu_features);
  }
  if (test->status != IREE_UK_TEST_STATUS_RUN) {
    fprintf(stderr, " (%" PRIi64 " ms)",
            (iree_time_now() - test->time_start) / (1000 * 1000));
  }
  fprintf(stderr, "\n");
}

// Log an info message.
static void iree_uk_test_log_info(const iree_uk_test_t* test, const char* emoji,
                                  const char* msg) {
  fprintf(stderr, "[   INFO   ] %s %s\n", emoji, msg);
}

// Log an error message.
static void iree_uk_test_log_error(const iree_uk_test_t* test,
                                   const char* msg) {
  fprintf(stderr, "[   ERROR  ] ❌ %s\n", msg);
}

// Tracks whether iree_uk_test_exit_status has been called.
static bool global_iree_uk_test_exit_status_called = false;

// atexit handler. Checks that iree_uk_test_exit_status has been called.
static void iree_uk_test_check_test_exit_status_called(void) {
  if (!global_iree_uk_test_exit_status_called) {
    fprintf(stderr, "Fatal: iree_uk_test_exit_status has not been called.\n");
    iree_abort();
  }
}

// Sets the atexit handler.
static void iree_uk_test_set_atexit(void) {
  atexit(iree_uk_test_check_test_exit_status_called);
}

// Global variables tracking counts of run/skipped/failed tests.
static int global_iree_uk_test_run_count = 0;
static int global_iree_uk_test_skipped_count = 0;
static int global_iree_uk_test_failed_count = 0;

void iree_uk_test(const char* name,
                  void (*test_func)(iree_uk_test_t*, const void*),
                  const void* params, const char* cpu_features) {
  // The first iree_uk_test sets the atexit handler.
  if (global_iree_uk_test_run_count == 0) {
    iree_uk_test_set_atexit();
  }
  ++global_iree_uk_test_run_count;
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
      .status = IREE_UK_TEST_STATUS_RUN,
  };
  iree_uk_test_log_status(&test);
  // First try without any optional CPU feature (test.cpu_data is still zeros).
  // This matters even when the feature is supported by the CPU because we want
  // to test the fallback to the architecture-default code path.
  test_func(&test, params);
  // Then try with optional CPU features, if specified.
  if (strlen(cpu_features)) {
    // Are specified CPU features supported by the CPU?
    iree_uk_initialize_cpu_once();
    iree_uk_make_cpu_data_for_features(cpu_features, test.cpu_data);
    if (iree_uk_cpu_supports(test.cpu_data)) {
      // CPU supports features. Run this part of the test.
      iree_uk_test_log_info(&test, "🚀", "CPU supports required features");
      test_func(&test, params);
    } else {
      // CPU does not support features. Skip this part of the test.
      char msg[128];
      snprintf(msg, sizeof msg, "CPU does not support required feature %s",
               iree_uk_cpu_first_unsupported_feature(test.cpu_data));
      iree_uk_test_log_info(&test, "🦕", msg);
      // Set test status to SKIPPED if it was still the initial RUN.
      // Do not overwrite a FAILED from the run without optional CPU features.
      if (test.status == IREE_UK_TEST_STATUS_RUN) {
        test.status = IREE_UK_TEST_STATUS_SKIPPED;
      }
    }
  }
  if (test.status == IREE_UK_TEST_STATUS_FAILED) {
    ++global_iree_uk_test_failed_count;
  } else if (test.status == IREE_UK_TEST_STATUS_SKIPPED) {
    ++global_iree_uk_test_skipped_count;
  } else {
    test.status = IREE_UK_TEST_STATUS_OK;
  }
  iree_uk_test_log_status(&test);
}

static const char iree_uk_test_abort_on_error_env[] =
    "IREE_UK_TEST_ABORT_ON_ERROR";

void iree_uk_test_fail(iree_uk_test_t* test, const char* file, int line) {
  test->status = IREE_UK_TEST_STATUS_FAILED;
  char msg_buf[256];
  snprintf(msg_buf, sizeof msg_buf, "Error occurred at %s:%d", file, line);
  iree_uk_test_log_error(test, msg_buf);
  if (getenv(iree_uk_test_abort_on_error_env)) {
    iree_abort();
  }
}

int iree_uk_test_exit_status(void) {
  global_iree_uk_test_exit_status_called = true;
  fprintf(stderr, "\nSummary: %d tests run, %d failed, %d skipped.\n",
          global_iree_uk_test_run_count, global_iree_uk_test_failed_count,
          global_iree_uk_test_skipped_count);
  if (!global_iree_uk_test_run_count) {
    fprintf(stderr, "Error: 0 tests run, is that normal?!\n");
    return EXIT_FAILURE;
  }
  if (global_iree_uk_test_failed_count) {
    fprintf(stderr,
            "To make errors fatal, define the %s environment variable.\n",
            iree_uk_test_abort_on_error_env);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
