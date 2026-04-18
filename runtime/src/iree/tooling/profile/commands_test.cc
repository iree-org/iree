// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/commands.h"

#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static iree_profile_command_options_t DefaultOptions() {
  iree_profile_command_options_t options;
  memset(&options, 0, sizeof(options));
  options.format = IREE_SV("text");
  options.filter = IREE_SV("*");
  options.output_path = IREE_SV("-");
  options.id = -1;
  options.host_allocator = iree_allocator_system();
  return options;
}

TEST(ProfileCommandValidationTest, AcceptsReportFormats) {
  iree_profile_command_options_t options = DefaultOptions();

  IREE_ASSERT_OK(iree_profile_command_validate_options(
      iree_profile_summary_command(), &options));

  options.format = IREE_SV("jsonl");
  IREE_ASSERT_OK(iree_profile_command_validate_options(
      iree_profile_summary_command(), &options));
}

TEST(ProfileCommandValidationTest, ExportUsesInterchangeFormat) {
  iree_profile_command_options_t options = DefaultOptions();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_profile_command_validate_options(
                            iree_profile_export_command(), &options));

  options.format = IREE_SV("ireeperf-jsonl");
  options.output_path = IREE_SV("profile.ireeperf.jsonl");
  IREE_ASSERT_OK(iree_profile_command_validate_options(
      iree_profile_export_command(), &options));
}

TEST(ProfileCommandValidationTest, RejectsOptionsUnusedByCommand) {
  iree_profile_command_options_t options = DefaultOptions();
  options.filter = IREE_SV("copy");
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_profile_command_validate_options(
                            iree_profile_summary_command(), &options));

  options = DefaultOptions();
  options.output_path = IREE_SV("profile.ireeperf.jsonl");
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_profile_command_validate_options(
                            iree_profile_cat_command(), &options));

  options = DefaultOptions();
  options.rocm_library_path = IREE_SV("/opt/rocm/lib");
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_profile_command_validate_options(
                            iree_profile_dispatch_command(), &options));
}

TEST(ProfileCommandValidationTest, DetailRowsRequireJsonl) {
  iree_profile_command_options_t options = DefaultOptions();
  options.emit_dispatch_events = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_profile_command_validate_options(
                            iree_profile_dispatch_command(), &options));

  options.format = IREE_SV("jsonl");
  IREE_ASSERT_OK(iree_profile_command_validate_options(
      iree_profile_dispatch_command(), &options));

  options = DefaultOptions();
  options.emit_counter_samples = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_profile_command_validate_options(
                            iree_profile_counter_command(), &options));

  options.format = IREE_SV("jsonl");
  IREE_ASSERT_OK(iree_profile_command_validate_options(
      iree_profile_counter_command(), &options));
}

TEST(ProfileCommandValidationTest, RejectsDetailRowsOnWrongCommandFamily) {
  iree_profile_command_options_t options = DefaultOptions();
  options.format = IREE_SV("jsonl");
  options.emit_dispatch_events = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_profile_command_validate_options(
                            iree_profile_counter_command(), &options));

  options = DefaultOptions();
  options.format = IREE_SV("jsonl");
  options.emit_counter_samples = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_profile_command_validate_options(
                            iree_profile_dispatch_command(), &options));
}

TEST(ProfileCommandValidationTest, FindCommandUsesProvidedRegistry) {
  const iree_profile_command_t* const commands[] = {
      iree_profile_summary_command(),
      iree_profile_cat_command(),
  };

  EXPECT_EQ(iree_profile_summary_command(),
            iree_profile_find_command(commands, IREE_ARRAYSIZE(commands),
                                      IREE_SV("summary")));
  EXPECT_EQ(nullptr, iree_profile_find_command(
                         commands, IREE_ARRAYSIZE(commands), IREE_SV("att")));
}

}  // namespace
