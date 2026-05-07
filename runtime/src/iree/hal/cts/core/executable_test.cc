// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for HAL executable creation, export metadata, and reflection.

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

using iree::testing::status::StatusIs;
using ::testing::AnyOf;

class ExecutableTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase::SetUp();
    if (HasFatalFailure() || IsSkipped()) return;

    IREE_ASSERT_OK(iree_hal_executable_cache_create(
        device_, iree_make_cstring_view("default"), &executable_cache_));

    iree_hal_executable_params_t executable_params;
    iree_hal_executable_params_initialize(&executable_params);
    executable_params.caching_mode =
        IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
    executable_params.executable_format =
        iree_make_cstring_view(executable_format());
    executable_params.executable_data =
        executable_data(iree_make_cstring_view("executable_test.bin"));

    IREE_ASSERT_OK(iree_hal_executable_cache_prepare_executable(
        executable_cache_, &executable_params, &executable_));
  }

  void TearDown() override {
    if (executable_) {
      iree_hal_executable_release(executable_);
      executable_ = nullptr;
    }
    if (executable_cache_) {
      iree_hal_executable_cache_release(executable_cache_);
      executable_cache_ = nullptr;
    }
    CtsTestBase::TearDown();
  }

  // Probes whether export_info is available and reports parameter metadata.
  // Returns false without recording a test failure when the backend does not
  // implement reflection — callers use this to GTEST_SKIP().
  bool ExportHasParameterInfo(
      iree_hal_executable_export_ordinal_t export_ordinal) {
    iree_hal_executable_export_info_t info;
    iree_status_t status =
        iree_hal_executable_export_info(executable_, export_ordinal, &info);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return false;
    }
    return info.parameter_count != 0;
  }

  iree_hal_executable_cache_t* executable_cache_ = nullptr;
  iree_hal_executable_t* executable_ = nullptr;
};

TEST_P(ExecutableTest, ExportCount) {
  ASSERT_EQ(iree_hal_executable_export_count(executable_), 1);
}

TEST_P(ExecutableTest, ExportInfoOutOfRange) {
  iree_hal_executable_export_info_t info;
  EXPECT_THAT(Status(iree_hal_executable_export_info(executable_, 100, &info)),
              StatusIs(StatusCode::kOutOfRange));
}

TEST_P(ExecutableTest, ExportInfo) {
  iree_hal_executable_export_info_t info;

  // export0: #hal.pipeline.layout<constants = 2, bindings = [
  //   #hal.pipeline.binding<storage_buffer>,
  //   #hal.pipeline.binding<storage_buffer>
  // ]>
  IREE_ASSERT_OK(iree_hal_executable_export_info(executable_, 0, &info));
  EXPECT_EQ(std::string_view(info.name.data, info.name.size), "export0");
  EXPECT_EQ(info.flags, IREE_HAL_EXECUTABLE_EXPORT_FLAG_NONE);
  EXPECT_EQ(info.constant_count, 2);
  EXPECT_EQ(info.binding_count, 2);
}

TEST_P(ExecutableTest, ExportParametersOutOfRange) {
  if (!ExportHasParameterInfo(0)) {
    GTEST_SKIP() << "parameter reflection not available";
  }

  iree_hal_executable_export_parameter_t parameters[64];
  EXPECT_THAT(Status(iree_hal_executable_export_parameters(
                  executable_, 100, IREE_ARRAYSIZE(parameters), parameters)),
              StatusIs(StatusCode::kOutOfRange));
}

TEST_P(ExecutableTest, ExportParametersNoCapacity) {
  if (!ExportHasParameterInfo(0)) {
    GTEST_SKIP() << "parameter reflection not available";
  }

  iree_hal_executable_export_parameter_t parameters[1];
  IREE_EXPECT_OK(iree_hal_executable_export_parameters(
      executable_, 0, /*capacity=*/1, parameters));
  EXPECT_EQ(parameters[0].type,
            IREE_HAL_EXECUTABLE_EXPORT_PARAMETER_TYPE_BINDING);
}

TEST_P(ExecutableTest, ExportParameters) {
  if (!ExportHasParameterInfo(0)) {
    GTEST_SKIP() << "parameter reflection not available";
  }

  iree_hal_executable_export_parameter_t parameters[64];
  IREE_ASSERT_OK(iree_hal_executable_export_parameters(
      executable_, 0, IREE_ARRAYSIZE(parameters), parameters));
}

TEST_P(ExecutableTest, LookupExportByNameNotFound) {
  iree_hal_executable_export_ordinal_t ordinal = 0;
  EXPECT_THAT(Status(iree_hal_executable_lookup_export_by_name(
                  executable_, IREE_SV("NOT_FOUND"), &ordinal)),
              StatusIs(StatusCode::kNotFound));
}

TEST_P(ExecutableTest, LookupExportByName) {
  iree_hal_executable_export_ordinal_t ordinal = 0;

  IREE_ASSERT_OK(iree_hal_executable_lookup_export_by_name(
      executable_, IREE_SV("export0"), &ordinal));
  EXPECT_EQ(ordinal, 0);
}

TEST_P(ExecutableTest, LookupGlobalByNameNotFoundOrUnsupported) {
  iree_hal_buffer_t* buffer = nullptr;
  EXPECT_THAT(Status(iree_hal_executable_lookup_global_by_name(
                  executable_, IREE_SV("NOT_FOUND"),
                  IREE_HAL_QUEUE_AFFINITY_ANY, &buffer)),
              AnyOf(StatusIs(StatusCode::kNotFound),
                    StatusIs(StatusCode::kUnimplemented)));
  EXPECT_EQ(buffer, nullptr);
}

CTS_REGISTER_EXECUTABLE_TEST_SUITE(ExecutableTest);

}  // namespace iree::hal::cts
