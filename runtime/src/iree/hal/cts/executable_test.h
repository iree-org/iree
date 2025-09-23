// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CTS_EXECUTABLE_TEST_H_
#define IREE_HAL_CTS_EXECUTABLE_TEST_H_

#include "iree/base/api.h"
#include "iree/base/string_view.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/cts_test_base.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::cts {

using iree::testing::status::StatusIs;

class ExecutableTest : public CTSTestBase<> {
 protected:
  virtual void SetUp() {
    IREE_ASSERT_OK(iree_hal_executable_cache_create(
        device_, iree_make_cstring_view("default"),
        iree_loop_inline(&loop_status_), &executable_cache_));

    iree_hal_executable_params_t executable_params;
    iree_hal_executable_params_initialize(&executable_params);
    executable_params.caching_mode =
        IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
    executable_params.executable_format =
        iree_make_cstring_view(get_test_executable_format());
    executable_params.executable_data =
        get_test_executable_data(iree_make_cstring_view("executable_test.bin"));

    iree_hal_executable_t* executable = NULL;
    IREE_ASSERT_OK(iree_hal_executable_cache_prepare_executable(
        executable_cache_, &executable_params, &executable_));

    IREE_ASSERT_OK(loop_status_);
  }

  virtual void TearDown() {
    iree_hal_executable_release(executable_);
    iree_hal_executable_cache_release(executable_cache_);
    iree_status_free(loop_status_);
    IREE_ASSERT_OK(loop_status_);
  }

  bool ExportHasParameterInfo(
      iree_hal_executable_export_ordinal_t export_ordinal) {
    iree_hal_executable_export_info_t info;
    IREE_CHECK_OK(
        iree_hal_executable_export_info(executable_, export_ordinal, &info));
    return info.parameter_count != 0;
  }

  iree_status_t loop_status_ = iree_ok_status();
  iree_hal_executable_cache_t* executable_cache_ = nullptr;
  iree_hal_executable_t* executable_ = nullptr;
};

TEST_F(ExecutableTest, ExportCount) {
  ASSERT_EQ(iree_hal_executable_export_count(executable_), 1);
}

TEST_F(ExecutableTest, ExportInfoOutOfRange) {
  iree_hal_executable_export_info_t info;
  EXPECT_THAT(Status(iree_hal_executable_export_info(executable_, 100, &info)),
              StatusIs(StatusCode::kOutOfRange));
}

TEST_F(ExecutableTest, ExportInfo) {
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

TEST_F(ExecutableTest, ExportParametersOutOfRange) {
  if (!ExportHasParameterInfo(0)) {
    GTEST_SKIP() << "parameter reflection not available";
  }

  iree_hal_executable_export_parameter_t parameters[64];
  EXPECT_THAT(Status(iree_hal_executable_export_parameters(
                  executable_, 100, IREE_ARRAYSIZE(parameters), parameters)),
              StatusIs(StatusCode::kOutOfRange));
}

// Tests that the parameter query clamps the returned count to the capacity.
TEST_F(ExecutableTest, ExportParametersNoCapacity) {
  if (!ExportHasParameterInfo(0)) {
    GTEST_SKIP() << "parameter reflection not available";
  }

  // We rely on ASAN to flag we if go over capacity.
  iree_hal_executable_export_parameter_t parameters[1];
  IREE_EXPECT_OK(iree_hal_executable_export_parameters(
      executable_, 0, /*capacity=*/1, parameters));
  EXPECT_EQ(parameters[0].type,
            IREE_HAL_EXECUTABLE_EXPORT_PARAMETER_TYPE_BINDING);
}

TEST_F(ExecutableTest, ExportParameters) {
  if (!ExportHasParameterInfo(0)) {
    GTEST_SKIP() << "parameter reflection not available";
  }

  iree_hal_executable_export_parameter_t parameters[64];
  IREE_ASSERT_OK(iree_hal_executable_export_parameters(
      executable_, 0, IREE_ARRAYSIZE(parameters), parameters));

  // TODO(benvanik): compiler mechanism for exporting parameter info. Today no
  // target supports populating parameter info so this is never hit.
}

TEST_F(ExecutableTest, LookupExportByNameNotFound) {
  iree_hal_executable_export_ordinal_t ordinal = 0;
  EXPECT_THAT(Status(iree_hal_executable_lookup_export_by_name(
                  executable_, IREE_SV("NOT_FOUND"), &ordinal)),
              StatusIs(StatusCode::kNotFound));
}

TEST_F(ExecutableTest, LookupExportByName) {
  iree_hal_executable_export_ordinal_t ordinal = 0;

  IREE_ASSERT_OK(iree_hal_executable_lookup_export_by_name(
      executable_, IREE_SV("export0"), &ordinal));
  EXPECT_EQ(ordinal, 0);
}

}  // namespace iree::hal::cts

#endif  // IREE_HAL_CTS_EXECUTABLE_TEST_H_
