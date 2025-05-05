// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hip/dynamic_symbols.h"

#include <iostream>

#include "iree/base/api.h"
#include "iree/hal/drivers/hip/rccl_dynamic_symbols.h"
#include "iree/testing/gtest.h"

namespace iree {
namespace hal {
namespace hip {
namespace {

#define HIP_CHECK_ERRORS(expr)     \
  {                                \
    hipError_t status = expr;      \
    ASSERT_EQ(hipSuccess, status); \
  }

TEST(DynamicSymbolsTest, CreateFromSystemLoader) {
  iree_hal_hip_dynamic_symbols_t symbols;
  iree_status_t status = iree_hal_hip_dynamic_symbols_initialize(
      iree_allocator_default(), /*hip_lib_search_path_count=*/0,
      /*hip_lib_search_paths=*/NULL, &symbols);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    GTEST_SKIP() << "Symbols cannot be loaded, skipping test.";
  }

  int device_count = 0;
  HIP_CHECK_ERRORS(symbols.hipInit(0));
  HIP_CHECK_ERRORS(symbols.hipGetDeviceCount(&device_count));
  if (device_count > 0) {
    hipDevice_t device;
    HIP_CHECK_ERRORS(symbols.hipDeviceGet(&device, /*ordinal=*/0));
  }

  iree_hal_hip_dynamic_symbols_deinitialize(&symbols);
}

static const iree_string_view_t non_existing_search_paths[] = {
    iree_make_cstring_view("/path/that/does/not/exist"),
    iree_make_cstring_view("file:nowhere/libamdhip64.so"),
    iree_make_cstring_view("filename_that_does_not_exist.dll"),
};

TEST(DynamicSymbolsTest, SearchPathsFail) {
  iree_hal_hip_dynamic_symbols_t symbols;
  iree_status_t status = iree_hal_hip_dynamic_symbols_initialize(
      iree_allocator_default(),
      /*hip_lib_search_path_count=*/IREE_ARRAYSIZE(non_existing_search_paths),
      non_existing_search_paths, &symbols);

  ASSERT_TRUE(iree_status_is_unavailable(status));
}

#define NCCL_CHECK_ERRORS(expr)     \
  {                                 \
    ncclResult_t status = expr;     \
    ASSERT_EQ(ncclSuccess, status); \
  }

TEST(NCCLDynamicSymbolsTest, CreateFromSystemLoader) {
  iree_hal_hip_dynamic_symbols_t hip_symbols;
  iree_status_t status = iree_hal_hip_dynamic_symbols_initialize(
      iree_allocator_default(), /*hip_lib_search_path_count=*/0,
      /*hip_lib_search_paths=*/NULL, &hip_symbols);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    GTEST_SKIP() << "HIP symbols cannot be loaded, skipping test.";
  }

  iree_hal_hip_nccl_dynamic_symbols_t nccl_symbols;
  status = iree_hal_hip_nccl_dynamic_symbols_initialize(
      iree_allocator_default(), &hip_symbols, &nccl_symbols);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    GTEST_SKIP() << "HIP RCCL symbols cannot be loaded, skipping test.";
  }

  // Check that the loaded version is at least the version we compiled for.
  int nccl_version = 0;
  NCCL_CHECK_ERRORS(nccl_symbols.ncclGetVersion(&nccl_version));
  ASSERT_GE(nccl_version, NCCL_VERSION_CODE);

  iree_hal_hip_nccl_dynamic_symbols_deinitialize(&nccl_symbols);
  iree_hal_hip_dynamic_symbols_deinitialize(&hip_symbols);
}

}  // namespace
}  // namespace hip
}  // namespace hal
}  // namespace iree
