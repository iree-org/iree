// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/loaders/static_library_loader.h"

#include <cstring>

#include "iree/hal/local/executable_library.h"
#include "iree/hal/local/local_executable.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Fails with a code a status-returning dispatch would propagate.
int ReturnUnavailable(const iree_hal_executable_environment_v0_t* environment,
                      const iree_hal_executable_dispatch_state_v0_t* dispatch,
                      const iree_hal_executable_workgroup_state_v0_t* state) {
  return IREE_STATUS_UNAVAILABLE;
}

// A one-export library of version `kVersion` whose export sets
// RETURNS_STATUS. With `kExactQuery` it answers only a query for its own
// version, as libraries built for exactly v0.6 do.
template <iree_hal_executable_library_version_t kVersion, bool kExactQuery>
const iree_hal_executable_library_header_t** QueryLibrary(
    iree_hal_executable_library_version_t max_version,
    const iree_hal_executable_environment_v0_t* environment) {
  static const iree_hal_executable_library_header_t header = {
      kVersion, "versioned", IREE_HAL_EXECUTABLE_LIBRARY_FEATURE_NONE,
      IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_NONE};
  static const iree_hal_executable_dispatch_v0_t ptrs[1] = {ReturnUnavailable};
  static const iree_hal_executable_dispatch_attrs_v0_t* attrs = [] {
    static iree_hal_executable_dispatch_attrs_v0_t value[1];
    std::memset(value, 0, sizeof(value));
    value[0].flags = IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_RETURNS_STATUS;
    return value;
  }();
  static const char* names[1] = {"fail"};
  static const iree_hal_executable_library_v0_t* library = [] {
    static iree_hal_executable_library_v0_t value;
    std::memset(&value, 0, sizeof(value));
    value.header = &header;
    value.exports.count = 1;
    value.exports.ptrs = ptrs;
    value.exports.attrs = attrs;
    value.exports.names = names;
    return &value;
  }();
  if (kExactQuery ? max_version != kVersion : max_version < kVersion) {
    return nullptr;
  }
  return (const iree_hal_executable_library_header_t**)library;
}

iree_status_t CreateLoader(iree_hal_executable_library_query_fn_t query,
                           iree_hal_executable_loader_t** out_loader) {
  return iree_hal_static_library_loader_create(
      /*library_count=*/1, &query, iree_hal_executable_import_provider_null(),
      iree_allocator_system(), out_loader);
}

// Loads the library and calls its export once.
iree_status_t LoadAndCall(iree_hal_executable_library_query_fn_t query) {
  iree_hal_executable_loader_t* loader = nullptr;
  IREE_RETURN_IF_ERROR(CreateLoader(query, &loader));
  iree_hal_executable_params_t params;
  iree_hal_executable_params_initialize(&params);
  params.caching_mode = IREE_HAL_EXECUTABLE_CACHING_MODE_DISABLE_VERIFICATION;
  params.executable_format = iree_make_cstring_view("static");
  static const char kName[] = "versioned";
  params.executable_data =
      iree_make_const_byte_span((const uint8_t*)kName, sizeof(kName));
  iree_hal_executable_t* executable = nullptr;
  iree_status_t status = iree_hal_executable_loader_try_load(
      loader, &params, /*worker_capacity=*/1, &executable);
  if (iree_status_is_ok(status)) {
    iree_hal_executable_dispatch_state_v0_t dispatch_state;
    std::memset(&dispatch_state, 0, sizeof(dispatch_state));
    iree_hal_executable_workgroup_state_v0_t workgroup_state;
    std::memset(&workgroup_state, 0, sizeof(workgroup_state));
    status = iree_hal_local_executable_issue_call(
        iree_hal_local_executable_cast(executable), /*ordinal=*/0,
        &dispatch_state, &workgroup_state, /*worker_id=*/0);
  }
  iree_hal_executable_release(executable);
  iree_hal_executable_loader_release(loader);
  return status;
}

TEST(StaticLibraryLoaderTest, AcceptsSupportedVersions) {
  iree_hal_executable_loader_t* loader = nullptr;
  IREE_ASSERT_OK(CreateLoader(
      QueryLibrary<IREE_HAL_EXECUTABLE_LIBRARY_VERSION_0_6, false>, &loader));
  iree_hal_executable_loader_release(loader);
  IREE_ASSERT_OK(CreateLoader(
      QueryLibrary<IREE_HAL_EXECUTABLE_LIBRARY_VERSION_0_7, false>, &loader));
  iree_hal_executable_loader_release(loader);
}

// A v0.5 library lays out and interprets dispatches differently.
TEST(StaticLibraryLoaderTest, RejectsOlderVersions) {
  iree_hal_executable_loader_t* loader = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      CreateLoader(
          QueryLibrary<IREE_HAL_EXECUTABLE_LIBRARY_VERSION_0_6 - 1, false>,
          &loader));
  EXPECT_EQ(loader, nullptr);
}

// A library that needs a newer runtime answers neither supported query.
TEST(StaticLibraryLoaderTest, RejectsNewerVersions) {
  iree_hal_executable_loader_t* loader = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      CreateLoader(
          QueryLibrary<IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST + 1, false>,
          &loader));
  EXPECT_EQ(loader, nullptr);
}

// A library built for exactly v0.6 answers only a v0.6 query.
TEST(StaticLibraryLoaderTest, RetriesExactV06Query) {
  iree_hal_executable_loader_t* loader = nullptr;
  IREE_ASSERT_OK(CreateLoader(
      QueryLibrary<IREE_HAL_EXECUTABLE_LIBRARY_VERSION_0_6, true>, &loader));
  iree_hal_executable_loader_release(loader);
}

// Bit 3 is reserved in v0.6, where any nonzero return is INTERNAL.
TEST(StaticLibraryLoaderTest, V06IgnoresStatusFlag) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INTERNAL,
      LoadAndCall(QueryLibrary<IREE_HAL_EXECUTABLE_LIBRARY_VERSION_0_6, true>));
}

TEST(StaticLibraryLoaderTest, V07ReturnsStatus) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      LoadAndCall(
          QueryLibrary<IREE_HAL_EXECUTABLE_LIBRARY_VERSION_0_7, false>));
}

}  // namespace
