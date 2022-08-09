// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/dynamic_library.h"

#include <cstdlib>
#include <iostream>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>

#include "iree/base/api.h"
#include "iree/base/internal/file_io.h"
#include "iree/base/testing/dynamic_library_test_library_embed.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

using iree::testing::status::StatusIs;

static const char* kUnknownName = "library_that_does_not_exist.so";

class DynamicLibraryTest : public ::testing::Test {
 public:
  static std::string GetTempFilename(const char* suffix) {
    static int unique_id = 0;
    char* test_tmpdir = getenv("TEST_TMPDIR");
    if (!test_tmpdir) {
      test_tmpdir = getenv("TMPDIR");
    }
    if (!test_tmpdir) {
      test_tmpdir = getenv("TEMP");
    }
    if (!test_tmpdir) {
      std::cerr << "TEST_TMPDIR/TMPDIR/TEMP not defined\n";
      exit(1);
    }
    return test_tmpdir + std::string("/iree_test_") +
           std::to_string(unique_id++) + suffix;
  }

  static void SetUpTestCase() {
    // Making files available to tests, particularly across operating systems
    // and build tools (Bazel/CMake) is complicated. Rather than include a test
    // dynamic library as a "testdata" file, we use c_embed_data to package
    // the file so it's embedded in a C module, then write that embedded file
    // to a platform/test-environment specific temp file for loading.

    // System APIs for loading dynamic libraries typically require an extension.
#if defined(IREE_PLATFORM_WINDOWS)
    static constexpr const char* ext = ".dll";
#else
    static constexpr const char* ext = ".so";
#endif
    library_temp_path_ = GetTempFilename(ext);

    const struct iree_file_toc_t* file_toc =
        dynamic_library_test_library_create();
    IREE_ASSERT_OK(iree_file_write_contents(
        library_temp_path_.c_str(),
        iree_make_const_byte_span(file_toc->data, file_toc->size)));

    std::cout << "Embedded test library written to temp path: "
              << library_temp_path_ << "\n";
  }

  static std::string library_temp_path_;
};

std::string DynamicLibraryTest::library_temp_path_;

TEST_F(DynamicLibraryTest, LoadLibrarySuccess) {
  iree_dynamic_library_t* library = NULL;
  IREE_ASSERT_OK(iree_dynamic_library_load_from_file(
      library_temp_path_.c_str(), IREE_DYNAMIC_LIBRARY_FLAG_NONE,
      iree_allocator_system(), &library));
  iree_dynamic_library_release(library);
}

TEST_F(DynamicLibraryTest, LoadLibraryFailure) {
  iree_dynamic_library_t* library = NULL;
  iree_status_t status = iree_dynamic_library_load_from_file(
      kUnknownName, IREE_DYNAMIC_LIBRARY_FLAG_NONE, iree_allocator_system(),
      &library);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
  iree_status_free(status);
}

TEST_F(DynamicLibraryTest, LoadLibraryTwice) {
  iree_dynamic_library_t* library1 = NULL;
  iree_dynamic_library_t* library2 = NULL;
  IREE_ASSERT_OK(iree_dynamic_library_load_from_file(
      library_temp_path_.c_str(), IREE_DYNAMIC_LIBRARY_FLAG_NONE,
      iree_allocator_system(), &library1));
  IREE_ASSERT_OK(iree_dynamic_library_load_from_file(
      library_temp_path_.c_str(), IREE_DYNAMIC_LIBRARY_FLAG_NONE,
      iree_allocator_system(), &library2));
  iree_dynamic_library_release(library1);
  iree_dynamic_library_release(library2);
}

TEST_F(DynamicLibraryTest, GetSymbolSuccess) {
  iree_dynamic_library_t* library = NULL;
  IREE_ASSERT_OK(iree_dynamic_library_load_from_file(
      library_temp_path_.c_str(), IREE_DYNAMIC_LIBRARY_FLAG_NONE,
      iree_allocator_system(), &library));

  int (*fn_ptr)(int);
  IREE_ASSERT_OK(iree_dynamic_library_lookup_symbol(library, "times_two",
                                                    (void**)&fn_ptr));
  ASSERT_NE(nullptr, fn_ptr);
  EXPECT_EQ(246, fn_ptr(123));

  iree_dynamic_library_release(library);
}

TEST_F(DynamicLibraryTest, GetSymbolFailure) {
  iree_dynamic_library_t* library = NULL;
  IREE_ASSERT_OK(iree_dynamic_library_load_from_file(
      library_temp_path_.c_str(), IREE_DYNAMIC_LIBRARY_FLAG_NONE,
      iree_allocator_system(), &library));

  int (*fn_ptr)(int);
  iree_status_t status =
      iree_dynamic_library_lookup_symbol(library, "unknown", (void**)&fn_ptr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
  iree_status_free(status);
  EXPECT_EQ(nullptr, fn_ptr);

  iree_dynamic_library_release(library);
}

}  // namespace
}  // namespace iree
