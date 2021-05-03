// Copyright 2020 Google LLC
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

#include "iree/base/internal/dynamic_library.h"

#include <string>

#include "iree/base/internal/file_io.h"
#include "iree/base/target_platform.h"
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
    IREE_CHECK(test_tmpdir) << "TEST_TMPDIR/TMPDIR/TEMP not defined";
    return test_tmpdir + std::string("/iree_test_") +
           std::to_string(unique_id++) + suffix;
  }

  static void SetUpTestCase() {
    // Making files available to tests, particularly across operating systems
    // and build tools (Bazel/CMake) is complicated. Rather than include a test
    // dynamic library as a "testdata" file, we use cc_embed_data to package
    // the file so it's embedded in a C++ module, then write that embedded file
    // to a platform/test-environment specific temp file for loading.

    // System APIs for loading dynamic libraries typically require an extension.
#if defined(IREE_PLATFORM_WINDOWS)
    static constexpr const char* ext = ".dll";
#else
    static constexpr const char* ext = ".so";
#endif
    library_temp_path_ = GetTempFilename(ext);

    const auto* file_toc = dynamic_library_test_library_create();
    IREE_ASSERT_OK(iree_file_write_contents(
        library_temp_path_.c_str(),
        iree_make_const_byte_span(file_toc->data, file_toc->size)));

    IREE_LOG(INFO) << "Embedded test library written to temp path: "
                   << library_temp_path_;
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
  EXPECT_THAT(iree_dynamic_library_load_from_file(
                  kUnknownName, IREE_DYNAMIC_LIBRARY_FLAG_NONE,
                  iree_allocator_system(), &library),
              StatusIs(iree::StatusCode::kNotFound));
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
  EXPECT_THAT(
      iree_dynamic_library_lookup_symbol(library, "unknown", (void**)&fn_ptr),
      StatusIs(iree::StatusCode::kNotFound));
  EXPECT_EQ(nullptr, fn_ptr);

  iree_dynamic_library_release(library);
}

}  // namespace
}  // namespace iree
