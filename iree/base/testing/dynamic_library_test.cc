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

#include "iree/base/dynamic_library.h"

#include <string>

#include "iree/base/internal/file_io.h"
#include "iree/base/status.h"
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
  static void SetUpTestCase() {
    // Making files available to tests, particularly across operating systems
    // and build tools (Bazel/CMake) is complicated. Rather than include a test
    // dynamic library as a "testdata" file, we use cc_embed_data to package
    // the file so it's embedded in a C++ module, then write that embedded file
    // to a platform/test-environment specific temp file for loading.

    std::string base_name = "dynamic_library_test_library";
    IREE_ASSERT_OK(file_io::GetTempFile(base_name, &library_temp_path_));
    // System APIs for loading dynamic libraries typically require an extension.
#if defined(IREE_PLATFORM_WINDOWS)
    library_temp_path_ += ".dll";
#else
    library_temp_path_ += ".so";
#endif

    const auto* file_toc = dynamic_library_test_library_create();
    absl::string_view file_data(reinterpret_cast<const char*>(file_toc->data),
                                file_toc->size);
    IREE_ASSERT_OK(file_io::SetFileContents(library_temp_path_, file_data));

    IREE_LOG(INFO) << "Embedded test library written to temp path: "
                   << library_temp_path_;
  }

  static std::string library_temp_path_;
};

std::string DynamicLibraryTest::library_temp_path_;

TEST_F(DynamicLibraryTest, LoadLibrarySuccess) {
  std::unique_ptr<DynamicLibrary> library;
  IREE_ASSERT_OK(DynamicLibrary::Load(library_temp_path_.c_str(), &library));
}

TEST_F(DynamicLibraryTest, LoadLibraryFailure) {
  std::unique_ptr<DynamicLibrary> library;
  EXPECT_THAT(DynamicLibrary::Load(kUnknownName, &library),
              StatusIs(iree::StatusCode::kUnavailable));
}

TEST_F(DynamicLibraryTest, LoadLibraryTwice) {
  std::unique_ptr<DynamicLibrary> library1;
  IREE_ASSERT_OK(DynamicLibrary::Load(library_temp_path_.c_str(), &library1));
  std::unique_ptr<DynamicLibrary> library2;
  IREE_ASSERT_OK(DynamicLibrary::Load(library_temp_path_.c_str(), &library2));
}

TEST_F(DynamicLibraryTest, GetSymbolSuccess) {
  std::unique_ptr<DynamicLibrary> library;
  IREE_ASSERT_OK(DynamicLibrary::Load(library_temp_path_.c_str(), &library));

  auto times_two_fn = library->GetSymbol<int (*)(int)>("times_two");
  ASSERT_NE(nullptr, times_two_fn);
  EXPECT_EQ(246, times_two_fn(123));
}

TEST_F(DynamicLibraryTest, GetSymbolFailure) {
  std::unique_ptr<DynamicLibrary> library;
  IREE_ASSERT_OK(DynamicLibrary::Load(library_temp_path_.c_str(), &library));

  auto unknown_fn = library->GetSymbol<int (*)(int)>("unknown");
  EXPECT_EQ(nullptr, unknown_fn);
}

}  // namespace
}  // namespace iree
