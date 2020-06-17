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

#include "iree/base/dynamic_library_test_library_embed.h"
#include "iree/base/file_io.h"
#include "iree/base/status.h"
#include "iree/base/status_matchers.h"
#include "iree/base/target_platform.h"
#include "iree/testing/gtest.h"

namespace iree {
namespace {

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
    ASSERT_OK_AND_ASSIGN(library_temp_path_, file_io::GetTempFile(base_name));
    // System APIs for loading dynamic libraries typically require an extension.
#if defined(IREE_PLATFORM_WINDOWS)
    library_temp_path_ += ".dll";
#else
    library_temp_path_ += ".so";
#endif

    const auto* file_toc = dynamic_library_test_library_create();
    absl::string_view file_data(reinterpret_cast<const char*>(file_toc->data),
                                file_toc->size);
    ASSERT_OK(file_io::SetFileContents(library_temp_path_, file_data));

    LOG(INFO) << "Embedded test library written to temp path: "
              << library_temp_path_;
  }

  static std::string library_temp_path_;
};

std::string DynamicLibraryTest::library_temp_path_;

TEST_F(DynamicLibraryTest, LoadLibrarySuccess) {
  auto library_or = DynamicLibrary::Load(library_temp_path_.c_str());
  ASSERT_OK(library_or);

  auto library = std::move(library_or.value());

  EXPECT_EQ(library_temp_path_, library->file_name());
}

TEST_F(DynamicLibraryTest, LoadLibraryFailure) {
  auto library_or = DynamicLibrary::Load(kUnknownName);
  EXPECT_TRUE(IsUnavailable(library_or.status()));
}

TEST_F(DynamicLibraryTest, LoadLibraryTwice) {
  ASSERT_OK_AND_ASSIGN(auto library1,
                       DynamicLibrary::Load(library_temp_path_.c_str()));
  ASSERT_OK_AND_ASSIGN(auto library2,
                       DynamicLibrary::Load(library_temp_path_.c_str()));
}

TEST_F(DynamicLibraryTest, GetSymbolSuccess) {
  ASSERT_OK_AND_ASSIGN(auto library,
                       DynamicLibrary::Load(library_temp_path_.c_str()));

  auto times_two_fn = library->GetSymbol<int (*)(int)>("times_two");
  ASSERT_NE(nullptr, times_two_fn);
  EXPECT_EQ(246, times_two_fn(123));
}

TEST_F(DynamicLibraryTest, GetSymbolFailure) {
  ASSERT_OK_AND_ASSIGN(auto library,
                       DynamicLibrary::Load(library_temp_path_.c_str()));

  auto unknown_fn = library->GetSymbol<int (*)(int)>("unknown");
  EXPECT_EQ(nullptr, unknown_fn);
}

}  // namespace
}  // namespace iree
