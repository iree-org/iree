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
#include <vector>

#include "absl/strings/str_cat.h"
#include "iree/base/status.h"
#include "iree/base/status_matchers.h"
#include "iree/base/target_platform.h"
#include "iree/testing/gtest.h"

#if defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_APPLE) || \
    defined(IREE_PLATFORM_LINUX)
#include <libgen.h>
#endif

namespace iree {
namespace {

static const char* kLibrarySearchNames[] = {
    "libdynamic_library_test_library.so",                  // Bazel
    "libiree_base_libdynamic_library_test_library.so.so",  // CMake Linux
    "iree_base_libdynamic_library_test_library.so.dll",    // CMake Windows
};
static const char* kUnknownName = "liblibrary_that_does_not_exist.so";

class DynamicLibraryTest : public ::testing::Test {
 public:
  static void SetUpTestCase() {
    for (int i = 0; i < 3; ++i) {
      library_search_paths_.push_back(absl::StrCat(kLibrarySearchNames[i]));
    }

// Windows can access shared libraries (DLLs) in the same directory as the
// test executable. Other platforms might not be able to.
// For those platforms, also augment the search paths with absolute paths.
#if defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_APPLE) || \
    defined(IREE_PLATFORM_LINUX)
    char bin_path_result[PATH_MAX];
    ssize_t count = ::readlink("/proc/self/exe", bin_path_result, PATH_MAX);
    const char* bin_path = nullptr;
    if (count != -1) {
      bin_path = dirname(bin_path_result);
    }

    char* cwd_path = ::getcwd(nullptr, 0);

    for (int i = 0; i < 3; ++i) {
      library_search_paths_.push_back(
          absl::StrCat(bin_path, "/", kLibrarySearchNames[i]));
      library_search_paths_.push_back(
          absl::StrCat(cwd_path, "/iree/base/", kLibrarySearchNames[i]));
    }
    free(cwd_path);
#endif
  }

  static std::vector<std::string> library_search_paths_;
};

std::vector<std::string> DynamicLibraryTest::library_search_paths_;

TEST_F(DynamicLibraryTest, LoadLibrarySuccess) {
  auto library_or = DynamicLibrary::Load(library_search_paths_);
  ASSERT_OK(library_or);

  auto library = std::move(library_or.value());

  // Check that one of the search names matches the actual loaded name.
  std::string library_file_name = library->file_name();
  bool found_match = false;
  for (int i = 0; i < library_search_paths_.size(); ++i) {
    if (std::strcmp(library_search_paths_[i].c_str(),
                    library_file_name.c_str()) == 0) {
      found_match = true;
    }
  }
  EXPECT_TRUE(found_match);
}

TEST_F(DynamicLibraryTest, LoadLibraryFailure) {
  auto library_or = DynamicLibrary::Load(kUnknownName);
  EXPECT_TRUE(IsUnavailable(library_or.status()));
}

TEST_F(DynamicLibraryTest, LoadLibraryTwice) {
  ASSERT_OK_AND_ASSIGN(auto library1,
                       DynamicLibrary::Load(library_search_paths_));
  ASSERT_OK_AND_ASSIGN(auto library2,
                       DynamicLibrary::Load(library_search_paths_));
}

TEST_F(DynamicLibraryTest, GetSymbolSuccess) {
  ASSERT_OK_AND_ASSIGN(auto library,
                       DynamicLibrary::Load(library_search_paths_));

  auto times_two_fn = library->GetSymbol<int (*)(int)>("times_two");
  ASSERT_NE(nullptr, times_two_fn);
  EXPECT_EQ(246, times_two_fn(123));
}

TEST_F(DynamicLibraryTest, GetSymbolFailure) {
  ASSERT_OK_AND_ASSIGN(auto library,
                       DynamicLibrary::Load(library_search_paths_));

  auto unknown_fn = library->GetSymbol<int (*)(int)>("unknown");
  EXPECT_EQ(nullptr, unknown_fn);
}

}  // namespace
}  // namespace iree
