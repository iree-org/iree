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

#include "iree/base/status.h"
#include "iree/base/status_matchers.h"
#include "iree/testing/gtest.h"

namespace iree {
namespace {

static const char* kLibrarySearchNames[] = {
    "libdynamic_library_test_library.so",                // Bazel
    "iree_base_libdynamic_library_test_library.so",      // CMake Linux
    "iree_base_libdynamic_library_test_library.so.dll",  // CMake Windows
};
static const char* kUnknownName = "liblibrary_that_does_not_exist.so";

TEST(DynamicLibraryTest, LoadLibrarySuccess) {
  auto library_or = DynamicLibrary::Load(absl::MakeSpan(kLibrarySearchNames));
  ASSERT_OK(library_or);

  auto library = std::move(library_or.value());

  // Check that one of the search names matches the actual loaded name.
  std::string library_file_name = library->file_name();
  bool found_match = false;
  for (int i = 0; i < 3; ++i) {
    if (std::strcmp(kLibrarySearchNames[i], library_file_name.c_str()) == 0) {
      found_match = true;
    }
  }
  EXPECT_TRUE(found_match);
}

TEST(DynamicLibraryTest, LoadLibraryFailure) {
  auto library_or = DynamicLibrary::Load(kUnknownName);
  EXPECT_TRUE(IsUnavailable(library_or.status()));
}

TEST(DynamicLibraryTest, LoadLibraryTwice) {
  ASSERT_OK_AND_ASSIGN(
      auto library1, DynamicLibrary::Load(absl::MakeSpan(kLibrarySearchNames)));
  ASSERT_OK_AND_ASSIGN(
      auto library2, DynamicLibrary::Load(absl::MakeSpan(kLibrarySearchNames)));
}

TEST(DynamicLibraryTest, GetSymbolSuccess) {
  ASSERT_OK_AND_ASSIGN(
      auto library, DynamicLibrary::Load(absl::MakeSpan(kLibrarySearchNames)));

  auto times_two_fn = library->GetSymbol<int (*)(int)>("times_two");
  ASSERT_NE(nullptr, times_two_fn);
  EXPECT_EQ(246, times_two_fn(123));
}

TEST(DynamicLibraryTest, GetSymbolFailure) {
  ASSERT_OK_AND_ASSIGN(
      auto library, DynamicLibrary::Load(absl::MakeSpan(kLibrarySearchNames)));

  auto unknown_fn = library->GetSymbol<int (*)(int)>("unknown");
  EXPECT_EQ(nullptr, unknown_fn);
}

}  // namespace
}  // namespace iree
