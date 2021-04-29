// Copyright 2021 Google LLC
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

#include <string>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"

namespace {

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

TEST(StringViewTest, Trim) {
  auto trim = [](const char* value) -> std::string {
    return ToString(iree_string_view_trim(iree_make_cstring_view(value)));
  };
  ASSERT_EQ(trim(""), "");
  ASSERT_EQ(trim("a"), "a");
  ASSERT_EQ(trim(" a"), "a");
  ASSERT_EQ(trim("a "), "a");
  ASSERT_EQ(trim("a b"), "a b");
  ASSERT_EQ(trim(" a b "), "a b");
  ASSERT_EQ(trim("\t\t\na b\n \t "), "a b");
  ASSERT_EQ(trim("\n"), "");
  ASSERT_EQ(trim("\r\n"), "");
}

}  // namespace
