// Copyright 2019 Google LLC
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

#include "build_tools/embed_data/testembed1.h"

#include "iree/testing/gtest.h"

namespace {

TEST(Generator, TestContents) {
  auto* toc = ::foobar::testembed1_create();
  ASSERT_EQ("file1.txt", std::string(toc->name));
  ASSERT_EQ(R"(Are you '"Still"' here?)"
            "\n",
            std::string(toc->data));
  ASSERT_EQ(24, toc->size);
  ASSERT_EQ(0, *(toc->data + toc->size));

  ++toc;
  ASSERT_EQ("file2.txt", std::string(toc->name));
  ASSERT_EQ(R"(¯\_(ツ)_/¯)"
            "\n",
            std::string(toc->data));
  ASSERT_EQ(14, toc->size);
  ASSERT_EQ(0, *(toc->data + toc->size));

  ++toc;
  ASSERT_EQ(nullptr, toc->name);
  ASSERT_EQ(nullptr, toc->data);
}

}  // namespace
