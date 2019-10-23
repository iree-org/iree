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

#include "iree/base/bitfield.h"

#include <cstdint>
#include <vector>

#include "iree/testing/gtest.h"

namespace iree {

// NOTE: define here so that we don't get internal linkage warnings.
enum class MyValue : uint32_t {
  kNone = 0,
  kA = 1 << 0,
  kB = 1 << 1,
  kAll = kA | kB,
};
IREE_BITFIELD(MyValue);

namespace {

// Tests general usage.
TEST(BitfieldTest, FormatBitfieldValue) {
  std::vector<std::pair<MyValue, const char *>> mappings = {
      {MyValue::kA, "kA"},
      {MyValue::kB, "kB"},
  };
  EXPECT_EQ("",
            FormatBitfieldValue(MyValue::kNone, absl::MakeConstSpan(mappings)));
  EXPECT_EQ("kA",
            FormatBitfieldValue(MyValue::kA, absl::MakeConstSpan(mappings)));
  EXPECT_EQ("kA|kB", FormatBitfieldValue(MyValue::kA | MyValue::kB,
                                         absl::MakeConstSpan(mappings)));
}

// Tests that empty mapping tables are fine.
TEST(BitfieldTest, FormatBitfieldValueEmpty) {
  EXPECT_EQ("", FormatBitfieldValue(MyValue::kNone, {}));
}

// Tests that values not found in the mappings are still displayed.
TEST(BitfieldTest, FormatBitfieldValueUnhandledValues) {
  EXPECT_EQ("kA|2h", FormatBitfieldValue(MyValue::kA | MyValue::kB,
                                         {
                                             {MyValue::kA, "kA"},
                                         }));
}

// Tests priority order in the mapping table.
TEST(BitfieldTest, FormatBitfieldValuePriority) {
  // No priority, will do separate.
  EXPECT_EQ("kA|kB", FormatBitfieldValue(MyValue::kA | MyValue::kB,
                                         {
                                             {MyValue::kA, "kA"},
                                             {MyValue::kB, "kB"},
                                             {MyValue::kAll, "kAll"},
                                         }));

  // Priority on the combined flag, use that instead.
  EXPECT_EQ("kAll", FormatBitfieldValue(MyValue::kA | MyValue::kB,
                                        {
                                            {MyValue::kAll, "kAll"},
                                            {MyValue::kA, "kA"},
                                            {MyValue::kB, "kB"},
                                        }));
}

}  // namespace
}  // namespace iree
