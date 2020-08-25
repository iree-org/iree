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

#include "iree/tools/vm_util.h"

#include <sstream>

#include "absl/strings/str_cat.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/api.h"

namespace iree {
namespace {

class VmUtilTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    IREE_ASSERT_OK(iree_hal_module_register_types());
    IREE_ASSERT_OK(CreateDevice("vmla", &device_));
    allocator_ = iree_hal_device_allocator(device_);
  }

  virtual void TearDown() { iree_hal_device_release(device_); }

  iree_hal_device_t* device_ = nullptr;
  iree_hal_allocator_t* allocator_ = nullptr;
};

TEST_F(VmUtilTest, ParsePrintBuffer) {
  absl::string_view buf_string = "2x2xi32=[42 43][44 45]";
  RawSignatureParser::Description desc;
  desc.type = RawSignatureParser::Type::kBuffer;
  desc.buffer.scalar_type = AbiConstants::ScalarType::kSint32;
  desc.dims = {2, 2};

  IREE_ASSERT_OK_AND_ASSIGN(
      auto variant_list, ParseToVariantList({desc}, allocator_, {buf_string}));
  std::stringstream os;
  IREE_ASSERT_OK(PrintVariantList({desc}, variant_list.get(), &os));
  EXPECT_EQ(os.str(), absl::StrCat(buf_string, "\n"));
}

TEST_F(VmUtilTest, ParsePrintScalar) {
  absl::string_view input_string = "i32=42";
  RawSignatureParser::Description desc;
  desc.type = RawSignatureParser::Type::kScalar;
  desc.scalar.type = AbiConstants::ScalarType::kSint32;

  IREE_ASSERT_OK_AND_ASSIGN(
      auto variant_list,
      ParseToVariantList({desc}, allocator_, {input_string}));
  std::stringstream os;
  IREE_ASSERT_OK(PrintVariantList({desc}, variant_list.get(), &os));
  EXPECT_EQ(os.str(), absl::StrCat(input_string, "\n"));
}

TEST_F(VmUtilTest, ParsePrintRank0Buffer) {
  absl::string_view buf_string = "i32=42";
  RawSignatureParser::Description desc;
  desc.type = RawSignatureParser::Type::kBuffer;
  desc.buffer.scalar_type = AbiConstants::ScalarType::kSint32;

  IREE_ASSERT_OK_AND_ASSIGN(
      auto variant_list, ParseToVariantList({desc}, allocator_, {buf_string}));
  std::stringstream os;
  IREE_ASSERT_OK(PrintVariantList({desc}, variant_list.get(), &os));
  EXPECT_EQ(os.str(), absl::StrCat(buf_string, "\n"));
}

TEST_F(VmUtilTest, ParsePrintMultipleBuffers) {
  absl::string_view buf_string1 = "2x2xi32=[42 43][44 45]";
  RawSignatureParser::Description desc1;
  desc1.type = RawSignatureParser::Type::kBuffer;
  desc1.buffer.scalar_type = AbiConstants::ScalarType::kSint32;
  desc1.dims = {2, 2};

  absl::string_view buf_string2 = "2x3xf64=[1 2 3][4 5 6]";
  RawSignatureParser::Description desc2;
  desc2.type = RawSignatureParser::Type::kBuffer;
  desc2.buffer.scalar_type = AbiConstants::ScalarType::kIeeeFloat64;
  desc2.dims = {2, 3};

  IREE_ASSERT_OK_AND_ASSIGN(auto variant_list,
                            ParseToVariantList({desc1, desc2}, allocator_,
                                               {buf_string1, buf_string2}));
  std::stringstream os;
  IREE_ASSERT_OK(PrintVariantList({desc1, desc2}, variant_list.get(), &os));
  EXPECT_EQ(os.str(), absl::StrCat(buf_string1, "\n", buf_string2, "\n"));
}

}  // namespace
}  // namespace iree
