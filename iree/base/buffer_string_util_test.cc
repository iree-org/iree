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

#include "iree/base/buffer_string_util.h"

#include "absl/strings/string_view.h"
#include "iree/base/memory.h"
#include "iree/base/status.h"
#include "iree/base/status_matchers.h"
#include "iree/testing/gtest.h"

namespace iree {
namespace {

using testing::ElementsAre;

TEST(BufferStringUtilTest, ParseBufferDataPrintMode) {
  EXPECT_EQ(BufferDataPrintMode::kBinary,
            ParseBufferDataPrintMode("b").ValueOrDie());
  EXPECT_EQ(BufferDataPrintMode::kSignedInteger,
            ParseBufferDataPrintMode("i").ValueOrDie());
  EXPECT_EQ(BufferDataPrintMode::kUnsignedInteger,
            ParseBufferDataPrintMode("u").ValueOrDie());
  EXPECT_EQ(BufferDataPrintMode::kFloatingPoint,
            ParseBufferDataPrintMode("f").ValueOrDie());

  EXPECT_FALSE(ParseBufferDataPrintMode("").ok());
  EXPECT_FALSE(ParseBufferDataPrintMode("s").ok());
  EXPECT_FALSE(ParseBufferDataPrintMode("asdfasdf").ok());
}

TEST(BufferStringUtilTest, ParseBufferTypeElementSize) {
  EXPECT_EQ(1, ParseBufferTypeElementSize("1").ValueOrDie());
  EXPECT_EQ(7, ParseBufferTypeElementSize("7").ValueOrDie());
  EXPECT_EQ(4, ParseBufferTypeElementSize("i32").ValueOrDie());
  EXPECT_EQ(8, ParseBufferTypeElementSize("f64").ValueOrDie());

  EXPECT_FALSE(ParseBufferTypeElementSize("").ok());
  EXPECT_FALSE(ParseBufferTypeElementSize(" ").ok());
  EXPECT_FALSE(ParseBufferTypeElementSize("a").ok());
  EXPECT_FALSE(ParseBufferTypeElementSize("ib").ok());
  EXPECT_FALSE(ParseBufferTypeElementSize("i").ok());
  EXPECT_FALSE(ParseBufferTypeElementSize("i543ff").ok());
}

TEST(BufferStringUtilTest, MakeBufferTypeString) {
  EXPECT_EQ("f32",
            MakeBufferTypeString(4, BufferDataPrintMode::kFloatingPoint));
}

TEST(BufferStringUtilTest, ParseShape) {
  EXPECT_EQ((Shape{}), ParseShape("").ValueOrDie());
  EXPECT_EQ((Shape{1}), ParseShape("1").ValueOrDie());
  EXPECT_EQ((Shape{1, 2}), ParseShape("1x2").ValueOrDie());
  EXPECT_EQ((Shape{1, 2}), ParseShape(" 1 x 2 ").ValueOrDie());

  EXPECT_FALSE(ParseShape("abc").ok());
  EXPECT_FALSE(ParseShape("1xf").ok());
  EXPECT_FALSE(ParseShape("1xff23").ok());
}

TEST(BufferStringUtilTest, PrintShapedTypeToString) {
  EXPECT_EQ("f32", PrintShapedTypeToString(Shape{}, "f32"));
  EXPECT_EQ("1xi32", PrintShapedTypeToString(Shape{1}, "i32"));
  EXPECT_EQ("1x2xi8", PrintShapedTypeToString(Shape{1, 2}, "i8"));
}

TEST(BufferStringUtilTest, PrintBinaryDataToString) {
  EXPECT_EQ("00 01 02 03",
            PrintBinaryDataToString(1, {0, 1, 2, 3}, 10).ValueOrDie());
  EXPECT_EQ("0102 0304 ccdd eeff",
            PrintBinaryDataToString(
                2, {0x01, 0x02, 0x03, 0x04, 0xcc, 0xdd, 0xee, 0xff}, 10)
                .ValueOrDie());
  EXPECT_EQ("00...", PrintBinaryDataToString(1, {0, 1, 2, 3}, 1).ValueOrDie());

  EXPECT_EQ(
      "fabcfabc",
      PrintBinaryDataToString(4, {0xfa, 0xbc, 0xfa, 0xbc}, 10).ValueOrDie());

  EXPECT_EQ("fabcfabcfabcfabc",
            PrintBinaryDataToString(
                8, {0xfa, 0xbc, 0xfa, 0xbc, 0xfa, 0xbc, 0xfa, 0xbc}, 10)
                .ValueOrDie());
}

TEST(BufferStringUtilTest, PrintNumericalDataToString) {
  EXPECT_EQ(
      "0 1 2 3",
      PrintNumericalDataToString({4}, "u8", {0, 1, 2, 3}, 10).ValueOrDie());
  EXPECT_EQ(
      "[0 1][2 3]",
      PrintNumericalDataToString({2, 2}, "u8", {0, 1, 2, 3}, 10).ValueOrDie());
  std::vector<int32_t> data = {0, -1, 2, 3};
  auto bytes = ReinterpretSpan<uint8_t>(absl::MakeSpan(data));
  EXPECT_EQ("0 -1 2 3",
            PrintNumericalDataToString({4}, "i32", bytes, 10).ValueOrDie());
}

TEST(BufferStringUtilTest, ParseBufferDatai8) {
  std::vector<uint8_t> data(4);
  auto data_span = absl::MakeSpan(data);
  ASSERT_OK(ParseBufferDataAsType("0 1 2 3", "i8", data_span));
  EXPECT_THAT(ReinterpretSpan<int8_t>(data_span), ElementsAre(0, 1, 2, 3));
}

TEST(BufferStringUtilTest, ParseBufferDatai32) {
  std::vector<uint8_t> data(4 * sizeof(int32_t));
  auto data_span = absl::MakeSpan(data);
  ASSERT_OK(ParseBufferDataAsType("0 1 2 3", "i32", data_span));
  EXPECT_THAT(ReinterpretSpan<int32_t>(data_span), ElementsAre(0, 1, 2, 3));
}

TEST(BufferStringUtilTest, ParseBufferDataf32) {
  std::vector<uint8_t> data(4 * sizeof(float));
  auto data_span = absl::MakeSpan(data);
  ASSERT_OK(ParseBufferDataAsType("0 1.1 2 3", "f32", data_span));
  EXPECT_THAT(ReinterpretSpan<float>(data_span), ElementsAre(0, 1.1, 2, 3));
}

TEST(BufferStringUtilTest, ParseBufferDataBinary) {
  std::vector<uint8_t> data(4);
  auto data_span = absl::MakeSpan(data);
  ASSERT_OK(ParseBufferDataAsType("00 01 02 03", "8", data_span));
  EXPECT_THAT(data_span, ElementsAre(0, 1, 2, 3));
}

}  // namespace
}  // namespace iree
