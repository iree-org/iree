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

using ::iree::testing::status::IsOkAndHolds;
using ::iree::testing::status::StatusIs;
using ::testing::ElementsAre;
using ::testing::Eq;

TEST(BufferStringUtilTest, ParseBufferDataPrintMode) {
  EXPECT_THAT(ParseBufferDataPrintMode("b"),
              IsOkAndHolds(Eq(BufferDataPrintMode::kBinary)));
  EXPECT_THAT(ParseBufferDataPrintMode("i"),
              IsOkAndHolds(Eq(BufferDataPrintMode::kSignedInteger)));
  EXPECT_THAT(ParseBufferDataPrintMode("u"),
              IsOkAndHolds(Eq(BufferDataPrintMode::kUnsignedInteger)));
  EXPECT_THAT(ParseBufferDataPrintMode("f"),
              IsOkAndHolds(Eq(BufferDataPrintMode::kFloatingPoint)));

  EXPECT_THAT(ParseBufferDataPrintMode("bb"),
              IsOkAndHolds(Eq(BufferDataPrintMode::kBinary)));
  EXPECT_THAT(ParseBufferDataPrintMode("ii"),
              IsOkAndHolds(Eq(BufferDataPrintMode::kSignedInteger)));
  EXPECT_THAT(ParseBufferDataPrintMode("uu"),
              IsOkAndHolds(Eq(BufferDataPrintMode::kUnsignedInteger)));
  EXPECT_THAT(ParseBufferDataPrintMode("ff"),
              IsOkAndHolds(Eq(BufferDataPrintMode::kFloatingPoint)));

  EXPECT_THAT(ParseBufferDataPrintMode(""),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBufferDataPrintMode("s"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBufferDataPrintMode("asdfasdf"),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST(BufferStringUtilTest, ParseBufferTypeElementSize) {
  EXPECT_THAT(ParseBufferTypeElementSize("1"), IsOkAndHolds(Eq(1)));
  EXPECT_THAT(ParseBufferTypeElementSize("2"), IsOkAndHolds(Eq(2)));
  EXPECT_THAT(ParseBufferTypeElementSize("4"), IsOkAndHolds(Eq(4)));
  EXPECT_THAT(ParseBufferTypeElementSize("8"), IsOkAndHolds(Eq(8)));
  EXPECT_THAT(ParseBufferTypeElementSize("i8"), IsOkAndHolds(Eq(1)));
  EXPECT_THAT(ParseBufferTypeElementSize("u8"), IsOkAndHolds(Eq(1)));
  EXPECT_THAT(ParseBufferTypeElementSize("i16"), IsOkAndHolds(Eq(2)));
  EXPECT_THAT(ParseBufferTypeElementSize("u16"), IsOkAndHolds(Eq(2)));
  EXPECT_THAT(ParseBufferTypeElementSize("i32"), IsOkAndHolds(Eq(4)));
  EXPECT_THAT(ParseBufferTypeElementSize("u32"), IsOkAndHolds(Eq(4)));
  EXPECT_THAT(ParseBufferTypeElementSize("i64"), IsOkAndHolds(Eq(8)));
  EXPECT_THAT(ParseBufferTypeElementSize("u64"), IsOkAndHolds(Eq(8)));
  EXPECT_THAT(ParseBufferTypeElementSize("f32"), IsOkAndHolds(Eq(4)));
  EXPECT_THAT(ParseBufferTypeElementSize("f64"), IsOkAndHolds(Eq(8)));

  EXPECT_THAT(ParseBufferTypeElementSize(""),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBufferTypeElementSize(" "),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBufferTypeElementSize("a"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBufferTypeElementSize("ib"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBufferTypeElementSize("i"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBufferTypeElementSize("i543ff"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBufferTypeElementSize("i33"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBufferTypeElementSize("x32"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBufferTypeElementSize("f16"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBufferTypeElementSize("i1"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBufferTypeElementSize("i24"),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBufferTypeElementSize("i128"),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST(BufferStringUtilTest, MakeBufferTypeString) {
  EXPECT_THAT(MakeBufferTypeString(1, BufferDataPrintMode::kBinary),
              IsOkAndHolds(Eq("1")));
  EXPECT_THAT(MakeBufferTypeString(1, BufferDataPrintMode::kSignedInteger),
              IsOkAndHolds(Eq("i8")));
  EXPECT_THAT(MakeBufferTypeString(2, BufferDataPrintMode::kUnsignedInteger),
              IsOkAndHolds(Eq("u16")));
  EXPECT_THAT(MakeBufferTypeString(4, BufferDataPrintMode::kFloatingPoint),
              IsOkAndHolds(Eq("f32")));

  EXPECT_THAT(MakeBufferTypeString(0, BufferDataPrintMode::kBinary),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(MakeBufferTypeString(-1, BufferDataPrintMode::kBinary),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(MakeBufferTypeString(-1, BufferDataPrintMode::kSignedInteger),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(MakeBufferTypeString(-2, BufferDataPrintMode::kUnsignedInteger),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(MakeBufferTypeString(-4, BufferDataPrintMode::kFloatingPoint),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST(BufferStringUtilTest, ParseShape) {
  EXPECT_THAT(ParseShape(""), IsOkAndHolds(Eq(Shape{})));
  EXPECT_THAT(ParseShape("0"), IsOkAndHolds(Eq(Shape{0})));
  EXPECT_THAT(ParseShape("1"), IsOkAndHolds(Eq(Shape{1})));
  EXPECT_THAT(ParseShape("1x2"), IsOkAndHolds(Eq(Shape{1, 2})));
  EXPECT_THAT(ParseShape(" 1 x 2 "), IsOkAndHolds(Eq(Shape{1, 2})));
  EXPECT_THAT(ParseShape("1x2x3x4x5"), IsOkAndHolds(Eq(Shape{1, 2, 3, 4, 5})));

  EXPECT_THAT(ParseShape("abc"), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShape("1xf"), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShape("1xff23"), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShape("1xf32"), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShape("x"), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShape("x1"), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShape("1x"), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShape("x1x2"), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShape("1xx2"), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShape("1x2x"), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShape("0x-1"), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseShape("1x2x3x4x5x6"),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST(BufferStringUtilTest, PrintShapedTypeToString) {
  EXPECT_EQ("f32", PrintShapedTypeToString(Shape{}, "f32"));
  EXPECT_EQ("0xi32", PrintShapedTypeToString(Shape{0}, "i32"));
  EXPECT_EQ("1xi32", PrintShapedTypeToString(Shape{1}, "i32"));
  EXPECT_EQ("1x2xi8", PrintShapedTypeToString(Shape{1, 2}, "i8"));
}

TEST(BufferStringUtilTest, PrintBinaryDataToString) {
  EXPECT_THAT(PrintBinaryDataToString(1, {0, 1, 2, 3}, 10),
              IsOkAndHolds(Eq("00 01 02 03")));
  EXPECT_THAT(PrintBinaryDataToString(
                  2, {0x01, 0x02, 0x03, 0x04, 0xcc, 0xdd, 0xee, 0xff}, 10),
              IsOkAndHolds(Eq("0102 0304 ccdd eeff")));
  EXPECT_THAT(PrintBinaryDataToString(4, {0xfa, 0xbc, 0xfa, 0xbc}, 10),
              IsOkAndHolds(Eq("fabcfabc")));
  EXPECT_THAT(PrintBinaryDataToString(
                  8, {0xfa, 0xbc, 0xfa, 0xbc, 0xfa, 0xbc, 0xfa, 0xbc}, 10),
              IsOkAndHolds(Eq("fabcfabcfabcfabc")));

  EXPECT_THAT(PrintBinaryDataToString(1, {0, 1, 2, 3}, 0),
              IsOkAndHolds(Eq("...")));
  EXPECT_THAT(PrintBinaryDataToString(1, {0, 1, 2, 3}, 1),
              IsOkAndHolds(Eq("00...")));
  EXPECT_THAT(PrintBinaryDataToString(1, {0, 1, 2, 3}, 2),
              IsOkAndHolds(Eq("00 01...")));

  EXPECT_THAT(PrintBinaryDataToString(-1, {0, 1, 2, 3}, 10),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(
      PrintBinaryDataToString(
          16, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 10),
      StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(PrintBinaryDataToString(3, {0, 1, 2, 3}, 10),
              StatusIs(StatusCode::kInvalidArgument));
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
