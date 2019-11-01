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

#include "iree/base/shaped_buffer_string_util.h"

#include "absl/strings/string_view.h"
#include "iree/base/buffer_string_util.h"
#include "iree/base/memory.h"
#include "iree/base/shaped_buffer.h"
#include "iree/base/status.h"
#include "iree/base/status_matchers.h"
#include "iree/testing/gtest.h"

namespace iree {
namespace {

using ::testing::ElementsAre;

template <typename T>
absl::Span<const T> ReadAs(absl::Span<const uint8_t> data) {
  return ReinterpretSpan<T>(data);
}

void RoundTripTest(absl::string_view buffer_string,
                   BufferDataPrintMode print_mode) {
  ASSERT_OK_AND_ASSIGN(auto shaped_buf,
                       ParseShapedBufferFromString(buffer_string));
  ASSERT_OK_AND_ASSIGN(auto new_string, PrintShapedBufferToString(
                                            shaped_buf, print_mode, SIZE_MAX));
  EXPECT_EQ(buffer_string, new_string);
}

void RoundTripTest(ShapedBuffer shaped_buf, BufferDataPrintMode print_mode) {
  ASSERT_OK_AND_ASSIGN(auto new_string, PrintShapedBufferToString(
                                            shaped_buf, print_mode, SIZE_MAX));
  ASSERT_OK_AND_ASSIGN(auto new_shaped_buf,
                       ParseShapedBufferFromString(new_string));
  EXPECT_EQ(shaped_buf, new_shaped_buf);
}

TEST(ShapedBufferStringUtilTest, ParseShapedBufferFromStringEmpty) {
  // Empty string = empty buffer_view.
  ASSERT_OK_AND_ASSIGN(auto m0, ParseShapedBufferFromString(""));
  EXPECT_TRUE(m0.contents().empty());
  EXPECT_EQ(Shape{}, m0.shape());
  EXPECT_EQ(0, m0.element_size());

  // No = means no data.
  ASSERT_OK_AND_ASSIGN(auto m1, ParseShapedBufferFromString("4x2xf32"));
  EXPECT_EQ(4 * 2 * 4, m1.contents().size());
  EXPECT_EQ(Shape({4, 2}), m1.shape());
  EXPECT_EQ(4, m1.element_size());
  EXPECT_THAT(ReadAs<float>(m1.contents()),
              ElementsAre(0, 0, 0, 0, 0, 0, 0, 0));

  // No data after = means no data.
  ASSERT_OK_AND_ASSIGN(auto m2, ParseShapedBufferFromString("4x2xf32="));
  EXPECT_EQ(4 * 2 * 4, m2.contents().size());
  EXPECT_EQ(Shape({4, 2}), m2.shape());
  EXPECT_EQ(4, m2.element_size());
  EXPECT_THAT(ReadAs<float>(m2.contents()),
              ElementsAre(0, 0, 0, 0, 0, 0, 0, 0));
}

TEST(ShapedBufferStringUtilTest, ParseShapedBufferFromStringBinary) {
  ASSERT_OK_AND_ASSIGN(auto m0, ParseShapedBufferFromString("4x1=00 01 02 03"));
  EXPECT_EQ(Shape({4}), m0.shape());
  EXPECT_EQ(1, m0.element_size());
  EXPECT_THAT(ReadAs<uint8_t>(m0.contents()), ElementsAre(0, 1, 2, 3));

  // Whitespace shouldn't matter.
  ASSERT_OK_AND_ASSIGN(auto m1, ParseShapedBufferFromString("4x1=00,010203"));
  EXPECT_EQ(Shape({4}), m1.shape());
  EXPECT_EQ(1, m1.element_size());
  EXPECT_THAT(ReadAs<uint8_t>(m1.contents()), ElementsAre(0, 1, 2, 3));

  // Should fail on malformed hex bytes.
  EXPECT_FALSE(ParseShapedBufferFromString("4x1=1").ok());
  EXPECT_FALSE(ParseShapedBufferFromString("4x1=00003").ok());
  EXPECT_FALSE(ParseShapedBufferFromString("4x1=%0123%\1").ok());
  EXPECT_FALSE(ParseShapedBufferFromString("4x1=00010203040506").ok());
}

TEST(ShapedBufferStringUtilTest, ParseShapedBufferFromStringAllowBrackets) {
  ASSERT_OK_AND_ASSIGN(auto m0,
                       ParseShapedBufferFromString("4xi16=[[0][ 1 ][2]][3]"));
  EXPECT_EQ(Shape({4}), m0.shape());
  EXPECT_EQ(2, m0.element_size());
  EXPECT_THAT(ReadAs<int16_t>(m0.contents()), ElementsAre(0, 1, 2, 3));
}

TEST(ShapedBufferStringUtilTest, ParseShapedBufferFromStringInteger) {
  // Signed int16.
  ASSERT_OK_AND_ASSIGN(auto m0,
                       ParseShapedBufferFromString("4xi16=0 12345 65535 -2"));
  EXPECT_EQ(Shape({4}), m0.shape());
  EXPECT_EQ(2, m0.element_size());
  EXPECT_THAT(ReadAs<int16_t>(m0.contents()), ElementsAre(0, 12345, -1, -2));

  // Unsigned int16.
  ASSERT_OK_AND_ASSIGN(auto m1,
                       ParseShapedBufferFromString("4xu16=0 12345 65535 -2"));
  EXPECT_EQ(Shape({4}), m1.shape());
  EXPECT_EQ(2, m1.element_size());
  EXPECT_THAT(ReadAs<uint16_t>(m1.contents()),
              ElementsAre(0, 12345, 65535, 65534));

  // Mixing separator types is ok.
  ASSERT_OK_AND_ASSIGN(
      auto m2, ParseShapedBufferFromString("4xu16=0, 12345, 65535, -2"));
  EXPECT_EQ(Shape({4}), m2.shape());
  EXPECT_EQ(2, m2.element_size());
  EXPECT_THAT(ReadAs<uint16_t>(m2.contents()),
              ElementsAre(0, 12345, 65535, 65534));

  // Should fail on malformed integers bytes and out of bounds values.
  EXPECT_FALSE(ParseShapedBufferFromString("4xi32=asodfj").ok());
  EXPECT_FALSE(ParseShapedBufferFromString("4xi32=0 1 2 3 4").ok());
}

TEST(ShapedBufferStringUtilTest, ParseShapedBufferFromStringFloat) {
  // Float.
  ASSERT_OK_AND_ASSIGN(auto m0,
                       ParseShapedBufferFromString("4xf32=0 1.0 1234 -2.0e-5"));
  EXPECT_EQ(Shape({4}), m0.shape());
  EXPECT_EQ(4, m0.element_size());
  EXPECT_THAT(ReadAs<float>(m0.contents()),
              ElementsAre(0.0f, 1.0f, 1234.0f, -2.0e-5f));

  // Double.
  ASSERT_OK_AND_ASSIGN(auto m1, ParseShapedBufferFromString(
                                    "4xf64=0 1.0 123456789012345 -2.0e-5"));
  EXPECT_EQ(Shape({4}), m1.shape());
  EXPECT_EQ(8, m1.element_size());
  EXPECT_THAT(ReadAs<double>(m1.contents()),
              ElementsAre(0.0, 1.0, 123456789012345.0, -2.0e-5));

  // Should fail on malformed floats and out of bounds values.
  EXPECT_FALSE(ParseShapedBufferFromString("4xf32=asodfj").ok());
  EXPECT_FALSE(ParseShapedBufferFromString("4xf32=0").ok());
  EXPECT_FALSE(ParseShapedBufferFromString("4xf32=0 1 2 3 4").ok());
}

TEST(ShapedBufferStringUtilTest, RoundTripParsePrint) {
  RoundTripTest("4xi8=0 -1 2 3", BufferDataPrintMode::kSignedInteger);
  RoundTripTest("4xi16=0 -1 2 3", BufferDataPrintMode::kSignedInteger);
  RoundTripTest("4xu16=0 1 2 3", BufferDataPrintMode::kUnsignedInteger);
  RoundTripTest("4xf32=0 1.1 2 3", BufferDataPrintMode::kFloatingPoint);
  RoundTripTest("1x2x3xi8=[[0 1 2][3 4 5]]",
                BufferDataPrintMode::kSignedInteger);
}

TEST(ShapedBufferStringUtilTest, RoundTripPrintParse) {
  RoundTripTest(ShapedBuffer::Create<int8_t>({4}, {0, 1, 2, 3}),
                BufferDataPrintMode::kSignedInteger);
  RoundTripTest(ShapedBuffer::Create<int16_t>({4}, {0, 1, 2, 3}),
                BufferDataPrintMode::kSignedInteger);
  RoundTripTest(ShapedBuffer::Create<uint16_t>({4}, {0, 1, 2, 3}),
                BufferDataPrintMode::kSignedInteger);
  RoundTripTest(ShapedBuffer::Create<float>({4}, {0, 1.1, 2, 3}),
                BufferDataPrintMode::kSignedInteger);
  RoundTripTest(ShapedBuffer::Create<int8_t>({1, 2, 3}, {0, 1, 2, 3, 4, 5}),
                BufferDataPrintMode::kSignedInteger);
  RoundTripTest(ShapedBuffer(1, {4}, {0, 1, 2, 3}),
                BufferDataPrintMode::kBinary);
}

}  // namespace
}  // namespace iree
