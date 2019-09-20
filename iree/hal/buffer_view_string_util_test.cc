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

#include "iree/hal/buffer_view_string_util.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "iree/base/status.h"
#include "iree/base/status_matchers.h"

namespace iree {
namespace hal {
namespace {

using ::testing::ElementsAre;

template <typename T>
StatusOr<std::vector<T>> ReadBuffer(const ref_ptr<Buffer>& buffer) {
  std::vector<T> result;
  result.resize(buffer->byte_length() / sizeof(T));
  RETURN_IF_ERROR(
      buffer->ReadData(0, result.data(), result.size() * sizeof(T)));
  return result;
}

TEST(BufferViewUtilTest, GetTypeElementSize) {
  EXPECT_EQ(1, GetTypeElementSize("1").ValueOrDie());
  EXPECT_EQ(7, GetTypeElementSize("7").ValueOrDie());
  EXPECT_EQ(4, GetTypeElementSize("i32").ValueOrDie());
  EXPECT_EQ(8, GetTypeElementSize("f64").ValueOrDie());

  EXPECT_FALSE(GetTypeElementSize("").ok());
  EXPECT_FALSE(GetTypeElementSize(" ").ok());
  EXPECT_FALSE(GetTypeElementSize("a").ok());
  EXPECT_FALSE(GetTypeElementSize("ib").ok());
  EXPECT_FALSE(GetTypeElementSize("i").ok());
  EXPECT_FALSE(GetTypeElementSize("i543ff").ok());
}

TEST(BufferViewUtilTest, ParseShape) {
  EXPECT_EQ((Shape{}), ParseShape("").ValueOrDie());
  EXPECT_EQ((Shape{1}), ParseShape("1").ValueOrDie());
  EXPECT_EQ((Shape{1, 2}), ParseShape("1x2").ValueOrDie());
  EXPECT_EQ((Shape{1, 2}), ParseShape(" 1 x 2 ").ValueOrDie());

  EXPECT_FALSE(ParseShape("abc").ok());
  EXPECT_FALSE(ParseShape("1xf").ok());
  EXPECT_FALSE(ParseShape("1xff23").ok());
}

TEST(BufferViewUtilTest, ParseBufferViewFromStringEmpty) {
  // Empty string = empty buffer_view.
  ASSERT_OK_AND_ASSIGN(auto m0, ParseBufferViewFromString(""));
  EXPECT_EQ(nullptr, m0.buffer.get());
  EXPECT_EQ(Shape{}, m0.shape);
  EXPECT_EQ(0, m0.element_size);

  // No = means no data.
  ASSERT_OK_AND_ASSIGN(auto m1, ParseBufferViewFromString("4x2xf32"));
  EXPECT_EQ(4 * 2 * 4, m1.buffer->allocation_size());
  EXPECT_EQ(Shape({4, 2}), m1.shape);
  EXPECT_EQ(4, m1.element_size);
  EXPECT_THAT(ReadBuffer<float>(m1.buffer).ValueOrDie(),
              ElementsAre(0, 0, 0, 0, 0, 0, 0, 0));

  // No data after = means no data.
  ASSERT_OK_AND_ASSIGN(auto m2, ParseBufferViewFromString("4x2xf32="));
  EXPECT_EQ(4 * 2 * 4, m2.buffer->allocation_size());
  EXPECT_EQ(Shape({4, 2}), m2.shape);
  EXPECT_EQ(4, m2.element_size);
  EXPECT_THAT(ReadBuffer<float>(m2.buffer).ValueOrDie(),
              ElementsAre(0, 0, 0, 0, 0, 0, 0, 0));
}

TEST(BufferViewUtilTest, ParseBufferViewFromStringBinary) {
  ASSERT_OK_AND_ASSIGN(auto m0, ParseBufferViewFromString("4x1=00 01 02 03"));
  EXPECT_EQ(Shape({4}), m0.shape);
  EXPECT_EQ(1, m0.element_size);
  EXPECT_THAT(ReadBuffer<uint8_t>(m0.buffer).ValueOrDie(),
              ElementsAre(0, 1, 2, 3));

  // Whitespace shouldn't matter.
  ASSERT_OK_AND_ASSIGN(auto m1, ParseBufferViewFromString("4x1=00,010203"));
  EXPECT_EQ(Shape({4}), m1.shape);
  EXPECT_EQ(1, m1.element_size);
  EXPECT_THAT(ReadBuffer<uint8_t>(m1.buffer).ValueOrDie(),
              ElementsAre(0, 1, 2, 3));

  // Should fail on malformed hex bytes.
  EXPECT_FALSE(ParseBufferViewFromString("4x1=1").ok());
  EXPECT_FALSE(ParseBufferViewFromString("4x1=00003").ok());
  EXPECT_FALSE(ParseBufferViewFromString("4x1=%0123%\1").ok());
  EXPECT_FALSE(ParseBufferViewFromString("4x1=00010203040506").ok());
}

TEST(BufferViewUtilTest, ParseBufferViewFromStringAllowBrackets) {
  ASSERT_OK_AND_ASSIGN(auto m0,
                       ParseBufferViewFromString("4xi16=[[0][ 1 ][2]][3]"));
  EXPECT_EQ(Shape({4}), m0.shape);
  EXPECT_EQ(2, m0.element_size);
  EXPECT_THAT(ReadBuffer<int16_t>(m0.buffer).ValueOrDie(),
              ElementsAre(0, 1, 2, 3));
}

TEST(BufferViewUtilTest, ParseBufferViewFromStringInteger) {
  // Signed int16.
  ASSERT_OK_AND_ASSIGN(auto m0,
                       ParseBufferViewFromString("4xi16=0 12345 65535 -2"));
  EXPECT_EQ(Shape({4}), m0.shape);
  EXPECT_EQ(2, m0.element_size);
  EXPECT_THAT(ReadBuffer<int16_t>(m0.buffer).ValueOrDie(),
              ElementsAre(0, 12345, -1, -2));

  // Unsigned int16.
  ASSERT_OK_AND_ASSIGN(auto m1,
                       ParseBufferViewFromString("4xu16=0 12345 65535 -2"));
  EXPECT_EQ(Shape({4}), m1.shape);
  EXPECT_EQ(2, m1.element_size);
  EXPECT_THAT(ReadBuffer<uint16_t>(m1.buffer).ValueOrDie(),
              ElementsAre(0, 12345, 65535, 65534));

  // Mixing separator types is ok.
  ASSERT_OK_AND_ASSIGN(auto m2,
                       ParseBufferViewFromString("4xu16=0, 12345, 65535, -2"));
  EXPECT_EQ(Shape({4}), m2.shape);
  EXPECT_EQ(2, m2.element_size);
  EXPECT_THAT(ReadBuffer<uint16_t>(m2.buffer).ValueOrDie(),
              ElementsAre(0, 12345, 65535, 65534));

  // Should fail on malformed integers bytes and out of bounds values.
  EXPECT_FALSE(ParseBufferViewFromString("4xi32=asodfj").ok());
  EXPECT_FALSE(ParseBufferViewFromString("4xi32=0 1 2 3 4").ok());
}

TEST(BufferViewUtilTest, ParseBufferViewFromStringFloat) {
  // Float.
  ASSERT_OK_AND_ASSIGN(auto m0,
                       ParseBufferViewFromString("4xf32=0 1.0 1234 -2.0e-5"));
  EXPECT_EQ(Shape({4}), m0.shape);
  EXPECT_EQ(4, m0.element_size);
  EXPECT_THAT(ReadBuffer<float>(m0.buffer).ValueOrDie(),
              ElementsAre(0.0f, 1.0f, 1234.0f, -2.0e-5f));

  // Double.
  ASSERT_OK_AND_ASSIGN(auto m1, ParseBufferViewFromString(
                                    "4xf64=0 1.0 123456789012345 -2.0e-5"));
  EXPECT_EQ(Shape({4}), m1.shape);
  EXPECT_EQ(8, m1.element_size);
  EXPECT_THAT(ReadBuffer<double>(m1.buffer).ValueOrDie(),
              ElementsAre(0.0, 1.0, 123456789012345.0, -2.0e-5));

  // Should fail on malformed floats and out of bounds values.
  EXPECT_FALSE(ParseBufferViewFromString("4xf32=asodfj").ok());
  EXPECT_FALSE(ParseBufferViewFromString("4xf32=0").ok());
  EXPECT_FALSE(ParseBufferViewFromString("4xf32=0 1 2 3 4").ok());
}

TEST(BufferViewUtilTest, ParseBufferViewPrintMode) {
  EXPECT_EQ(BufferViewPrintMode::kBinary,
            ParseBufferViewPrintMode("b").ValueOrDie());
  EXPECT_EQ(BufferViewPrintMode::kSignedInteger,
            ParseBufferViewPrintMode("i").ValueOrDie());
  EXPECT_EQ(BufferViewPrintMode::kUnsignedInteger,
            ParseBufferViewPrintMode("u").ValueOrDie());
  EXPECT_EQ(BufferViewPrintMode::kFloatingPoint,
            ParseBufferViewPrintMode("f").ValueOrDie());

  EXPECT_FALSE(ParseBufferViewPrintMode("").ok());
  EXPECT_FALSE(ParseBufferViewPrintMode("s").ok());
  EXPECT_FALSE(ParseBufferViewPrintMode("asdfasdf").ok());
}

}  // namespace
}  // namespace hal
}  // namespace iree
