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

#include "iree/hal/buffer_view.h"

#include <numeric>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "iree/base/status.h"
#include "iree/base/status_matchers.h"
#include "iree/hal/buffer.h"
#include "iree/hal/heap_buffer.h"

namespace iree {
namespace hal {
namespace {

template <typename T>
BufferView MakeView(const std::vector<T> src_data, Shape shape) {
  auto parent_buffer = HeapBuffer::AllocateCopy(
      BufferUsage::kTransfer | BufferUsage::kMapping, absl::MakeSpan(src_data));

  return BufferView(std::move(parent_buffer), shape, sizeof(T));
}

template <typename T>
std::vector<T> ReadData(BufferView view) {
  std::vector<T> data(view.shape.element_count());
  EXPECT_OK(view.buffer->ReadData(0, data.data(), data.size() * sizeof(T)));
  return data;
}

TEST(BufferViewTest, SliceWholeBuffer) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3};
  Shape shape = {2, 2};
  auto parent_view = MakeView(src_data, shape);

  std::vector<int32_t> start_indices = {0, 0};
  std::vector<int32_t> lengths = {2, 2};
  ASSERT_OK_AND_ASSIGN(auto slice, parent_view.Slice(start_indices, lengths));

  EXPECT_TRUE(BufferView::Equal(parent_view, slice))
      << "original parent_view " << parent_view.DebugStringShort()
      << " and whole slice " << slice.DebugStringShort() << " are not equal";
}

TEST(BufferViewTest, SliceSingleRow) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3};
  Shape shape = {2, 2};
  auto parent_view = MakeView(src_data, shape);

  std::vector<int32_t> start_indices = {1, 0};
  std::vector<int32_t> lengths = {1, 2};
  ASSERT_OK_AND_ASSIGN(auto slice, parent_view.Slice(start_indices, lengths));

  EXPECT_EQ(ReadData<uint8_t>(slice), std::vector<uint8_t>({2, 3}));
}

TEST(BufferViewTest, SliceRowStart) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3, 4, 5, 6, 7};
  Shape shape = {2, 4};
  auto parent_view = MakeView(src_data, shape);

  std::vector<int32_t> start_indices = {1, 0};
  std::vector<int32_t> lengths = {1, 3};
  ASSERT_OK_AND_ASSIGN(auto slice, parent_view.Slice(start_indices, lengths));

  EXPECT_EQ(ReadData<uint8_t>(slice), std::vector<uint8_t>({4, 5, 6}));
}

TEST(BufferViewTest, SliceRowEnd) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3, 4, 5, 6, 7};
  Shape shape = {2, 4};
  auto parent_view = MakeView(src_data, shape);

  std::vector<int32_t> start_indices = {1, 1};
  std::vector<int32_t> lengths = {1, 3};
  ASSERT_OK_AND_ASSIGN(auto slice, parent_view.Slice(start_indices, lengths));

  EXPECT_EQ(ReadData<uint8_t>(slice), std::vector<uint8_t>({5, 6, 7}));
}

TEST(BufferViewTest, SliceRowMiddle) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3, 4, 5, 6, 7};
  Shape shape = {2, 4};
  auto parent_view = MakeView(src_data, shape);

  std::vector<int32_t> start_indices = {1, 1};
  std::vector<int32_t> lengths = {1, 2};
  ASSERT_OK_AND_ASSIGN(auto slice, parent_view.Slice(start_indices, lengths));

  EXPECT_EQ(ReadData<uint8_t>(slice), std::vector<uint8_t>({5, 6}));
}

TEST(BufferViewTest, SliceMultiRow) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  Shape shape = {3, 3};
  auto parent_view = MakeView(src_data, shape);

  std::vector<int32_t> start_indices = {1, 0};
  std::vector<int32_t> lengths = {2, 3};
  ASSERT_OK_AND_ASSIGN(auto slice, parent_view.Slice(start_indices, lengths));

  EXPECT_EQ(ReadData<uint8_t>(slice), std::vector<uint8_t>({3, 4, 5, 6, 7, 8}));
}

TEST(BufferViewTest, SliceHighRank) {
  std::vector<uint8_t> src_data(81);
  std::iota(src_data.begin(), src_data.end(), 0);
  Shape shape = {3, 3, 3, 3};

  auto parent_view = MakeView(src_data, shape);

  std::vector<int32_t> start_indices = {1, 2, 2, 1};
  std::vector<int32_t> lengths = {1, 1, 1, 2};
  ASSERT_OK_AND_ASSIGN(auto slice, parent_view.Slice(start_indices, lengths));

  EXPECT_EQ(ReadData<uint8_t>(slice), std::vector<uint8_t>({52, 53}));
}

TEST(BufferViewTest, SliceModifySlice) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3};
  Shape shape = {2, 2};
  auto parent_view = MakeView(src_data, shape);

  std::vector<int32_t> start_indices = {1, 0};
  std::vector<int32_t> lengths = {1, 2};
  ASSERT_OK_AND_ASSIGN(auto slice, parent_view.Slice(start_indices, lengths));

  EXPECT_OK(slice.buffer->Fill8(0, kWholeBuffer, 0xFFu));

  auto parent_data = ReadData<uint8_t>(parent_view);
  EXPECT_EQ(parent_data, std::vector<uint8_t>({0, 1, 0xFFu, 0xFFu}));
}

TEST(BufferViewTest, SliceModifyParent) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3};
  Shape shape = {2, 2};
  auto parent_view = MakeView(src_data, shape);

  std::vector<int32_t> start_indices = {1, 0};
  std::vector<int32_t> lengths = {1, 2};
  ASSERT_OK_AND_ASSIGN(auto slice, parent_view.Slice(start_indices, lengths));

  EXPECT_OK(parent_view.buffer->Fill8(0, kWholeBuffer, 0xFFu));

  EXPECT_EQ(ReadData<uint8_t>(slice), std::vector<uint8_t>({0xFFu, 0xFFu}));
}

TEST(BufferViewTest, SliceMultiByteElementWholeBuffer) {
  const std::vector<int32_t> src_data = {INT32_MAX, 1, 2, 3};

  Shape shape = {2, 2};
  auto parent_view = MakeView(src_data, shape);

  std::vector<int32_t> start_indices = {0, 0};
  std::vector<int32_t> lengths = {2, 2};
  ASSERT_OK_AND_ASSIGN(auto slice, parent_view.Slice(start_indices, lengths));

  EXPECT_TRUE(BufferView::Equal(parent_view, slice))
      << "original parent_view " << parent_view.DebugStringShort()
      << " and whole slice " << slice.DebugStringShort() << " are not equal";
}

TEST(BufferViewTest, SliceShapeAndElementSize) {
  std::vector<int32_t> src_data = {INT32_MAX, 1, 2, 3};
  Shape shape = {2, 2};
  auto parent_view = MakeView(src_data, shape);

  std::vector<int32_t> start_indices = {1, 0};
  std::vector<int32_t> lengths = {1, 2};
  ASSERT_OK_AND_ASSIGN(auto slice, parent_view.Slice(start_indices, lengths));
  EXPECT_EQ(slice.shape, Shape(lengths));
  EXPECT_EQ(slice.element_size, 4);
}

TEST(BufferViewTest, SliceMultiByteElement) {
  std::vector<int32_t> src_data = {INT32_MAX, 1, 2, 3};
  Shape shape = {2, 2};
  auto parent_view = MakeView(src_data, shape);

  std::vector<int32_t> start_indices = {1, 0};
  std::vector<int32_t> lengths = {1, 2};
  ASSERT_OK_AND_ASSIGN(auto slice, parent_view.Slice(start_indices, lengths));

  EXPECT_EQ(ReadData<int32_t>(slice), std::vector<int32_t>({2, 3}));
}

TEST(BufferViewTest, SliceIndexBadRank) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3};
  Shape shape = {2, 2};
  auto parent_view = MakeView(src_data, shape);

  std::vector<int32_t> start_indices = {0};
  std::vector<int32_t> lengths = {2};
  EXPECT_TRUE(
      IsInvalidArgument(parent_view.Slice(start_indices, lengths).status()));
}

TEST(BufferViewTest, SliceIndexLengthMismatch) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3};
  Shape shape = {2, 2};
  auto parent_view = MakeView(src_data, shape);

  std::vector<int32_t> start_indices = {0, 0};
  std::vector<int32_t> lengths = {2};
  EXPECT_TRUE(
      IsInvalidArgument(parent_view.Slice(start_indices, lengths).status()));
}

TEST(BufferViewTest, SliceIndicesOutOfBounds) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3};
  Shape shape = {2, 2};

  auto parent_view = MakeView(src_data, shape);

  std::vector<int32_t> start_indices = {0, 3};
  std::vector<int32_t> lengths = {1, 1};
  EXPECT_TRUE(
      IsInvalidArgument(parent_view.Slice(start_indices, lengths).status()));
}

TEST(BufferViewTest, SliceLengthsOutOfBounds) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3};
  Shape shape = {2, 2};

  auto parent_view = MakeView(src_data, shape);

  std::vector<int32_t> start_indices = {0, 0};
  std::vector<int32_t> lengths = {1, 3};
  EXPECT_TRUE(
      IsInvalidArgument(parent_view.Slice(start_indices, lengths).status()));
}

TEST(BufferViewTest, SliceNonContiguous) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  Shape shape = {3, 3};
  auto parent_view = MakeView(src_data, shape);

  std::vector<int32_t> start_indices = {1, 1};
  std::vector<int32_t> lengths = {2, 2};
  EXPECT_TRUE(
      IsUnimplemented(parent_view.Slice(start_indices, lengths).status()));
}

TEST(BufferViewTest, SliceNonContiguousMultiRowLeft) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  Shape shape = {3, 3};
  auto parent_view = MakeView(src_data, shape);

  std::vector<int32_t> start_indices = {1, 0};
  std::vector<int32_t> lengths = {2, 1};
  EXPECT_TRUE(
      IsUnimplemented(parent_view.Slice(start_indices, lengths).status()));
}

TEST(BufferViewTest, SliceHighRankNonContiguous) {
  std::vector<uint8_t> src_data(81);
  std::iota(src_data.begin(), src_data.end(), 0);
  Shape shape = {3, 3, 3, 3};

  auto parent_view = MakeView(src_data, shape);

  std::vector<int32_t> start_indices = {1, 0, 2, 1};
  std::vector<int32_t> lengths = {1, 2, 1, 2};
  EXPECT_TRUE(
      IsUnimplemented(parent_view.Slice(start_indices, lengths).status()));
}

}  // namespace
}  // namespace hal
}  // namespace iree
