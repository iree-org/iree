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

#include "iree/hal/vmla/op_kernels.h"

#include "absl/container/inlined_vector.h"
#include "iree/base/memory.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace vmla {
namespace kernels {

namespace {

constexpr float kEpsilon = 0.0001f;

using Shape = absl::InlinedVector<int32_t, 6>;

size_t GetShapeElementCount(const Shape& shape) {
  size_t count = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    count *= shape[i];
  }
  return count;
}

template <typename T>
std::vector<T> MakeIota(int size) {
  std::vector<T> v(size);
  std::iota(v.begin(), v.end(), static_cast<T>(1));
  return v;
}

TEST(Copy, WholeBuffer) {
  Shape src_shape = {2, 2};
  auto src_buffer = MakeIota<uint8_t>(4);
  std::vector<int32_t> src_indices = {0, 0};
  Shape dst_shape = src_shape;
  std::vector<uint8_t> dst_buffer(GetShapeElementCount(dst_shape));
  std::vector<int32_t> dst_indices = {0, 0};
  std::vector<int32_t> lengths = {2, 2};
  auto expected_dst = src_buffer;

  IREE_EXPECT_OK(Copy::Execute<1>(src_buffer, src_shape, src_indices,
                                  absl::MakeSpan(dst_buffer), dst_shape,
                                  dst_indices, lengths));
  EXPECT_EQ(dst_buffer, expected_dst);
}

TEST(Copy, FirstRow) {
  Shape src_shape = {3, 4};
  auto src_buffer = MakeIota<uint8_t>(12);
  std::vector<int32_t> src_indices = {0, 0};
  Shape dst_shape = {1, 4};
  std::vector<uint8_t> dst_buffer(GetShapeElementCount(dst_shape));
  std::vector<int32_t> dst_indices = {0, 0};
  std::vector<int32_t> lengths = {1, 4};
  std::vector<uint8_t> expected_dst = {1, 2, 3, 4};

  IREE_EXPECT_OK(Copy::Execute<1>(src_buffer, src_shape, src_indices,
                                  absl::MakeSpan(dst_buffer), dst_shape,
                                  dst_indices, lengths));
  EXPECT_EQ(dst_buffer, expected_dst);
}

TEST(Copy, RowPart) {
  Shape src_shape = {3, 4};
  auto src_buffer = MakeIota<uint8_t>(12);
  std::vector<int32_t> src_indices = {1, 1};
  Shape dst_shape = {1, 2};
  std::vector<uint8_t> dst_buffer(GetShapeElementCount(dst_shape));
  std::vector<int32_t> dst_indices = {0, 0};
  std::vector<int32_t> lengths = {1, 2};
  std::vector<uint8_t> expected_dst = {6, 7};

  IREE_EXPECT_OK(Copy::Execute<1>(src_buffer, src_shape, src_indices,
                                  absl::MakeSpan(dst_buffer), dst_shape,
                                  dst_indices, lengths));
  EXPECT_EQ(dst_buffer, expected_dst);
}

TEST(Copy, MultiRow) {
  Shape src_shape = {3, 4};
  auto src_buffer = MakeIota<uint8_t>(12);
  std::vector<int32_t> src_indices = {1, 0};
  Shape dst_shape = {2, 4};
  std::vector<uint8_t> dst_buffer(GetShapeElementCount(dst_shape));
  std::vector<int32_t> dst_indices = {0, 0};
  std::vector<int32_t> lengths = {2, 4};
  std::vector<uint8_t> expected_dst = {5, 6, 7, 8, 9, 10, 11, 12};

  IREE_EXPECT_OK(Copy::Execute<1>(src_buffer, src_shape, src_indices,
                                  absl::MakeSpan(dst_buffer), dst_shape,
                                  dst_indices, lengths));
  EXPECT_EQ(dst_buffer, expected_dst);
}

TEST(Copy, NonContiguous) {
  Shape src_shape = {3, 4};
  auto src_buffer = MakeIota<uint8_t>(12);
  std::vector<int32_t> src_indices = {1, 1};
  Shape dst_shape = {2, 2};
  std::vector<uint8_t> dst_buffer(GetShapeElementCount(dst_shape));
  std::vector<int32_t> dst_indices = {0, 0};
  std::vector<int32_t> lengths = {2, 2};
  std::vector<uint8_t> expected_dst = {6, 7, 10, 11};

  IREE_EXPECT_OK(Copy::Execute<1>(src_buffer, src_shape, src_indices,
                                  absl::MakeSpan(dst_buffer), dst_shape,
                                  dst_indices, lengths));
  EXPECT_EQ(dst_buffer, expected_dst);
}

TEST(Copy, MultiByte) {
  Shape src_shape = {3, 4};
  auto src_vals = MakeIota<int32_t>(12);
  auto src_buffer = ReinterpretSpan<uint8_t>(absl::MakeSpan(src_vals));
  std::vector<int32_t> src_indices = {1, 1};
  Shape dst_shape = {2, 2};
  std::vector<uint8_t> dst_buffer(GetShapeElementCount(dst_shape) *
                                  sizeof(int32_t));
  std::vector<int32_t> dst_indices = {0, 0};
  std::vector<int32_t> lengths = {2, 2};
  std::vector<int32_t> expected_dst = {6, 7, 10, 11};

  IREE_EXPECT_OK(Copy::Execute<4>(src_buffer, src_shape, src_indices,
                                  absl::MakeSpan(dst_buffer), dst_shape,
                                  dst_indices, lengths));

  absl::Span<int32_t> dst_buffer_int32_t =
      ReinterpretSpan<int32_t>(absl::MakeSpan(dst_buffer));

  EXPECT_EQ(dst_buffer_int32_t, expected_dst);
}

TEST(Copy, NotFullDst) {
  Shape src_shape = {3, 4};
  auto src_buffer = MakeIota<uint8_t>(12);
  std::vector<int32_t> src_indices = {0, 0};
  Shape dst_shape = {4, 3};
  std::vector<uint8_t> dst_buffer(12, 42);
  std::vector<int32_t> dst_indices = {1, 1};
  std::vector<int32_t> lengths = {2, 2};
  // clang-format off
  std::vector<uint8_t> expected_dst = {42, 42, 42,
                                     42,  1,  2,
                                     42,  5,  6,
                                     42, 42, 42};
  // clang-format on

  IREE_EXPECT_OK(Copy::Execute<1>(src_buffer, src_shape, src_indices,
                                  absl::MakeSpan(dst_buffer), dst_shape,
                                  dst_indices, lengths));
  EXPECT_EQ(dst_buffer, expected_dst);
}

TEST(Copy, HighRank) {
  Shape src_shape = {3, 3, 3, 3};
  auto src_buffer = MakeIota<uint8_t>(81);
  std::vector<int32_t> src_indices = {1, 1, 1, 1};
  Shape dst_shape = {2, 2, 2, 2};
  std::vector<uint8_t> dst_buffer(GetShapeElementCount(dst_shape));
  std::vector<int32_t> dst_indices = {0, 0, 0, 0};
  std::vector<int32_t> lengths = {2, 2, 2, 2};
  std::vector<uint8_t> expected_dst = {41, 42, 44, 45, 50, 51, 53, 54,
                                       68, 69, 71, 72, 77, 78, 80, 81};

  IREE_EXPECT_OK(Copy::Execute<1>(src_buffer, src_shape, src_indices,
                                  absl::MakeSpan(dst_buffer), dst_shape,
                                  dst_indices, lengths));
  EXPECT_EQ(dst_buffer, expected_dst);
}

TEST(Copy, Scalar) {
  Shape src_shape = {};
  std::vector<uint8_t> src_buffer = {42};
  std::vector<int32_t> src_indices = {};
  Shape dst_shape = {};
  std::vector<uint8_t> dst_buffer(GetShapeElementCount(dst_shape));
  std::vector<int32_t> dst_indices = {};
  std::vector<int32_t> lengths = {};
  std::vector<uint8_t> expected_dst = {42};

  IREE_EXPECT_OK(Copy::Execute<1>(src_buffer, src_shape, src_indices,
                                  absl::MakeSpan(dst_buffer), dst_shape,
                                  dst_indices, lengths));
  EXPECT_EQ(dst_buffer, expected_dst);
}

TEST(Copy, ScalarMultiByte) {
  Shape src_shape = {};
  std::vector<int32_t> src_vals = {INT32_MAX};
  auto src_buffer = ReinterpretSpan<uint8_t>(absl::MakeSpan(src_vals));
  std::vector<int32_t> src_indices = {};
  Shape dst_shape = {};
  std::vector<uint8_t> dst_buffer(sizeof(int32_t));
  std::vector<int32_t> dst_indices = {};
  std::vector<int32_t> lengths = {};
  std::vector<int32_t> expected_dst = {INT32_MAX};

  IREE_EXPECT_OK(Copy::Execute<4>(src_buffer, src_shape, src_indices,
                                  absl::MakeSpan(dst_buffer), dst_shape,
                                  dst_indices, lengths));

  absl::Span<int32_t> dst_buffer_int32_t =
      ReinterpretSpan<int32_t>(absl::MakeSpan(dst_buffer));

  EXPECT_EQ(dst_buffer_int32_t, expected_dst);
}

TEST(Pad, NoPadding) {
  Shape src_shape = {2, 3};
  auto src_buffer = MakeIota<uint16_t>(GetShapeElementCount(src_shape));
  std::vector<uint16_t> pad_value_buffer = {0};
  std::vector<int32_t> edge_padding_low = {0, 0};
  std::vector<int32_t> edge_padding_high = {0, 0};
  std::vector<int32_t> interior_padding = {0, 0};
  Shape dst_shape = src_shape;
  std::vector<uint16_t> dst_buffer(GetShapeElementCount(dst_shape), UINT16_MAX);
  auto expected_dst = src_buffer;

  IREE_EXPECT_OK(Pad::Execute<uint16_t>(
      src_buffer, pad_value_buffer, absl::MakeSpan(dst_buffer), src_shape,
      dst_shape, edge_padding_low, edge_padding_high, interior_padding));
  EXPECT_EQ(dst_buffer, expected_dst);
}

TEST(Pad, LowHighPadding) {
  Shape src_shape = {2, 3};
  auto src_buffer = MakeIota<uint16_t>(GetShapeElementCount(src_shape));
  std::vector<uint16_t> pad_value_buffer = {0};
  std::vector<int32_t> edge_padding_low = {0, 1};
  std::vector<int32_t> edge_padding_high = {1, 2};
  std::vector<int32_t> interior_padding = {0, 0};
  Shape dst_shape = {3, 6};
  std::vector<uint16_t> dst_buffer(GetShapeElementCount(dst_shape), UINT16_MAX);
  // clang-format off
  std::vector<uint16_t> expected_dst = {0, 1, 2, 3, 0, 0,
                                      0, 4, 5, 6, 0, 0,
                                      0, 0, 0, 0, 0, 0};
  // clang-format on

  IREE_EXPECT_OK(Pad::Execute<uint16_t>(
      src_buffer, pad_value_buffer, absl::MakeSpan(dst_buffer), src_shape,
      dst_shape, edge_padding_low, edge_padding_high, interior_padding));
  EXPECT_EQ(dst_buffer, expected_dst);
}

TEST(Pad, OnlyHighPadding) {
  Shape src_shape = {2, 3};
  auto src_buffer = MakeIota<uint16_t>(GetShapeElementCount(src_shape));
  std::vector<uint16_t> pad_value_buffer = {0};
  std::vector<int32_t> edge_padding_low = {0, 0};
  std::vector<int32_t> edge_padding_high = {1, 3};
  std::vector<int32_t> interior_padding = {0, 0};
  Shape dst_shape = {3, 6};
  std::vector<uint16_t> dst_buffer(GetShapeElementCount(dst_shape), UINT16_MAX);
  // clang-format off
  std::vector<uint16_t> expected_dst = {1, 2, 3, 0, 0, 0,
                                      4, 5, 6, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0};
  // clang-format on

  IREE_EXPECT_OK(Pad::Execute<uint16_t>(
      src_buffer, pad_value_buffer, absl::MakeSpan(dst_buffer), src_shape,
      dst_shape, edge_padding_low, edge_padding_high, interior_padding));
  EXPECT_EQ(dst_buffer, expected_dst);
}

TEST(Pad, OnlyLowPadding) {
  Shape src_shape = {2, 3};
  auto src_buffer = MakeIota<uint16_t>(GetShapeElementCount(src_shape));
  std::vector<uint16_t> pad_value_buffer = {0};
  std::vector<int32_t> edge_padding_low = {1, 3};
  std::vector<int32_t> edge_padding_high = {0, 0};
  std::vector<int32_t> interior_padding = {0, 0};
  Shape dst_shape = {3, 6};
  std::vector<uint16_t> dst_buffer(GetShapeElementCount(dst_shape), UINT16_MAX);
  // clang-format off
  std::vector<uint16_t> expected_dst = {0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 1, 2, 3,
                                      0, 0, 0, 4, 5, 6};
  // clang-format on

  IREE_EXPECT_OK(Pad::Execute<uint16_t>(
      src_buffer, pad_value_buffer, absl::MakeSpan(dst_buffer), src_shape,
      dst_shape, edge_padding_low, edge_padding_high, interior_padding));
  EXPECT_EQ(dst_buffer, expected_dst);
}

TEST(Pad, OnlyInteriorPadding) {
  Shape src_shape = {2, 3};
  auto src_buffer = MakeIota<uint16_t>(GetShapeElementCount(src_shape));
  std::vector<uint16_t> pad_value_buffer = {0};
  std::vector<int32_t> edge_padding_low = {0, 0};
  std::vector<int32_t> edge_padding_high = {0, 0};
  std::vector<int32_t> interior_padding = {1, 1};
  Shape dst_shape = {3, 5};
  std::vector<uint16_t> dst_buffer(GetShapeElementCount(dst_shape), UINT16_MAX);
  // clang-format off
  std::vector<uint16_t> expected_dst = {1, 0, 2, 0, 3,
                                      0, 0, 0, 0, 0,
                                      4, 0, 5, 0, 6};
  // clang-format on

  IREE_EXPECT_OK(Pad::Execute<uint16_t>(
      src_buffer, pad_value_buffer, absl::MakeSpan(dst_buffer), src_shape,
      dst_shape, edge_padding_low, edge_padding_high, interior_padding));
  EXPECT_EQ(dst_buffer, expected_dst);
}

TEST(Pad, AllPaddingTypes) {
  Shape src_shape = {2, 3};
  auto src_buffer = MakeIota<uint16_t>(GetShapeElementCount(src_shape));
  std::vector<uint16_t> pad_value_buffer = {0};
  std::vector<int32_t> edge_padding_low = {1, 1};
  std::vector<int32_t> edge_padding_high = {1, 2};
  std::vector<int32_t> interior_padding = {1, 1};
  Shape dst_shape = {5, 8};
  std::vector<uint16_t> dst_buffer(GetShapeElementCount(dst_shape), UINT16_MAX);
  // clang-format off
  std::vector<uint16_t> expected_dst = {0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 1, 0, 2, 0, 3, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 4, 0, 5, 0, 6, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0};
  // clang-format on

  IREE_EXPECT_OK(Pad::Execute<uint16_t>(
      src_buffer, pad_value_buffer, absl::MakeSpan(dst_buffer), src_shape,
      dst_shape, edge_padding_low, edge_padding_high, interior_padding));
  EXPECT_EQ(dst_buffer, expected_dst);
}

TEST(Pad, HighRank) {
  Shape src_shape = {2, 2, 2, 2};
  auto src_buffer = MakeIota<uint16_t>(GetShapeElementCount(src_shape));
  std::vector<uint16_t> pad_value_buffer = {0};
  std::vector<int32_t> edge_padding_low = {1, 0, 0, 0};
  std::vector<int32_t> edge_padding_high = {0, 1, 0, 0};
  std::vector<int32_t> interior_padding = {0, 0, 1, 0};
  Shape dst_shape = {3, 3, 3, 2};
  std::vector<uint16_t> dst_buffer(GetShapeElementCount(dst_shape), UINT16_MAX);
  // clang-format off
  std::vector<uint16_t> expected_dst = { 0,  0,   0, 0,   0,  0,
                                       0,  0,   0, 0,   0,  0,
                                       0,  0,   0, 0,   0,  0,

                                       1,  2,   0, 0,   3,  4,
                                       5,  6,   0, 0,   7,  8,
                                       0,  0,   0, 0,   0,  0,

                                       9, 10,   0, 0,  11, 12,
                                      13, 14,   0, 0,  15, 16,
                                       0,  0,   0, 0,   0,  0};
  // clang-format on

  ASSERT_EQ(dst_buffer.size(), expected_dst.size());

  IREE_EXPECT_OK(Pad::Execute<uint16_t>(
      src_buffer, pad_value_buffer, absl::MakeSpan(dst_buffer), src_shape,
      dst_shape, edge_padding_low, edge_padding_high, interior_padding));
  EXPECT_EQ(dst_buffer, expected_dst);
}

TEST(ReduceSum, Scalar) {
  Shape src_shape = {5};
  int32_t dimension = 0;
  Shape dst_shape = {1};
  std::vector<float> src_buffer = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> init_buffer = {0.0f};
  std::vector<float> dst_buffer(GetShapeElementCount(dst_shape), 0.0f);
  std::vector<float> expected_dst = {5.0f};

  IREE_EXPECT_OK(ReduceSum::Execute<float>(src_buffer, init_buffer,
                                           absl::MakeSpan(dst_buffer),
                                           dimension, src_shape, dst_shape));

  for (int i = 0; i < dst_buffer.size(); ++i) {
    EXPECT_NEAR(expected_dst[i], dst_buffer[i], kEpsilon);
  }
}

TEST(ReduceMin, TwoDimensionsToOne) {
  Shape src_shape = {3, 3};
  int32_t dimension = 0;
  Shape dst_shape = {3};
  std::vector<float> src_buffer =
      MakeIota<float>(GetShapeElementCount(src_shape));
  std::vector<float> init_buffer = {std::numeric_limits<float>::max()};
  std::vector<float> dst_buffer(GetShapeElementCount(dst_shape), 0.0f);
  std::vector<float> expected_dst = {1.0f, 2.0f, 3.0f};

  IREE_EXPECT_OK(ReduceMin::Execute<float>(src_buffer, init_buffer,
                                           absl::MakeSpan(dst_buffer),
                                           dimension, src_shape, dst_shape));

  for (int i = 0; i < dst_buffer.size(); ++i) {
    EXPECT_NEAR(expected_dst[i], dst_buffer[i], kEpsilon);
  }
}

TEST(PoolingMax, NoOverlapping) {
  Shape src_shape = {1, 4, 6, 1};
  Shape dst_shape = {1, 2, 2, 1};
  Shape window_sizes = {1, 2, 3, 1};
  Shape strides = {1, 2, 3, 1};
  Shape pad_low = {0, 0, 0, 0};
  std::vector<int> src_buffer = MakeIota<int>(GetShapeElementCount(src_shape));
  std::vector<int> init_buffer(1, 0.0f);
  std::vector<int> dst_buffer(GetShapeElementCount(dst_shape), 0.0f);
  std::vector<int> expected_dst = {9, 12, 21, 24};

  IREE_EXPECT_OK(PoolingMax::Execute<int>(
      src_buffer, init_buffer, absl::MakeSpan(dst_buffer), src_shape, dst_shape,
      window_sizes, strides, pad_low));
  EXPECT_EQ(dst_buffer, expected_dst);
}

TEST(PoolingMin, Padding) {
  // Padded input:
  // 100 100 100 100
  // 100   1   2   3
  // 100   4   5   6
  Shape src_shape = {2, 3};
  Shape dst_shape = {2, 3};
  Shape window_sizes = {2, 2};
  Shape strides = {1, 1};
  Shape pad_low = {1, 1};
  std::vector<int> src_buffer = MakeIota<int>(GetShapeElementCount(src_shape));
  std::vector<int> init_buffer(1, 100.0);
  std::vector<int> dst_buffer(GetShapeElementCount(dst_shape), 0.0f);
  std::vector<int> expected_dst = {1, 1, 2, 1, 1, 2};

  IREE_EXPECT_OK(PoolingMin::Execute<int>(
      src_buffer, init_buffer, absl::MakeSpan(dst_buffer), src_shape, dst_shape,
      window_sizes, strides, pad_low));
  EXPECT_EQ(dst_buffer, expected_dst);
}

TEST(PoolingSum, Overlapping) {
  Shape src_shape = {3, 4};
  Shape dst_shape = {2, 2};
  Shape window_sizes = {2, 3};
  Shape strides = {1, 1};
  Shape pad_low = {0, 0};
  std::vector<float> src_buffer =
      MakeIota<float>(GetShapeElementCount(src_shape));
  std::vector<float> init_buffer(1, 0.0f);
  std::vector<float> dst_buffer(GetShapeElementCount(dst_shape), 0.0f);
  std::vector<float> expected_dst = {24, 30, 48, 54};

  IREE_EXPECT_OK(PoolingSum::Execute<float>(
      src_buffer, init_buffer, absl::MakeSpan(dst_buffer), src_shape, dst_shape,
      window_sizes, strides, pad_low));
  for (int i = 0; i < dst_buffer.size(); ++i) {
    EXPECT_NEAR(expected_dst[i], dst_buffer[i], kEpsilon);
  }
}

TEST(Conv2d, NoDilation) {
  Shape input_shape = {4, 5, 2};
  Shape filter_shape = {3, 2, 2, 1};
  Shape dst_shape = {2, 4, 1};
  Shape strides = {1, 1};
  Shape pad_h = {0, 0};
  Shape pad_w = {0, 0};
  Shape dilation = {1, 1};
  std::vector<float> input_buffer(GetShapeElementCount(input_shape));
  std::vector<float> filter_buffer(GetShapeElementCount(filter_shape));
  std::vector<float> expected_dst = {1310, 1466, 1622, 1778,
                                     2090, 2246, 2402, 2558};
  for (int i = 0; i < GetShapeElementCount(input_shape); ++i) {
    input_buffer[i] = i + 1;
    if (i < GetShapeElementCount(filter_shape)) {
      filter_buffer[i] = i + 1;
    }
  }
  std::vector<float> dst_buffer(GetShapeElementCount(dst_shape), 0.0f);

  IREE_EXPECT_OK(Conv2D::Execute<float>(input_buffer, input_shape,
                                        filter_buffer, filter_shape,
                                        absl::MakeSpan(dst_buffer), dst_shape,
                                        strides, pad_h, pad_w, dilation, 1));

  for (int i = 0; i < dst_buffer.size(); ++i) {
    EXPECT_NEAR(expected_dst[i], dst_buffer[i], kEpsilon);
  }
}

TEST(Conv2d, DepthwiseConv) {
  Shape input_shape = {4, 5, 2};
  Shape filter_shape = {3, 2, 2, 2};
  Shape dst_shape = {2, 4, 4};
  Shape strides = {1, 1};
  Shape pad_h = {0, 0};
  Shape pad_w = {0, 0};
  Shape dilation = {1, 1};
  std::vector<float> input_buffer(GetShapeElementCount(input_shape));
  std::vector<float> filter_buffer(GetShapeElementCount(filter_shape));
  std::vector<float> expected_dst = {
      1124, 1196, 1346, 1424, 1256, 1340, 1502, 1592, 1388, 1484, 1658,
      1760, 1520, 1628, 1814, 1928, 1784, 1916, 2126, 2264, 1916, 2060,
      2282, 2432, 2048, 2204, 2438, 2600, 2180, 2348, 2594, 2768};
  for (int i = 0; i < GetShapeElementCount(input_shape); ++i) {
    input_buffer[i] = i + 1;
    if (i < GetShapeElementCount(filter_shape)) {
      filter_buffer[i] = i + 1;
    }
  }
  std::vector<float> dst_buffer(GetShapeElementCount(dst_shape), 0.0f);

  IREE_EXPECT_OK(Conv2D::Execute<float>(input_buffer, input_shape,
                                        filter_buffer, filter_shape,
                                        absl::MakeSpan(dst_buffer), dst_shape,
                                        strides, pad_h, pad_w, dilation, 2));

  for (int i = 0; i < dst_buffer.size(); ++i) {
    EXPECT_NEAR(expected_dst[i], dst_buffer[i], kEpsilon);
  }
}

}  // namespace
}  // namespace kernels
}  // namespace vmla
}  // namespace hal
}  // namespace iree
