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

#include "iree/base/memory.h"
#include "iree/base/status_matchers.h"
#include "iree/testing/gtest.h"

namespace iree {
namespace hal {
namespace vmla {
namespace kernels {

namespace {

constexpr float kEpsilon = 0.0001f;

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
  std::vector<uint8_t> dst_buffer(dst_shape.element_count());
  std::vector<int32_t> dst_indices = {0, 0};
  std::vector<int32_t> lengths = {2, 2};
  auto expected_dst = src_buffer;

  EXPECT_OK(Copy::Execute<1>(src_buffer, src_shape, src_indices,
                             absl::MakeSpan(dst_buffer), dst_shape, dst_indices,
                             lengths));
  EXPECT_EQ(dst_buffer, expected_dst);
}

TEST(Copy, FirstRow) {
  Shape src_shape = {3, 4};
  auto src_buffer = MakeIota<uint8_t>(12);
  std::vector<int32_t> src_indices = {0, 0};
  Shape dst_shape = {1, 4};
  std::vector<uint8_t> dst_buffer(dst_shape.element_count());
  std::vector<int32_t> dst_indices = {0, 0};
  std::vector<int32_t> lengths = {1, 4};
  std::vector<uint8_t> expected_dst = {1, 2, 3, 4};

  EXPECT_OK(Copy::Execute<1>(src_buffer, src_shape, src_indices,
                             absl::MakeSpan(dst_buffer), dst_shape, dst_indices,
                             lengths));
  EXPECT_EQ(dst_buffer, expected_dst);
}

TEST(Copy, RowPart) {
  Shape src_shape = {3, 4};
  auto src_buffer = MakeIota<uint8_t>(12);
  std::vector<int32_t> src_indices = {1, 1};
  Shape dst_shape = {1, 2};
  std::vector<uint8_t> dst_buffer(dst_shape.element_count());
  std::vector<int32_t> dst_indices = {0, 0};
  std::vector<int32_t> lengths = {1, 2};
  std::vector<uint8_t> expected_dst = {6, 7};

  EXPECT_OK(Copy::Execute<1>(src_buffer, src_shape, src_indices,
                             absl::MakeSpan(dst_buffer), dst_shape, dst_indices,
                             lengths));
  EXPECT_EQ(dst_buffer, expected_dst);
}

TEST(Copy, MultiRow) {
  Shape src_shape = {3, 4};
  auto src_buffer = MakeIota<uint8_t>(12);
  std::vector<int32_t> src_indices = {1, 0};
  Shape dst_shape = {2, 4};
  std::vector<uint8_t> dst_buffer(dst_shape.element_count());
  std::vector<int32_t> dst_indices = {0, 0};
  std::vector<int32_t> lengths = {2, 4};
  std::vector<uint8_t> expected_dst = {5, 6, 7, 8, 9, 10, 11, 12};

  EXPECT_OK(Copy::Execute<1>(src_buffer, src_shape, src_indices,
                             absl::MakeSpan(dst_buffer), dst_shape, dst_indices,
                             lengths));
  EXPECT_EQ(dst_buffer, expected_dst);
}

TEST(Copy, NonContiguous) {
  Shape src_shape = {3, 4};
  auto src_buffer = MakeIota<uint8_t>(12);
  std::vector<int32_t> src_indices = {1, 1};
  Shape dst_shape = {2, 2};
  std::vector<uint8_t> dst_buffer(dst_shape.element_count());
  std::vector<int32_t> dst_indices = {0, 0};
  std::vector<int32_t> lengths = {2, 2};
  std::vector<uint8_t> expected_dst = {6, 7, 10, 11};

  EXPECT_OK(Copy::Execute<1>(src_buffer, src_shape, src_indices,
                             absl::MakeSpan(dst_buffer), dst_shape, dst_indices,
                             lengths));
  EXPECT_EQ(dst_buffer, expected_dst);
}

TEST(Copy, MultiByte) {
  Shape src_shape = {3, 4};
  auto src_vals = MakeIota<int32_t>(12);
  auto src_buffer = ReinterpretSpan<uint8_t>(absl::MakeSpan(src_vals));
  std::vector<int32_t> src_indices = {1, 1};
  Shape dst_shape = {2, 2};
  std::vector<uint8_t> dst_buffer(dst_shape.element_count() * sizeof(int32_t));
  std::vector<int32_t> dst_indices = {0, 0};
  std::vector<int32_t> lengths = {2, 2};
  std::vector<int32_t> expected_dst = {6, 7, 10, 11};

  EXPECT_OK(Copy::Execute<4>(src_buffer, src_shape, src_indices,
                             absl::MakeSpan(dst_buffer), dst_shape, dst_indices,
                             lengths));

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

  EXPECT_OK(Copy::Execute<1>(src_buffer, src_shape, src_indices,
                             absl::MakeSpan(dst_buffer), dst_shape, dst_indices,
                             lengths));
  EXPECT_EQ(dst_buffer, expected_dst);
}

TEST(Copy, HighRank) {
  Shape src_shape = {3, 3, 3, 3};
  auto src_buffer = MakeIota<uint8_t>(81);
  std::vector<int32_t> src_indices = {1, 1, 1, 1};
  Shape dst_shape = {2, 2, 2, 2};
  std::vector<uint8_t> dst_buffer(dst_shape.element_count());
  std::vector<int32_t> dst_indices = {0, 0, 0, 0};
  std::vector<int32_t> lengths = {2, 2, 2, 2};
  std::vector<uint8_t> expected_dst = {41, 42, 44, 45, 50, 51, 53, 54,
                                       68, 69, 71, 72, 77, 78, 80, 81};

  EXPECT_OK(Copy::Execute<1>(src_buffer, src_shape, src_indices,
                             absl::MakeSpan(dst_buffer), dst_shape, dst_indices,
                             lengths));
  EXPECT_EQ(dst_buffer, expected_dst);
}

TEST(Copy, Scalar) {
  Shape src_shape = {};
  std::vector<uint8_t> src_buffer = {42};
  std::vector<int32_t> src_indices = {};
  Shape dst_shape = {};
  std::vector<uint8_t> dst_buffer(dst_shape.element_count());
  std::vector<int32_t> dst_indices = {};
  std::vector<int32_t> lengths = {};
  std::vector<uint8_t> expected_dst = {42};

  EXPECT_OK(Copy::Execute<1>(src_buffer, src_shape, src_indices,
                             absl::MakeSpan(dst_buffer), dst_shape, dst_indices,
                             lengths));
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

  EXPECT_OK(Copy::Execute<4>(src_buffer, src_shape, src_indices,
                             absl::MakeSpan(dst_buffer), dst_shape, dst_indices,
                             lengths));

  absl::Span<int32_t> dst_buffer_int32_t =
      ReinterpretSpan<int32_t>(absl::MakeSpan(dst_buffer));

  EXPECT_EQ(dst_buffer_int32_t, expected_dst);
}

TEST(Pad, NoPadding) {
  Shape src_shape = {2, 3};
  auto src_buffer = MakeIota<uint16_t>(src_shape.element_count());
  std::vector<uint16_t> pad_value_buffer = {0};
  std::vector<int32_t> edge_padding_low = {0, 0};
  std::vector<int32_t> edge_padding_high = {0, 0};
  std::vector<int32_t> interior_padding = {0, 0};
  Shape dst_shape = src_shape;
  std::vector<uint16_t> dst_buffer(dst_shape.element_count(), UINT16_MAX);
  auto expected_dst = src_buffer;

  EXPECT_OK(Pad::Execute<uint16_t>(
      src_buffer, pad_value_buffer, absl::MakeSpan(dst_buffer), src_shape,
      dst_shape, edge_padding_low, edge_padding_high, interior_padding));
  EXPECT_EQ(dst_buffer, expected_dst);
}

TEST(Pad, LowHighPadding) {
  Shape src_shape = {2, 3};
  auto src_buffer = MakeIota<uint16_t>(src_shape.element_count());
  std::vector<uint16_t> pad_value_buffer = {0};
  std::vector<int32_t> edge_padding_low = {0, 1};
  std::vector<int32_t> edge_padding_high = {1, 2};
  std::vector<int32_t> interior_padding = {0, 0};
  Shape dst_shape = {3, 6};
  std::vector<uint16_t> dst_buffer(dst_shape.element_count(), UINT16_MAX);
  // clang-format off
  std::vector<uint16_t> expected_dst = {0, 1, 2, 3, 0, 0,
                                      0, 4, 5, 6, 0, 0,
                                      0, 0, 0, 0, 0, 0};
  // clang-format on

  EXPECT_OK(Pad::Execute<uint16_t>(
      src_buffer, pad_value_buffer, absl::MakeSpan(dst_buffer), src_shape,
      dst_shape, edge_padding_low, edge_padding_high, interior_padding));
  EXPECT_EQ(dst_buffer, expected_dst);
}

TEST(Pad, OnlyHighPadding) {
  Shape src_shape = {2, 3};
  auto src_buffer = MakeIota<uint16_t>(src_shape.element_count());
  std::vector<uint16_t> pad_value_buffer = {0};
  std::vector<int32_t> edge_padding_low = {0, 0};
  std::vector<int32_t> edge_padding_high = {1, 3};
  std::vector<int32_t> interior_padding = {0, 0};
  Shape dst_shape = {3, 6};
  std::vector<uint16_t> dst_buffer(dst_shape.element_count(), UINT16_MAX);
  // clang-format off
  std::vector<uint16_t> expected_dst = {1, 2, 3, 0, 0, 0,
                                      4, 5, 6, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0};
  // clang-format on

  EXPECT_OK(Pad::Execute<uint16_t>(
      src_buffer, pad_value_buffer, absl::MakeSpan(dst_buffer), src_shape,
      dst_shape, edge_padding_low, edge_padding_high, interior_padding));
  EXPECT_EQ(dst_buffer, expected_dst);
}

TEST(Pad, OnlyLowPadding) {
  Shape src_shape = {2, 3};
  auto src_buffer = MakeIota<uint16_t>(src_shape.element_count());
  std::vector<uint16_t> pad_value_buffer = {0};
  std::vector<int32_t> edge_padding_low = {1, 3};
  std::vector<int32_t> edge_padding_high = {0, 0};
  std::vector<int32_t> interior_padding = {0, 0};
  Shape dst_shape = {3, 6};
  std::vector<uint16_t> dst_buffer(dst_shape.element_count(), UINT16_MAX);
  // clang-format off
  std::vector<uint16_t> expected_dst = {0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 1, 2, 3,
                                      0, 0, 0, 4, 5, 6};
  // clang-format on

  EXPECT_OK(Pad::Execute<uint16_t>(
      src_buffer, pad_value_buffer, absl::MakeSpan(dst_buffer), src_shape,
      dst_shape, edge_padding_low, edge_padding_high, interior_padding));
  EXPECT_EQ(dst_buffer, expected_dst);
}

TEST(Pad, OnlyInteriorPadding) {
  Shape src_shape = {2, 3};
  auto src_buffer = MakeIota<uint16_t>(src_shape.element_count());
  std::vector<uint16_t> pad_value_buffer = {0};
  std::vector<int32_t> edge_padding_low = {0, 0};
  std::vector<int32_t> edge_padding_high = {0, 0};
  std::vector<int32_t> interior_padding = {1, 1};
  Shape dst_shape = {3, 5};
  std::vector<uint16_t> dst_buffer(dst_shape.element_count(), UINT16_MAX);
  // clang-format off
  std::vector<uint16_t> expected_dst = {1, 0, 2, 0, 3,
                                      0, 0, 0, 0, 0,
                                      4, 0, 5, 0, 6};
  // clang-format on

  EXPECT_OK(Pad::Execute<uint16_t>(
      src_buffer, pad_value_buffer, absl::MakeSpan(dst_buffer), src_shape,
      dst_shape, edge_padding_low, edge_padding_high, interior_padding));
  EXPECT_EQ(dst_buffer, expected_dst);
}

TEST(Pad, AllPaddingTypes) {
  Shape src_shape = {2, 3};
  auto src_buffer = MakeIota<uint16_t>(src_shape.element_count());
  std::vector<uint16_t> pad_value_buffer = {0};
  std::vector<int32_t> edge_padding_low = {1, 1};
  std::vector<int32_t> edge_padding_high = {1, 2};
  std::vector<int32_t> interior_padding = {1, 1};
  Shape dst_shape = {5, 8};
  std::vector<uint16_t> dst_buffer(dst_shape.element_count(), UINT16_MAX);
  // clang-format off
  std::vector<uint16_t> expected_dst = {0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 1, 0, 2, 0, 3, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 4, 0, 5, 0, 6, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0};
  // clang-format on

  EXPECT_OK(Pad::Execute<uint16_t>(
      src_buffer, pad_value_buffer, absl::MakeSpan(dst_buffer), src_shape,
      dst_shape, edge_padding_low, edge_padding_high, interior_padding));
  EXPECT_EQ(dst_buffer, expected_dst);
}

TEST(Pad, HighRank) {
  Shape src_shape = {2, 2, 2, 2};
  auto src_buffer = MakeIota<uint16_t>(src_shape.element_count());
  std::vector<uint16_t> pad_value_buffer = {0};
  std::vector<int32_t> edge_padding_low = {1, 0, 0, 0};
  std::vector<int32_t> edge_padding_high = {0, 1, 0, 0};
  std::vector<int32_t> interior_padding = {0, 0, 1, 0};
  Shape dst_shape = {3, 3, 3, 2};
  std::vector<uint16_t> dst_buffer(dst_shape.element_count(), UINT16_MAX);
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

  EXPECT_OK(Pad::Execute<uint16_t>(
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
  std::vector<float> dst_buffer(dst_shape.element_count(), 0.0f);
  std::vector<float> expected_dst = {5.0f};

  EXPECT_OK(ReduceSum::Execute<float>(src_buffer, init_buffer,
                                      absl::MakeSpan(dst_buffer), dimension,
                                      src_shape, dst_shape));

  for (int i = 0; i < dst_buffer.size(); ++i) {
    EXPECT_NEAR(expected_dst[i], dst_buffer[i], kEpsilon);
  }
}

TEST(ReduceMin, TwoDimensionsToOne) {
  Shape src_shape = {3, 3};
  int32_t dimension = 0;
  Shape dst_shape = {3};
  std::vector<float> src_buffer = MakeIota<float>(src_shape.element_count());
  std::vector<float> init_buffer = {std::numeric_limits<float>::max()};
  std::vector<float> dst_buffer(dst_shape.element_count(), 0.0f);
  std::vector<float> expected_dst = {1.0f, 2.0f, 3.0f};

  EXPECT_OK(ReduceMin::Execute<float>(src_buffer, init_buffer,
                                      absl::MakeSpan(dst_buffer), dimension,
                                      src_shape, dst_shape));

  for (int i = 0; i < dst_buffer.size(); ++i) {
    EXPECT_NEAR(expected_dst[i], dst_buffer[i], kEpsilon);
  }
}

}  // namespace
}  // namespace kernels
}  // namespace vmla
}  // namespace hal
}  // namespace iree
