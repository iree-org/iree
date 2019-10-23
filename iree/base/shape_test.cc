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

#include "iree/base/shape.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "iree/base/status.h"
#include "iree/base/status_matchers.h"

namespace iree {
namespace {

using ::testing::ElementsAre;

// Tests shapes that represent 0-D scalar values.
TEST(ShapeTest, Scalar) {
  Shape shape;
  EXPECT_EQ(0, shape.size());
  EXPECT_TRUE(shape.empty());
  EXPECT_EQ(1, shape.element_count());
  EXPECT_EQ(shape, shape);
  EXPECT_EQ(0, shape.subspan().size());
  for (const int dim : shape) {
    FAIL() << "Should have no dimensions, have: " << dim;
  }
  EXPECT_EQ(shape.begin(), shape.end());
  EXPECT_EQ(shape.cbegin(), shape.cend());
  shape.clear();
  EXPECT_EQ(0, shape.size());
}

// Tests the various ways of constructing a 1+D shape.
TEST(ShapeTest, NonScalarConstruction) {
  EXPECT_EQ(0, Shape().size());
  EXPECT_EQ(0, Shape({}).size());
  EXPECT_EQ(1, Shape({10}).size());
  EXPECT_EQ(4, Shape({10, 20, 30, 40}).size());

  std::vector<int> empty_data = {};
  EXPECT_EQ(0, Shape(empty_data.data(), empty_data.size()).size());
  EXPECT_EQ(0, Shape(empty_data.begin(), empty_data.end()).size());
  EXPECT_EQ(0, Shape(absl::MakeConstSpan(empty_data)).size());

  EXPECT_THAT(Shape({}).subspan(), ElementsAre());
  EXPECT_THAT(Shape({10}).subspan(), ElementsAre(10));
  EXPECT_THAT(Shape({10, 20, 30, 40}).subspan(), ElementsAre(10, 20, 30, 40));

  std::vector<int> valid_data = {10, 20, 30, 40};
  EXPECT_THAT(Shape(valid_data.begin(), valid_data.end()).subspan(),
              ElementsAre(10, 20, 30, 40));
  EXPECT_THAT(Shape(absl::MakeConstSpan(valid_data)).subspan(),
              ElementsAre(10, 20, 30, 40));
}

// Tests shapes that represent 1+D multidimensional values.
TEST(ShapeTest, NonScalarAccess) {
  Shape shape = {1, 2, 3, 4};
  EXPECT_EQ(4, shape.size());
  EXPECT_FALSE(shape.empty());
  EXPECT_EQ(1 * 2 * 3 * 4, shape.element_count());
  EXPECT_EQ(shape, shape);
  EXPECT_NE(shape, Shape({4, 3, 2, 1}));
  EXPECT_THAT(shape.subspan(), ElementsAre(1, 2, 3, 4));
  std::vector<int> readout;
  for (const int dim : shape) {
    readout.push_back(dim);
  }
  EXPECT_THAT(readout, ElementsAre(1, 2, 3, 4));
  EXPECT_EQ(1, shape[0]);
  EXPECT_EQ(2, shape[1]);
  EXPECT_EQ(3, shape[2]);
  EXPECT_EQ(4, shape[3]);
  EXPECT_EQ(1, shape.front());
  EXPECT_EQ(4, shape.back());
}

TEST(ShapeTest, PushBack) {
  Shape shape;
  EXPECT_EQ(0, shape.size());

  shape.push_back(10);
  EXPECT_EQ(1, shape.size());
  EXPECT_EQ(10, shape.front());
  EXPECT_EQ(10, shape.back());
  EXPECT_EQ(10, shape[0]);
  EXPECT_THAT(shape.subspan(), ElementsAre(10));

  shape.push_back(20);
  EXPECT_EQ(2, shape.size());
  EXPECT_EQ(10, shape.front());
  EXPECT_EQ(20, shape.back());
  EXPECT_EQ(10, shape[0]);
  EXPECT_EQ(20, shape[1]);
  EXPECT_THAT(shape.subspan(), ElementsAre(10, 20));
}

TEST(ShapeTest, Insert) {
  Shape shape;
  EXPECT_EQ(0, shape.size());

  shape.insert(shape.begin(), 20);
  EXPECT_THAT(shape.subspan(), ElementsAre(20));
  shape.insert(shape.begin(), 10);
  EXPECT_THAT(shape.subspan(), ElementsAre(10, 20));
  shape.insert(shape.end(), 40);
  EXPECT_THAT(shape.subspan(), ElementsAre(10, 20, 40));
  shape.insert(shape.begin() + 2, 30);
  EXPECT_THAT(shape.subspan(), ElementsAre(10, 20, 30, 40));

  Shape ex_shape{72, 4};
  ex_shape.insert(ex_shape.begin(), 144);
  EXPECT_THAT(ex_shape.subspan(), ElementsAre(144, 72, 4));
}

TEST(ShapeTest, Erase) {
  Shape shape = {1, 2, 3, 4};
  EXPECT_THAT(shape.subspan(), ElementsAre(1, 2, 3, 4));
  shape.erase(shape.begin());
  EXPECT_THAT(shape.subspan(), ElementsAre(2, 3, 4));
  shape.erase(shape.end());
  EXPECT_THAT(shape.subspan(), ElementsAre(2, 3));
  shape.erase(shape.begin() + 1);
  EXPECT_THAT(shape.subspan(), ElementsAre(2));
  shape.erase(shape.end());
  EXPECT_THAT(shape.subspan(), ElementsAre());
}

TEST(ShapeTest, Clear) {
  Shape shape;
  EXPECT_EQ(0, shape.size());
  shape.clear();
  EXPECT_EQ(0, shape.size());

  shape = Shape({1});
  shape.clear();
  EXPECT_EQ(0, shape.size());

  shape = Shape({1, 2, 3, 4});
  shape.clear();
  EXPECT_EQ(0, shape.size());
}

TEST(ShapeTest, DebugString) {
  EXPECT_EQ("[]", Shape({}).DebugString());
  EXPECT_EQ("[1]", Shape({1}).DebugString());
  EXPECT_EQ("[1,2]", Shape({1, 2}).DebugString());
}

TEST(ShapeTest, ElementCount) {
  EXPECT_EQ(1, Shape({}).element_count());
  EXPECT_EQ(0, Shape({0}).element_count());
  EXPECT_EQ(1, Shape({1}).element_count());
  EXPECT_EQ(2, Shape({2, 1}).element_count());
  EXPECT_EQ(10, Shape({2, 5}).element_count());
  EXPECT_EQ(9216, Shape({72, 1, 128}).element_count());
  EXPECT_EQ(9216, Shape({1, 72, 128}).element_count());

  // Partial shaping should yield no elements.
  EXPECT_EQ(0, Shape({1, -1, 2, 3}).element_count());
}

TEST(ShapeTest, ResolveAxis) {
  int axis;
  ASSERT_OK_AND_ASSIGN(axis, Shape({0}).ResolveAxis(0));
  EXPECT_EQ(0, axis);
  ASSERT_OK_AND_ASSIGN(axis, Shape({0, 1, 2}).ResolveAxis(1));
  EXPECT_EQ(1, axis);
  ASSERT_OK_AND_ASSIGN(axis, Shape({0, 1, 2}).ResolveAxis(2));
  EXPECT_EQ(2, axis);

  EXPECT_TRUE(IsInvalidArgument(Shape({0, 1, 2}).ResolveAxis(3).status()));
}

TEST(ShapeTest, ResolveAxisNegative) {
  int axis;
  ASSERT_OK_AND_ASSIGN(axis, Shape({0, 1, 2}).ResolveAxis(-3));
  EXPECT_EQ(0, axis);
  ASSERT_OK_AND_ASSIGN(axis, Shape({0, 1, 2}).ResolveAxis(-2));
  EXPECT_EQ(1, axis);
  ASSERT_OK_AND_ASSIGN(axis, Shape({0, 1, 2}).ResolveAxis(-1));
  EXPECT_EQ(2, axis);

  EXPECT_TRUE(IsInvalidArgument(Shape({0, 1, 2}).ResolveAxis(-4).status()));
}

TEST(ShapeTest, ResolveAxisScalar) {
  int axis;
  ASSERT_OK_AND_ASSIGN(axis, Shape({}).ResolveAxis(0));
  EXPECT_EQ(0, axis);
  ASSERT_OK_AND_ASSIGN(axis, Shape({}).ResolveAxis(-1));
  EXPECT_EQ(0, axis);

  EXPECT_TRUE(IsInvalidArgument(Shape({}).ResolveAxis(1).status()));
}

TEST(ShapeTest, Equality) {
  EXPECT_EQ(Shape({}), Shape({}));
  EXPECT_EQ(Shape({0}), Shape({0}));
  EXPECT_EQ(Shape({1}), Shape({1}));
  EXPECT_EQ(Shape({1, 2}), Shape({1, 2}));

  EXPECT_NE(Shape({}), Shape({1}));
  EXPECT_NE(Shape({-1}), Shape({1}));
  EXPECT_NE(Shape({1}), Shape({}));
  EXPECT_NE(Shape({1}), Shape({2}));
  EXPECT_NE(Shape({1, 2}), Shape({3, 4}));
}

}  // namespace
}  // namespace iree
