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

#ifndef IREE_BASE_SHAPE_H_
#define IREE_BASE_SHAPE_H_

#include <array>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/meta/type_traits.h"
#include "absl/types/span.h"
#include "iree/base/logging.h"
#include "iree/base/status.h"

namespace iree {

// For simplicity we limit our shapes to a max of rank-N (shape.size() == N) as
// this prevents dynamic allocations and rarely are there greater ranks.
constexpr int kMaxRank = 5;

// Represent indices and lengths of tensors.
using Index = std::array<int, kMaxRank>;
using Length = std::array<int, kMaxRank>;

// Represents the number of elements in multiple dimensions.
// Can be rank-0 (scalar) to rank-kMaxRank. Tries to match the API of
// std::vector and can be converted to a Span via subspan().
//
// https://www.tensorflow.org/guide/tensors#shape
class Shape {
 public:
  using size_type = int;
  static constexpr size_type npos = ~(size_type(0));  // NOLINT
  using iterator = int*;
  using const_iterator = const int*;

  Shape() = default;
  Shape(const int* values, int size);
  Shape(std::initializer_list<int> values)
      : Shape(values.begin(), values.size()) {}
  explicit Shape(absl::Span<const int> values)
      : Shape(values.data(), values.size()) {}

  template <typename Iterator>
  using EnableIfForwardIterator = absl::enable_if_t<std::is_convertible<
      typename std::iterator_traits<Iterator>::iterator_category,
      std::forward_iterator_tag>::value>;
  template <typename Iterator, EnableIfForwardIterator<Iterator>* = nullptr>
  Shape(Iterator first, Iterator last) {
    rank_ = std::distance(first, last);
    QCHECK_LE(rank_, kMaxRank);
    for (int i = 0; first != last; ++i, static_cast<void>(++first)) {
      value_[i] = *first;
    }
  }

  // Returns a string representation of the given shape.
  std::string DebugString() const;

  // Size (aka 'rank') of the shape, counting the number of dimensions.
  constexpr size_type size() const noexcept { return rank_; }

  // Whether the shape is rank-0 (scalar).
  constexpr bool empty() const noexcept { return rank_ == 0; }

  // Returns the total elements in the tensor shape.
  // Returns 0 if the tensor shape is not complete and 1 if the shape is a
  // scalar value.
  int element_count() const;

  // Resolves an axis in [-R,R) to the real axis value and verifies the range.
  StatusOr<int> ResolveAxis(int axis) const;

  // Compares two shapes for equality.
  inline static bool Equal(const Shape& a, const Shape& b) {
    return a.rank_ == b.rank_ &&
           std::memcmp(a.value_, b.value_, a.rank_ * sizeof(value_[0])) == 0;
  }

  int& operator[](size_type i) noexcept {
    DCHECK_GE(i, 0);
    DCHECK_LT(i, rank_);
    return value_[i];
  }

  const int& operator[](size_type i) const noexcept {
    DCHECK_GE(i, 0);
    DCHECK_LT(i, rank_);
    return value_[i];
  }

  int front() const noexcept {
    DCHECK_GE(rank_, 1);
    return value_[0];
  }

  int back() const noexcept {
    DCHECK_GE(rank_, 1);
    return value_[rank_ - 1];
  }

  constexpr iterator begin() const noexcept {
    return const_cast<iterator>(&value_[0]);
  }
  constexpr iterator end() const noexcept {
    return const_cast<iterator>(&value_[rank_]);
  }
  constexpr const_iterator cbegin() const noexcept { return &value_[0]; }
  constexpr const_iterator cend() const noexcept { return &value_[rank_]; }

  absl::Span<const int> subspan(size_type pos = 0, size_type len = npos) const;
  absl::Span<const int> data() const { return subspan(); }

  void push_back(int dim);

  void insert(iterator pos, int dim);

  void erase(iterator pos);

  void clear() { rank_ = 0; }

 private:
  size_type rank_ = 0;
  int value_[kMaxRank];
};

inline bool operator==(const Shape& a, const Shape& b) {
  return Shape::Equal(a, b);
}

inline bool operator!=(const Shape& a, const Shape& b) { return !(a == b); }

inline std::ostream& operator<<(std::ostream& stream, const Shape& shape) {
  stream << shape.DebugString();
  return stream;
}

}  // namespace iree

#endif  // IREE_BASE_SHAPE_H_
