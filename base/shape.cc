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

#include "base/shape.h"

#include <cstddef>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "base/source_location.h"
#include "base/status.h"

namespace iree {

Shape::Shape(const int* values, int size) : rank_(size) {
  QCHECK_LE(size, kMaxRank)
      << "Max rank of " << kMaxRank << ", shape has " << size;
  std::memcpy(value_, values, size * sizeof(int));
}

std::string Shape::DebugString() const {
  return absl::StrCat("[", absl::StrJoin(subspan(), ","), "]");
}

absl::Span<const int> Shape::subspan(size_type pos, size_type len) const {
  if (len == npos) {
    len = rank_ - pos;
  }
  return absl::MakeConstSpan(&value_[pos], len);
}

void Shape::push_back(int dim) {
  DCHECK_LE(rank_ + 1, kMaxRank);
  value_[rank_++] = dim;
}

void Shape::insert(iterator pos, int dim) {
  int axis = static_cast<int>(pos - value_);
  DCHECK_GE(axis, 0);
  DCHECK_LE(axis, rank_);
  DCHECK_LE(rank_ + 1, kMaxRank);
  ++rank_;
  for (int i = rank_ - 1; i > axis; --i) {
    value_[i] = value_[i - 1];
  }
  value_[axis] = dim;
}

void Shape::erase(iterator pos) {
  int axis = static_cast<int>(pos - value_);
  DCHECK_GE(axis, 0);
  DCHECK_LE(axis, rank_);
  for (int i = axis; i < rank_ - 1; ++i) {
    value_[i] = value_[i + 1];
  }
  --rank_;
}

int Shape::element_count() const {
  size_t element_count = 1;
  for (int i = 0; i < rank_; ++i) {
    int dim = value_[i];
    if (dim == -1) {
      return 0;
    }
    element_count *= dim;
  }
  return element_count;
}

StatusOr<int> Shape::ResolveAxis(int axis) const {
  if (rank_ == 0 && (axis == -1 || axis == 0)) {
    // Scalar axes resolves to 0.
    return 0;
  }

  int new_axis = axis;
  if (new_axis < 0) {
    new_axis += rank_;
  }
  if (new_axis < 0 || new_axis >= rank_) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Axis " << new_axis << " (orig " << axis
           << ") out of bounds of rank " << rank_;
  }
  return new_axis;
}

}  // namespace iree
