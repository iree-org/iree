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

#ifndef IREE_IREE_BASE_SHAPED_BUFFER_H_
#define IREE_IREE_BASE_SHAPED_BUFFER_H_

#include <stdint.h>

#include <vector>

#include "absl/types/span.h"
#include "iree/base/logging.h"
#include "iree/base/memory.h"
#include "iree/base/shape.h"

namespace iree {

// A struct representing a buffer of bytes that should be interpreted as being
// of the specified shape with elements of the specified size.
class ShapedBuffer {
 public:
  ShapedBuffer() = default;
  // move only
  ShapedBuffer(ShapedBuffer&& other) = default;
  ShapedBuffer& operator=(ShapedBuffer&& other) = default;
  ShapedBuffer(int8_t element_size, Shape shape, std::vector<uint8_t> contents)
      : element_size_(element_size),
        shape_(shape),
        contents_(std::move(contents)) {}

  template <typename T>
  static ShapedBuffer Create(Shape shape, absl::Span<const T> contents) {
    CHECK_EQ(contents.size(), shape.element_count());
    auto byte_span = ReinterpretSpan<uint8_t>(contents);
    return ShapedBuffer(
        sizeof(T), shape,
        std::vector<uint8_t>(byte_span.begin(), byte_span.end()));
  }

  static inline bool Equal(const ShapedBuffer& a, const ShapedBuffer& b) {
    return a.element_size_ == b.element_size_ && a.shape_ == b.shape_ &&
           a.contents_ == b.contents_;
  }

  int8_t element_size() const { return element_size_; }
  Shape shape() const { return shape_; }
  absl::Span<const uint8_t> contents() const { return contents_; }

 private:
  // Size of the buffer elements, in bytes.
  int8_t element_size_;
  Shape shape_;
  std::vector<uint8_t> contents_;
};

inline bool operator==(const ShapedBuffer& a, const ShapedBuffer& b) {
  return ShapedBuffer::Equal(a, b);
}

inline bool operator!=(const ShapedBuffer& a, const ShapedBuffer& b) {
  return !(a == b);
}

}  // namespace iree

#endif  // IREE_IREE_BASE_SHAPED_BUFFER_H_
