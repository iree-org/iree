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

#ifndef IREE_HAL_BUFFER_VIEW_H_
#define IREE_HAL_BUFFER_VIEW_H_

#include <memory>
#include <ostream>

#include "base/shape.h"
#include "hal/buffer.h"

namespace iree {
namespace hal {

struct BufferView {
  // Returns true if the given buffer_views are exactly equal.
  static bool Equal(const BufferView& lhs, const BufferView& rhs);

  BufferView() = default;
  BufferView(ref_ptr<Buffer> buffer, Shape shape, int8_t element_size) noexcept
      : buffer(std::move(buffer)), shape(shape), element_size(element_size) {}

  BufferView(const BufferView& other) noexcept
      : buffer(add_ref(other.buffer)),
        shape(other.shape),
        element_size(other.element_size) {}
  BufferView& operator=(const BufferView& other) noexcept {
    buffer = add_ref(other.buffer);
    shape = other.shape;
    element_size = other.element_size;
    return *this;
  }
  BufferView(BufferView&& other) noexcept
      : buffer(std::move(other.buffer)),
        shape(other.shape),
        element_size(other.element_size) {}
  BufferView& operator=(BufferView&& other) noexcept {
    buffer = std::move(other.buffer);
    shape = other.shape;
    element_size = other.element_size;
    return *this;
  }

  // Returns a string useful for printing debug messages.
  std::string DebugStringShort() const;

  // Total length of the valid view range in bytes.
  device_size_t byte_length() const {
    return shape.element_count() * element_size;
  }

  // TODO(b/134586626): remove this when byte ranges are encoded in IR.
  // Calculates a byte offset into the buffer_view at the given dimension
  // indices.
  StatusOr<device_size_t> CalculateOffset(
      absl::Span<const int32_t> indices) const;

  // TODO(b/134586626): remove this when byte ranges are encoded in IR.
  // Returns a view onto the given range of the buffer underlying this view. The
  // returned view starts at the offset indicated by |start_indices| and has a
  // shape of |lengths|.
  // Only contiguous regions of memory are supported at the moment.
  StatusOr<BufferView> Slice(absl::Span<const int32_t> start_indices,
                             absl::Span<const int32_t> lengths) const;

  // TODO(b/134586626): remove this when byte ranges are encoded in IR.
  static Status Copy(BufferView* src,
                     absl::Span<const int32_t> src_start_indices,
                     BufferView* dst,
                     absl::Span<const int32_t> dst_start_indices,
                     absl::Span<const int32_t> lengths);

  ref_ptr<Buffer> buffer;
  Shape shape;
  int8_t element_size;
  // TODO(benvanik): strides.
};

inline bool operator==(const BufferView& a, const BufferView& b) {
  return BufferView::Equal(a, b);
}

inline bool operator!=(const BufferView& a, const BufferView& b) {
  return !(a == b);
}

inline std::ostream& operator<<(std::ostream& stream,
                                const BufferView& buffer_view) {
  stream << buffer_view.DebugStringShort();
  return stream;
}

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_BUFFER_VIEW_H_
