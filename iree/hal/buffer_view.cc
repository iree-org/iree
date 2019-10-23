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

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "iree/base/source_location.h"
#include "iree/base/status.h"
#include "iree/hal/buffer.h"

namespace iree {
namespace hal {

namespace {
// Pretty prints an array, e.g. [1, 2, 3, 4]
inline std::string PrettyPrint(absl::Span<const int32_t> arr) {
  return "[" + absl::StrJoin(arr, ",") + "]";
}
}  // namespace

// static
bool BufferView::Equal(const BufferView& lhs, const BufferView& rhs) {
  return lhs.buffer.get() == rhs.buffer.get() &&
         lhs.element_size == rhs.element_size && lhs.shape == rhs.shape;
}

std::string BufferView::DebugStringShort() const {
  if (element_size == 0) {
    return "Ø";
  }
  return shape.empty() ? std::to_string(element_size)
                       : absl::StrCat(absl::StrJoin(shape.subspan(), "x"), "x",
                                      element_size);
}

StatusOr<device_size_t> BufferView::CalculateOffset(
    absl::Span<const int32_t> indices) const {
  if (indices.empty()) {
    return 0;
  } else if (shape.empty() || indices.size() > shape.size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Indices " << PrettyPrint(indices)
           << " out of bounds of the rank of buffer_view "
           << DebugStringShort();
  }
  device_size_t offset = 0;
  for (int i = 0; i < indices.size(); ++i) {
    if (indices[i] >= shape[i]) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Indices[" << i << "]=" << indices[i]
             << " out of bounds of buffer_view " << DebugStringShort();
    }
    device_size_t axis_offset = indices[i];
    for (int j = i + 1; j < shape.size(); ++j) {
      axis_offset *= shape[j];
    }
    offset += axis_offset;
  }
  offset *= element_size;
  return offset;
}

StatusOr<BufferView> BufferView::Slice(
    absl::Span<const int32_t> start_indices,
    absl::Span<const int32_t> lengths) const {
  if (start_indices.size() != shape.size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Slice start_indices " << PrettyPrint(start_indices)
           << " do not match rank of buffer_view " << DebugStringShort();
  }
  if (start_indices.size() != lengths.size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Slice start_indices " << PrettyPrint(start_indices)
           << " and lengths " << PrettyPrint(lengths)
           << " are not the same size";
  }

  // Buffer::Subspan only support contiguous memory. To ensure that this slice
  // only requests such, we validate that the offset in the buffer between the
  // start and end indices is the same as the requested size of the slice.
  absl::InlinedVector<int32_t, 6> end_indices(lengths.size());
  device_size_t subspan_length = element_size;
  for (int i = 0; i < lengths.size(); ++i) {
    subspan_length *= lengths[i];
    end_indices[i] = start_indices[i] + lengths[i] - 1;
  }

  ASSIGN_OR_RETURN(auto start_byte_offset, CalculateOffset(start_indices));
  // Also validates the ends are in bounds.
  ASSIGN_OR_RETURN(auto end_byte_offset, CalculateOffset(end_indices));

  auto offset_length = end_byte_offset - start_byte_offset + element_size;
  if (subspan_length != offset_length) {
    return UnimplementedErrorBuilder(IREE_LOC)
           << "Slice for non-contiguous region of memory unimplemented. "
              "start_indices: "
           << PrettyPrint(start_indices) << " lengths: " << PrettyPrint(lengths)
           << " " << subspan_length << " " << offset_length << " "
           << PrettyPrint(end_indices);
  }

  ASSIGN_OR_RETURN(auto new_buffer,
                   Buffer::Subspan(buffer, start_byte_offset, subspan_length));
  return BufferView(std::move(new_buffer), Shape(lengths), element_size);
}

// static
Status BufferView::Copy(BufferView* src,
                        absl::Span<const int32_t> src_start_indices,
                        BufferView* dst,
                        absl::Span<const int32_t> dst_start_indices,
                        absl::Span<const int32_t> lengths) {
  if (src_start_indices.size() != src->shape.size() ||
      dst_start_indices.size() != dst->shape.size() ||
      src_start_indices.size() != lengths.size() ||
      dst_start_indices.size() != lengths.size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Src/dst shape/size mismatch: src=" << src->DebugStringShort()
           << ", dst=" << dst->DebugStringShort()
           << ", src_indices=" << PrettyPrint(src_start_indices)
           << ", dst_indices=" << PrettyPrint(dst_start_indices)
           << ", lengths=" << PrettyPrint(lengths);
  }

  // Copies only support contiguous memory. To ensure that this copy
  // only requests such, we validate that the offset in the buffer between the
  // start and end indices is the same as the requested size of the copy.
  absl::InlinedVector<int32_t, 4> src_end_indices(lengths.size());
  absl::InlinedVector<int32_t, 4> dst_end_indices(lengths.size());
  device_size_t total_length = src->element_size;
  for (int i = 0; i < lengths.size(); ++i) {
    total_length *= lengths[i];
    src_end_indices[i] = src_start_indices[i] + lengths[i] - 1;
    dst_end_indices[i] = dst_start_indices[i] + lengths[i] - 1;
  }

  ASSIGN_OR_RETURN(auto src_start_byte_offset,
                   src->CalculateOffset(src_start_indices));
  ASSIGN_OR_RETURN(auto src_end_byte_offset,
                   src->CalculateOffset(src_end_indices));
  ASSIGN_OR_RETURN(auto dst_start_byte_offset,
                   dst->CalculateOffset(dst_start_indices));
  ASSIGN_OR_RETURN(auto dst_end_byte_offset,
                   dst->CalculateOffset(dst_end_indices));

  auto src_length =
      src_end_byte_offset - src_start_byte_offset + src->element_size;
  auto dst_length =
      dst_end_byte_offset - dst_start_byte_offset + dst->element_size;
  if (src_length != dst_length || src_length != total_length) {
    return UnimplementedErrorBuilder(IREE_LOC)
           << "Copy for non-contiguous region of memory unimplemented: "
           << src->DebugStringShort() << ", dst=" << dst->DebugStringShort()
           << ", src_indices=" << PrettyPrint(src_start_indices)
           << ", dst_indices=" << PrettyPrint(dst_start_indices)
           << ", lengths=" << PrettyPrint(lengths);
  }

  RETURN_IF_ERROR(dst->buffer->CopyData(dst_start_byte_offset,
                                        src->buffer.get(),
                                        src_start_byte_offset, total_length));

  return OkStatus();
}

}  // namespace hal
}  // namespace iree
