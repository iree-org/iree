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

#include "iree/hal/interpreter/bytecode_dispatch_util.h"

namespace iree {
namespace hal {

bool BufferViewIsTrue(const BufferView& buffer_view) {
  if (buffer_view.element_size == 0 || !buffer_view.buffer ||
      buffer_view.byte_length() == 0) {
    return false;
  }
  // TODO(benvanik): map more efficiently (based on element size?).
  auto mapping =
      buffer_view.buffer->MapMemory<uint8_t>(hal::MemoryAccess::kRead);
  if (!mapping.ok()) {
    return false;
  }
  for (uint8_t value : mapping.ValueOrDie().contents()) {
    if (value) return true;
  }
  return false;
}

Status ValidateElementwiseUnaryOp(BufferView* src_local,
                                  BufferView* dst_local) {
  // TODO(benvanik): validate shapes.
  return OkStatus();
}

Status ValidateElementwiseBinaryOp(BufferView* lhs_local, BufferView* rhs_local,
                                   BufferView* dst_local) {
  // TODO(benvanik): validate shapes.
  return OkStatus();
}

Status ValidateElementwiseTernaryOp(BufferView* a_local, BufferView* b_local,
                                    BufferView* c_local,
                                    BufferView* dst_local) {
  // TODO(benvanik): validate shapes.
  return OkStatus();
}

Status ValidateMatMulOpI(BufferView* lhs_local, BufferView* rhs_local,
                         BufferView* bias_local,
                         BufferView* multiplier_mantissa_local,
                         BufferView* multiplier_exponent_local,
                         BufferView* dst_local) {
  // TODO(benvanik): validate shapes.
  return OkStatus();
}

Status ValidateMatMulOpF(BufferView* lhs_local, BufferView* rhs_local,
                         BufferView* bias_local, BufferView* dst_local) {
  // TODO(benvanik): validate shapes.
  return OkStatus();
}

Status ApplyCopy(BufferView* src_local, absl::Span<const int32_t> src_indices,
                 BufferView* dst_local, absl::Span<const int32_t> dst_indices,
                 absl::Span<const int32_t> lengths) {
  ASSIGN_OR_RETURN(auto src_buffer,
                   src_local->buffer->MapMemory<uint8_t>(MemoryAccess::kRead));
  // TODO(benvanik): discard if overwriting the entire buffer.
  ASSIGN_OR_RETURN(auto dst_buffer,
                   dst_local->buffer->MapMemory<uint8_t>(MemoryAccess::kWrite));
  switch (src_local->element_size) {
    case 1:
      return kernels::Copy::Execute<1>(src_buffer.contents(), src_local->shape,
                                       src_indices,
                                       dst_buffer.mutable_contents(),
                                       dst_local->shape, dst_indices, lengths);
    case 2:
      return kernels::Copy::Execute<2>(src_buffer.contents(), src_local->shape,
                                       src_indices,
                                       dst_buffer.mutable_contents(),
                                       dst_local->shape, dst_indices, lengths);
    case 4:
      return kernels::Copy::Execute<4>(src_buffer.contents(), src_local->shape,
                                       src_indices,
                                       dst_buffer.mutable_contents(),
                                       dst_local->shape, dst_indices, lengths);
    case 8:
      return kernels::Copy::Execute<8>(src_buffer.contents(), src_local->shape,
                                       src_indices,
                                       dst_buffer.mutable_contents(),
                                       dst_local->shape, dst_indices, lengths);
    default:
      return UnimplementedErrorBuilder(IREE_LOC)
             << "Unimplemented element size: " << src_local->element_size;
  }
}

}  // namespace hal
}  // namespace iree
