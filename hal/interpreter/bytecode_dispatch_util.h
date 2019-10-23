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

// Utilities used by the bytecode_dispatch routines to aid in working with the
// bytecode stream and kernel dispatch.

#ifndef IREE_HAL_INTERPRETER_BYTECODE_DISPATCH_UTIL_H_
#define IREE_HAL_INTERPRETER_BYTECODE_DISPATCH_UTIL_H_

#include "absl/base/attributes.h"
#include "absl/container/inlined_vector.h"
#include "base/status.h"
#include "hal/buffer_view.h"
#include "hal/heap_buffer.h"
#include "hal/interpreter/bytecode_kernels.h"
#include "rt/function.h"
#include "rt/stack.h"
#include "schemas/bytecode/interpreter_bytecode_v0.h"
#include "vm/bytecode_reader.h"
#include "vm/type.h"

// TODO(benvanik): move to dedicated config file/build flags.
#define IREE_SUPPORT_F32 1
#define IREE_SUPPORT_F64 1

namespace iree {
namespace hal {

// Returns true if the contents of the BufferView are bitwise non-zero.
// Returns false if there is no buffer, the buffer is empty, or the contents are
// bitwise zero.
bool BufferViewIsTrue(const BufferView& buffer_view);

Status ValidateElementwiseUnaryOp(BufferView* src_local, BufferView* dst_local);
Status ValidateElementwiseBinaryOp(BufferView* lhs_local, BufferView* rhs_local,
                                   BufferView* dst_local);
Status ValidateElementwiseTernaryOp(BufferView* a_local, BufferView* b_local,
                                    BufferView* c_local, BufferView* dst_local);
Status ValidateMatMulOpI(BufferView* lhs_local, BufferView* rhs_local,
                         BufferView* bias_local,
                         BufferView* multiplier_mantissa_local,
                         BufferView* multiplier_exponent_local,
                         BufferView* dst_local);
Status ValidateMatMulOpF(BufferView* lhs_local, BufferView* rhs_local,
                         BufferView* bias_local, BufferView* dst_local);

template <typename KERNEL, typename T, typename... ARGS>
Status ApplyUnaryOp(BufferView* src_local, BufferView* dst_local,
                    ARGS... args) {
  // TODO(benvanik): avoid mapping by changing buffer type?
  ASSIGN_OR_RETURN(auto src_buffer,
                   src_local->buffer->MapMemory<T>(MemoryAccess::kRead));
  ASSIGN_OR_RETURN(auto dst_buffer, dst_local->buffer->MapMemory<T>(
                                        MemoryAccess::kDiscardWrite));
  return KERNEL::Execute(src_buffer.contents(), dst_buffer.mutable_contents(),
                         args...);
}

template <typename KERNEL, typename T, typename... ARGS>
Status ApplyBinaryOp(BufferView* lhs_local, BufferView* rhs_local,
                     BufferView* dst_local, ARGS... args) {
  ASSIGN_OR_RETURN(auto lhs_buffer,
                   lhs_local->buffer->MapMemory<T>(MemoryAccess::kRead));
  ASSIGN_OR_RETURN(auto rhs_buffer,
                   rhs_local->buffer->MapMemory<T>(MemoryAccess::kRead));
  ASSIGN_OR_RETURN(auto dst_buffer, dst_local->buffer->MapMemory<T>(
                                        MemoryAccess::kDiscardWrite));
  return KERNEL::Execute(lhs_buffer.contents(), rhs_buffer.contents(),
                         dst_buffer.mutable_contents(), args...);
}

template <typename KERNEL, typename T, typename... ARGS>
Status ApplyTernaryOp(BufferView* a_local, BufferView* b_local,
                      BufferView* c_local, BufferView* dst_local,
                      ARGS... args) {
  ASSIGN_OR_RETURN(auto a_buffer,
                   a_local->buffer->MapMemory<T>(MemoryAccess::kRead));
  ASSIGN_OR_RETURN(auto b_buffer,
                   b_local->buffer->MapMemory<T>(MemoryAccess::kRead));
  ASSIGN_OR_RETURN(auto c_buffer,
                   c_local->buffer->MapMemory<T>(MemoryAccess::kRead));
  ASSIGN_OR_RETURN(auto dst_buffer, dst_local->buffer->MapMemory<T>(
                                        MemoryAccess::kDiscardWrite));
  return KERNEL::Execute(a_buffer.contents(), b_buffer.contents(),
                         c_buffer.contents(), dst_buffer.mutable_contents(),
                         args...);
}

template <typename KERNEL, typename T>
Status ApplyComparisonOp(BufferView* lhs_local, BufferView* rhs_local,
                         BufferView* dst_local) {
  ASSIGN_OR_RETURN(auto lhs_buffer,
                   lhs_local->buffer->MapMemory<T>(MemoryAccess::kRead));
  ASSIGN_OR_RETURN(auto rhs_buffer,
                   rhs_local->buffer->MapMemory<T>(MemoryAccess::kRead));
  ASSIGN_OR_RETURN(auto dst_buffer, dst_local->buffer->MapMemory<uint8_t>(
                                        MemoryAccess::kDiscardWrite));
  return KERNEL::Execute(lhs_buffer.contents(), rhs_buffer.contents(),
                         dst_buffer.mutable_contents());
}

template <typename KERNEL, typename... ARGS>
Status ApplyUnaryOpIS(BufferView* src_local, BufferView* dst_local,
                      ARGS... args) {
  switch (src_local->element_size) {
    case 1:
      return ApplyUnaryOp<KERNEL, int8_t>(src_local, dst_local, args...);
    case 2:
      return ApplyUnaryOp<KERNEL, int16_t>(src_local, dst_local, args...);
    case 4:
      return ApplyUnaryOp<KERNEL, int32_t>(src_local, dst_local, args...);
    case 8:
      return ApplyUnaryOp<KERNEL, int64_t>(src_local, dst_local, args...);
    default:
      return UnimplementedErrorBuilder(IREE_LOC)
             << "Unimplemented element size: " << src_local->element_size;
  }
}

template <typename KERNEL, typename... ARGS>
Status ApplyUnaryOpIU(BufferView* src_local, BufferView* dst_local,
                      ARGS... args) {
  switch (src_local->element_size) {
    case 1:
      return ApplyUnaryOp<KERNEL, uint8_t>(src_local, dst_local, args...);
    case 2:
      return ApplyUnaryOp<KERNEL, uint16_t>(src_local, dst_local, args...);
    case 4:
      return ApplyUnaryOp<KERNEL, uint32_t>(src_local, dst_local, args...);
    case 8:
      return ApplyUnaryOp<KERNEL, uint64_t>(src_local, dst_local, args...);
    default:
      return UnimplementedErrorBuilder(IREE_LOC)
             << "Unimplemented element size: " << src_local->element_size;
  }
}

template <typename KERNEL, typename... ARGS>
Status ApplyUnaryOpF(BufferView* src_local, BufferView* dst_local,
                     ARGS... args) {
  switch (src_local->element_size) {
#if defined(IREE_SUPPORT_F32)
    case 4:
      return ApplyUnaryOp<KERNEL, float>(src_local, dst_local, args...);
#endif  // IREE_SUPPORT_F32
#if defined(IREE_SUPPORT_F64)
    case 8:
      return ApplyUnaryOp<KERNEL, double>(src_local, dst_local, args...);
#endif  // IREE_SUPPORT_F64
    default:
      return UnimplementedErrorBuilder(IREE_LOC)
             << "Unimplemented element size: " << src_local->element_size;
  }
}

template <typename KERNEL, typename... ARGS>
Status ApplyBinaryOpIS(BufferView* lhs_local, BufferView* rhs_local,
                       BufferView* dst_local, ARGS... args) {
  switch (lhs_local->element_size) {
    case 1:
      return ApplyBinaryOp<KERNEL, int8_t>(lhs_local, rhs_local, dst_local,
                                           args...);
    case 2:
      return ApplyBinaryOp<KERNEL, int16_t>(lhs_local, rhs_local, dst_local,
                                            args...);
    case 4:
      return ApplyBinaryOp<KERNEL, int32_t>(lhs_local, rhs_local, dst_local,
                                            args...);
    case 8:
      return ApplyBinaryOp<KERNEL, int64_t>(lhs_local, rhs_local, dst_local,
                                            args...);
    default:
      return UnimplementedErrorBuilder(IREE_LOC)
             << "Unimplemented element size: " << lhs_local->element_size;
  }
}

template <typename KERNEL, typename... ARGS>
Status ApplyBinaryOpIU(BufferView* lhs_local, BufferView* rhs_local,
                       BufferView* dst_local, ARGS... args) {
  switch (lhs_local->element_size) {
    case 1:
      return ApplyBinaryOp<KERNEL, uint8_t>(lhs_local, rhs_local, dst_local,
                                            args...);
    case 2:
      return ApplyBinaryOp<KERNEL, uint16_t>(lhs_local, rhs_local, dst_local,
                                             args...);
    case 4:
      return ApplyBinaryOp<KERNEL, uint32_t>(lhs_local, rhs_local, dst_local,
                                             args...);
    case 8:
      return ApplyBinaryOp<KERNEL, uint64_t>(lhs_local, rhs_local, dst_local,
                                             args...);
    default:
      return UnimplementedErrorBuilder(IREE_LOC)
             << "Unimplemented element size: " << lhs_local->element_size;
  }
}

template <typename KERNEL, typename... ARGS>
Status ApplyBinaryOpF(BufferView* lhs_local, BufferView* rhs_local,
                      BufferView* dst_local, ARGS... args) {
  switch (lhs_local->element_size) {
#if defined(IREE_SUPPORT_F32)
    case 4:
      return ApplyBinaryOp<KERNEL, float>(lhs_local, rhs_local, dst_local,
                                          args...);
#endif  // IREE_SUPPORT_F32
#if defined(IREE_SUPPORT_F64)
    case 8:
      return ApplyBinaryOp<KERNEL, double>(lhs_local, rhs_local, dst_local,
                                           args...);
#endif  // IREE_SUPPORT_F64
    default:
      return UnimplementedErrorBuilder(IREE_LOC)
             << "Unimplemented element size: " << lhs_local->element_size;
  }
}

template <typename KERNEL, typename... ARGS>
Status ApplyTernaryOpIS(BufferView* a_local, BufferView* b_local,
                        BufferView* c_local, BufferView* dst_local,
                        ARGS... args) {
  switch (a_local->element_size) {
    case 1:
      return ApplyTernaryOp<KERNEL, int8_t>(a_local, b_local, c_local,
                                            dst_local, args...);
    case 2:
      return ApplyTernaryOp<KERNEL, int16_t>(a_local, b_local, c_local,
                                             dst_local, args...);
    case 4:
      return ApplyTernaryOp<KERNEL, int32_t>(a_local, b_local, c_local,
                                             dst_local, args...);
    case 8:
      return ApplyTernaryOp<KERNEL, int64_t>(a_local, b_local, c_local,
                                             dst_local, args...);
    default:
      return UnimplementedErrorBuilder(IREE_LOC)
             << "Unimplemented element size: " << a_local->element_size;
  }
}

template <typename KERNEL, typename... ARGS>
Status ApplyTernaryOpIU(BufferView* a_local, BufferView* b_local,
                        BufferView* c_local, BufferView* dst_local,
                        ARGS... args) {
  switch (a_local->element_size) {
    case 1:
      return ApplyTernaryOp<KERNEL, uint8_t>(a_local, b_local, c_local,
                                             dst_local, args...);
    case 2:
      return ApplyTernaryOp<KERNEL, uint16_t>(a_local, b_local, c_local,
                                              dst_local, args...);
    case 4:
      return ApplyTernaryOp<KERNEL, uint32_t>(a_local, b_local, c_local,
                                              dst_local, args...);
    case 8:
      return ApplyTernaryOp<KERNEL, uint64_t>(a_local, b_local, c_local,
                                              dst_local, args...);
    default:
      return UnimplementedErrorBuilder(IREE_LOC)
             << "Unimplemented element size: " << a_local->element_size;
  }
}

template <typename KERNEL, typename... ARGS>
Status ApplyTernaryOpF(BufferView* a_local, BufferView* b_local,
                       BufferView* c_local, BufferView* dst_local,
                       ARGS... args) {
  switch (a_local->element_size) {
#if defined(IREE_SUPPORT_F32)
    case 4:
      return ApplyTernaryOp<KERNEL, float>(a_local, b_local, c_local, dst_local,
                                           args...);
#endif  // IREE_SUPPORT_F32
#if defined(IREE_SUPPORT_F64)
    case 8:
      return ApplyTernaryOp<KERNEL, double>(a_local, b_local, c_local,
                                            dst_local, args...);
#endif  // IREE_SUPPORT_F64
    default:
      return UnimplementedErrorBuilder(IREE_LOC)
             << "Unimplemented element size: " << a_local->element_size;
  }
}

template <typename KERNEL>
Status ApplyComparisonOpIS(BufferView* lhs_local, BufferView* rhs_local,
                           BufferView* dst_local) {
  switch (lhs_local->element_size) {
    case 1:
      return ApplyComparisonOp<KERNEL, int8_t>(lhs_local, rhs_local, dst_local);
    case 2:
      return ApplyComparisonOp<KERNEL, int16_t>(lhs_local, rhs_local,
                                                dst_local);
    case 4:
      return ApplyComparisonOp<KERNEL, int32_t>(lhs_local, rhs_local,
                                                dst_local);
    case 8:
      return ApplyComparisonOp<KERNEL, int64_t>(lhs_local, rhs_local,
                                                dst_local);
    default:
      return UnimplementedErrorBuilder(IREE_LOC)
             << "Unimplemented element size: " << lhs_local->element_size;
  }
}

template <typename KERNEL>
Status ApplyComparisonOpIU(BufferView* lhs_local, BufferView* rhs_local,
                           BufferView* dst_local) {
  switch (lhs_local->element_size) {
    case 1:
      return ApplyComparisonOp<KERNEL, uint8_t>(lhs_local, rhs_local,
                                                dst_local);
    case 2:
      return ApplyComparisonOp<KERNEL, uint16_t>(lhs_local, rhs_local,
                                                 dst_local);
    case 4:
      return ApplyComparisonOp<KERNEL, uint32_t>(lhs_local, rhs_local,
                                                 dst_local);
    case 8:
      return ApplyComparisonOp<KERNEL, uint64_t>(lhs_local, rhs_local,
                                                 dst_local);
    default:
      return UnimplementedErrorBuilder(IREE_LOC)
             << "Unimplemented element size: " << lhs_local->element_size;
  }
}

template <typename KERNEL>
Status ApplyComparisonOpF(BufferView* lhs_local, BufferView* rhs_local,
                          BufferView* dst_local) {
  switch (lhs_local->element_size) {
    case 4:
      return ApplyComparisonOp<KERNEL, float>(lhs_local, rhs_local, dst_local);
    case 8:
      return ApplyComparisonOp<KERNEL, double>(lhs_local, rhs_local, dst_local);
    default:
      return UnimplementedErrorBuilder(IREE_LOC)
             << "Unimplemented element size: " << lhs_local->element_size;
  }
}

template <typename T, typename ACC = int32_t>
Status ApplyMatMulOpI(kernels::MatMul::RuntimeState* runtime_state,
                      BufferView* lhs_local, BufferView* rhs_local,
                      BufferView* bias_local,
                      BufferView* multiplier_mantissa_local,
                      BufferView* multiplier_exponent_local,
                      BufferView* dst_local) {
  kernels::MatMul::Buffers<T, ACC> buffers;
  ASSIGN_OR_RETURN(auto lhs_buffer,
                   lhs_local->buffer->MapMemory<T>(MemoryAccess::kRead));
  buffers.lhs_buffer = lhs_buffer.contents();
  buffers.lhs_shape = lhs_local->shape;
  ASSIGN_OR_RETURN(auto rhs_buffer,
                   rhs_local->buffer->MapMemory<T>(MemoryAccess::kRead));
  buffers.rhs_buffer = rhs_buffer.contents();
  buffers.rhs_shape = rhs_local->shape;
  MappedMemory<ACC> bias_buffer;
  if (bias_local && bias_local->buffer && !bias_local->shape.empty()) {
    if (bias_local->element_size != sizeof(ACC)) {
      return UnimplementedErrorBuilder(IREE_LOC)
             << "Only " << sizeof(ACC) << "b biases are supported right now";
    }
    ASSIGN_OR_RETURN(bias_buffer,
                     bias_local->buffer->MapMemory<ACC>(MemoryAccess::kRead));
    buffers.bias_buffer = bias_buffer.contents();
  }
  ASSIGN_OR_RETURN(
      auto multiplier_mantissa_buffer,
      multiplier_mantissa_local->buffer->MapMemory<ACC>(MemoryAccess::kRead));
  buffers.multiplier_mantissa_buffer = multiplier_mantissa_buffer.contents();
  ASSIGN_OR_RETURN(auto multiplier_exponent_buffer,
                   multiplier_exponent_local->buffer->MapMemory<int32_t>(
                       MemoryAccess::kRead));
  buffers.multiplier_exponent_buffer = multiplier_exponent_buffer.contents();
  ASSIGN_OR_RETURN(auto dst_buffer, dst_local->buffer->MapMemory<T>(
                                        MemoryAccess::kDiscardWrite));
  buffers.dst_buffer = dst_buffer.mutable_contents();
  buffers.dst_shape = dst_local->shape;
  return kernels::MatMul::Execute(runtime_state, buffers);
}

template <typename T>
Status ApplyMatMulOpF(kernels::MatMul::RuntimeState* runtime_state,
                      BufferView* lhs_local, BufferView* rhs_local,
                      BufferView* bias_local, BufferView* dst_local) {
  kernels::MatMul::Buffers<T, T> buffers;
  ASSIGN_OR_RETURN(auto lhs_buffer,
                   lhs_local->buffer->MapMemory<T>(MemoryAccess::kRead));
  buffers.lhs_buffer = lhs_buffer.contents();
  buffers.lhs_shape = lhs_local->shape;
  ASSIGN_OR_RETURN(auto rhs_buffer,
                   rhs_local->buffer->MapMemory<T>(MemoryAccess::kRead));
  buffers.rhs_buffer = rhs_buffer.contents();
  buffers.rhs_shape = rhs_local->shape;
  MappedMemory<T> bias_buffer;
  if (bias_local && bias_local->buffer && !bias_local->shape.empty()) {
    ASSIGN_OR_RETURN(bias_buffer,
                     bias_local->buffer->MapMemory<T>(MemoryAccess::kRead));
    buffers.bias_buffer = bias_buffer.contents();
  }
  ASSIGN_OR_RETURN(auto dst_buffer, dst_local->buffer->MapMemory<T>(
                                        MemoryAccess::kDiscardWrite));
  buffers.dst_buffer = dst_buffer.mutable_contents();
  buffers.dst_shape = dst_local->shape;
  return kernels::MatMul::Execute(runtime_state, buffers);
}

template <typename KERNEL>
Status DispatchElementwiseUnaryOpIS(vm::BytecodeReader* reader) {
  ASSIGN_OR_RETURN(auto* src_local, reader->ReadLocal());
  ASSIGN_OR_RETURN(auto* dst_local, reader->ReadLocal());
  RETURN_IF_ERROR(ValidateElementwiseUnaryOp(src_local, dst_local));
  return ApplyUnaryOpIS<KERNEL>(src_local, dst_local);
}

template <typename KERNEL>
Status DispatchElementwiseUnaryOpIU(vm::BytecodeReader* reader) {
  ASSIGN_OR_RETURN(auto* src_local, reader->ReadLocal());
  ASSIGN_OR_RETURN(auto* dst_local, reader->ReadLocal());
  RETURN_IF_ERROR(ValidateElementwiseUnaryOp(src_local, dst_local));
  return ApplyUnaryOpIU<KERNEL>(src_local, dst_local);
}

template <typename KERNEL>
Status DispatchElementwiseUnaryOpF(vm::BytecodeReader* reader) {
  ASSIGN_OR_RETURN(auto* src_local, reader->ReadLocal());
  ASSIGN_OR_RETURN(auto* dst_local, reader->ReadLocal());
  RETURN_IF_ERROR(ValidateElementwiseUnaryOp(src_local, dst_local));
  return ApplyUnaryOpF<KERNEL>(src_local, dst_local);
}

template <typename KERNEL>
Status DispatchElementwiseBinaryOpIS(vm::BytecodeReader* reader) {
  ASSIGN_OR_RETURN(auto* lhs_local, reader->ReadLocal());
  ASSIGN_OR_RETURN(auto* rhs_local, reader->ReadLocal());
  ASSIGN_OR_RETURN(auto* dst_local, reader->ReadLocal());
  RETURN_IF_ERROR(ValidateElementwiseBinaryOp(lhs_local, rhs_local, dst_local));
  return ApplyBinaryOpIS<KERNEL>(lhs_local, rhs_local, dst_local);
}

template <typename KERNEL>
Status DispatchElementwiseBinaryOpIU(vm::BytecodeReader* reader) {
  ASSIGN_OR_RETURN(auto* lhs_local, reader->ReadLocal());
  ASSIGN_OR_RETURN(auto* rhs_local, reader->ReadLocal());
  ASSIGN_OR_RETURN(auto* dst_local, reader->ReadLocal());
  RETURN_IF_ERROR(ValidateElementwiseBinaryOp(lhs_local, rhs_local, dst_local));
  return ApplyBinaryOpIU<KERNEL>(lhs_local, rhs_local, dst_local);
}

template <typename KERNEL>
Status DispatchElementwiseBinaryOpF(vm::BytecodeReader* reader) {
  ASSIGN_OR_RETURN(auto* lhs_local, reader->ReadLocal());
  ASSIGN_OR_RETURN(auto* rhs_local, reader->ReadLocal());
  ASSIGN_OR_RETURN(auto* dst_local, reader->ReadLocal());
  RETURN_IF_ERROR(ValidateElementwiseBinaryOp(lhs_local, rhs_local, dst_local));
  return ApplyBinaryOpF<KERNEL>(lhs_local, rhs_local, dst_local);
}

template <typename KERNEL>
Status DispatchElementwiseTernaryOpIS(vm::BytecodeReader* reader) {
  ASSIGN_OR_RETURN(auto* a_local, reader->ReadLocal());
  ASSIGN_OR_RETURN(auto* b_local, reader->ReadLocal());
  ASSIGN_OR_RETURN(auto* c_local, reader->ReadLocal());
  ASSIGN_OR_RETURN(auto* dst_local, reader->ReadLocal());
  RETURN_IF_ERROR(
      ValidateElementwiseTernaryOp(a_local, b_local, c_local, dst_local));
  return ApplyTernaryOpIS<KERNEL>(a_local, b_local, c_local, dst_local);
}

template <typename KERNEL>
Status DispatchElementwiseTernaryOpIU(vm::BytecodeReader* reader) {
  ASSIGN_OR_RETURN(auto* a_local, reader->ReadLocal());
  ASSIGN_OR_RETURN(auto* b_local, reader->ReadLocal());
  ASSIGN_OR_RETURN(auto* c_local, reader->ReadLocal());
  ASSIGN_OR_RETURN(auto* dst_local, reader->ReadLocal());
  RETURN_IF_ERROR(
      ValidateElementwiseTernaryOp(a_local, b_local, c_local, dst_local));
  return ApplyTernaryOpIU<KERNEL>(a_local, b_local, c_local, dst_local);
}

template <typename KERNEL>
Status DispatchElementwiseTernaryOpF(vm::BytecodeReader* reader) {
  ASSIGN_OR_RETURN(auto* a_local, reader->ReadLocal());
  ASSIGN_OR_RETURN(auto* b_local, reader->ReadLocal());
  ASSIGN_OR_RETURN(auto* c_local, reader->ReadLocal());
  ASSIGN_OR_RETURN(auto* dst_local, reader->ReadLocal());
  RETURN_IF_ERROR(
      ValidateElementwiseTernaryOp(a_local, b_local, c_local, dst_local));
  return ApplyTernaryOpF<KERNEL>(a_local, b_local, c_local, dst_local);
}

Status ApplyCopy(BufferView* src_local, absl::Span<const int32_t> src_indices,
                 BufferView* dst_local, absl::Span<const int32_t> dst_indices,
                 absl::Span<const int32_t> lengths);

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_INTERPRETER_BYTECODE_DISPATCH_UTIL_H_
