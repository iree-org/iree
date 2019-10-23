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

// Conversion helper tables.

#ifndef IREE_HAL_INTERPRETER_BYTECODE_DISPATCH_CONVERSION_H_
#define IREE_HAL_INTERPRETER_BYTECODE_DISPATCH_CONVERSION_H_

#include "iree/base/status.h"
#include "iree/hal/buffer_view.h"
#include "iree/hal/interpreter/bytecode_dispatch_util.h"
#include "iree/schemas/bytecode/interpreter_bytecode_v0.h"
#include "iree/vm/type.h"

namespace iree {
namespace hal {

template <typename KERNEL, bool src_signed, bool dst_signed, typename... ARGS>
struct ApplyConversionOp {
  static Status Apply(const vm::Type& src_type, BufferView* src_local,
                      const vm::Type& dst_type, BufferView* dst_local,
                      ARGS... args) {
    // Validate ranges so that we cannot go out of bounds on thunk table.
    int src_type_index = src_type.type_index();
    int dst_type_index = dst_type.type_index();
    if (src_type_index < 0 || src_type_index >= kBuiltinTypeCount) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Conversion from invalid source builtin type "
             << src_type_index;
    } else if (dst_type_index < 0 || dst_type_index >= kBuiltinTypeCount) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Conversion to invalid dest builtin type " << dst_type_index;
    }

    // All possible combinations of conversions.
    using KernelFn = Status (*)(BufferView * src_local, BufferView * dst_local,
                                ARGS... args);
    KernelFn fn = nullptr;
    if (src_signed && dst_signed) {
      // Signed -> signed.
      static const KernelFn
          kConversionTable[kBuiltinTypeCount * kBuiltinTypeCount] = {
              // src_type = kI8:
              /* kI8 */ Thunk<int8_t, int8_t>::Apply,
              /* kI16 */ Thunk<int8_t, int16_t>::Apply,
              /* kI32 */ Thunk<int8_t, int32_t>::Apply,
              /* kI64 */ Thunk<int8_t, int64_t>::Apply,
              /* kF16 */ nullptr,
              /* kF32 */ Thunk<int8_t, float>::Apply,
              /* kF64 */ Thunk<int8_t, double>::Apply,

              // src_type = kI16:
              /* kI8 */ Thunk<int16_t, int8_t>::Apply,
              /* kI16 */ Thunk<int16_t, int16_t>::Apply,
              /* kI32 */ Thunk<int16_t, int32_t>::Apply,
              /* kI64 */ Thunk<int16_t, int64_t>::Apply,
              /* kF16 */ nullptr,
              /* kF32 */ Thunk<int16_t, float>::Apply,
              /* kF64 */ Thunk<int16_t, double>::Apply,

              // src_type = kI32:
              /* kI8 */ Thunk<int32_t, int8_t>::Apply,
              /* kI16 */ Thunk<int32_t, int16_t>::Apply,
              /* kI32 */ Thunk<int32_t, int32_t>::Apply,
              /* kI64 */ Thunk<int32_t, int64_t>::Apply,
              /* kF16 */ nullptr,
              /* kF32 */ Thunk<int32_t, float>::Apply,
              /* kF64 */ Thunk<int32_t, double>::Apply,

              // src_type = kI64:
              /* kI8 */ Thunk<int64_t, int8_t>::Apply,
              /* kI16 */ Thunk<int64_t, int16_t>::Apply,
              /* kI32 */ Thunk<int64_t, int32_t>::Apply,
              /* kI64 */ Thunk<int64_t, int64_t>::Apply,
              /* kF16 */ nullptr,
              /* kF32 */ Thunk<int64_t, float>::Apply,
              /* kF64 */ Thunk<int64_t, double>::Apply,

              // src_type = kF16:
              /* kI8 */ nullptr,
              /* kI16 */ nullptr,
              /* kI32 */ nullptr,
              /* kI64 */ nullptr,
              /* kF16 */ Thunk<uint16_t, uint16_t>::Apply,
              /* kF32 */ nullptr,
              /* kF64 */ nullptr,

              // src_type = kF32:
              /* kI8 */ Thunk<float, int8_t>::Apply,
              /* kI16 */ Thunk<float, int16_t>::Apply,
              /* kI32 */ Thunk<float, int32_t>::Apply,
              /* kI64 */ Thunk<float, int64_t>::Apply,
              /* kF16 */ nullptr,
              /* kF32 */ Thunk<float, float>::Apply,
              /* kF64 */ Thunk<float, double>::Apply,

              // src_type = kF64:
              /* kI8 */ Thunk<double, int8_t>::Apply,
              /* kI16 */ Thunk<double, int16_t>::Apply,
              /* kI32 */ Thunk<double, int32_t>::Apply,
              /* kI64 */ Thunk<double, int64_t>::Apply,
              /* kF16 */ nullptr,
              /* kF32 */ Thunk<double, float>::Apply,
              /* kF64 */ Thunk<double, double>::Apply,
          };
      fn =
          kConversionTable[src_type_index * kBuiltinTypeCount + dst_type_index];
    } else if (src_signed && !dst_signed) {
      // Signed -> unsigned.
      static const KernelFn
          kConversionTable[kBuiltinTypeCount * kBuiltinTypeCount] = {
              // src_type = kI8:
              /* kI8 */ Thunk<int8_t, uint8_t>::Apply,
              /* kI16 */ Thunk<int8_t, uint16_t>::Apply,
              /* kI32 */ Thunk<int8_t, uint32_t>::Apply,
              /* kI64 */ Thunk<int8_t, uint64_t>::Apply,
              /* kF16 */ nullptr,
              /* kF32 */ nullptr,
              /* kF64 */ nullptr,

              // src_type = kI16:
              /* kI8 */ Thunk<int16_t, uint8_t>::Apply,
              /* kI16 */ Thunk<int16_t, uint16_t>::Apply,
              /* kI32 */ Thunk<int16_t, uint32_t>::Apply,
              /* kI64 */ Thunk<int16_t, uint64_t>::Apply,
              /* kF16 */ nullptr,
              /* kF32 */ nullptr,
              /* kF64 */ nullptr,

              // src_type = kI32:
              /* kI8 */ Thunk<int32_t, uint8_t>::Apply,
              /* kI16 */ Thunk<int32_t, uint16_t>::Apply,
              /* kI32 */ Thunk<int32_t, uint32_t>::Apply,
              /* kI64 */ Thunk<int32_t, uint64_t>::Apply,
              /* kF16 */ nullptr,
              /* kF32 */ nullptr,
              /* kF64 */ nullptr,

              // src_type = kI64:
              /* kI8 */ Thunk<int64_t, uint8_t>::Apply,
              /* kI16 */ Thunk<int64_t, uint16_t>::Apply,
              /* kI32 */ Thunk<int64_t, uint32_t>::Apply,
              /* kI64 */ Thunk<int64_t, uint64_t>::Apply,
              /* kF16 */ nullptr,
              /* kF32 */ nullptr,
              /* kF64 */ nullptr,

              // src_type = kF16:
              /* kI8 */ nullptr,
              /* kI16 */ nullptr,
              /* kI32 */ nullptr,
              /* kI64 */ nullptr,
              /* kF16 */ nullptr,
              /* kF32 */ nullptr,
              /* kF64 */ nullptr,

              // src_type = kF32:
              /* kI8 */ Thunk<float, uint8_t>::Apply,
              /* kI16 */ Thunk<float, uint16_t>::Apply,
              /* kI32 */ Thunk<float, uint32_t>::Apply,
              /* kI64 */ Thunk<float, uint64_t>::Apply,
              /* kF16 */ nullptr,
              /* kF32 */ nullptr,
              /* kF64 */ nullptr,

              // src_type = kF64:
              /* kI8 */ Thunk<double, uint8_t>::Apply,
              /* kI16 */ Thunk<double, uint16_t>::Apply,
              /* kI32 */ Thunk<double, uint32_t>::Apply,
              /* kI64 */ Thunk<double, uint64_t>::Apply,
              /* kF16 */ nullptr,
              /* kF32 */ nullptr,
              /* kF64 */ nullptr,
          };
      fn =
          kConversionTable[src_type_index * kBuiltinTypeCount + dst_type_index];
    } else if (!src_signed && dst_signed) {
      // Unsigned -> signed.
      static const KernelFn
          kConversionTable[kBuiltinTypeCount * kBuiltinTypeCount] = {
              // src_type = kI8:
              /* kI8 */ Thunk<uint8_t, int8_t>::Apply,
              /* kI16 */ Thunk<uint8_t, int16_t>::Apply,
              /* kI32 */ Thunk<uint8_t, int32_t>::Apply,
              /* kI64 */ Thunk<uint8_t, int64_t>::Apply,
              /* kF16 */ nullptr,
              /* kF32 */ Thunk<uint8_t, float>::Apply,
              /* kF64 */ Thunk<uint8_t, double>::Apply,

              // src_type = kI16:
              /* kI8 */ Thunk<uint16_t, int8_t>::Apply,
              /* kI16 */ Thunk<uint16_t, int16_t>::Apply,
              /* kI32 */ Thunk<uint16_t, int32_t>::Apply,
              /* kI64 */ Thunk<uint16_t, int64_t>::Apply,
              /* kF16 */ nullptr,
              /* kF32 */ Thunk<uint16_t, float>::Apply,
              /* kF64 */ Thunk<uint16_t, double>::Apply,

              // src_type = kI32:
              /* kI8 */ Thunk<uint32_t, int8_t>::Apply,
              /* kI16 */ Thunk<uint32_t, int16_t>::Apply,
              /* kI32 */ Thunk<uint32_t, int32_t>::Apply,
              /* kI64 */ Thunk<uint32_t, int64_t>::Apply,
              /* kF16 */ nullptr,
              /* kF32 */ Thunk<uint32_t, float>::Apply,
              /* kF64 */ Thunk<uint32_t, double>::Apply,

              // src_type = kI64:
              /* kI8 */ Thunk<uint64_t, int8_t>::Apply,
              /* kI16 */ Thunk<uint64_t, int16_t>::Apply,
              /* kI32 */ Thunk<uint64_t, int32_t>::Apply,
              /* kI64 */ Thunk<uint64_t, int64_t>::Apply,
              /* kF16 */ nullptr,
              /* kF32 */ Thunk<uint64_t, float>::Apply,
              /* kF64 */ Thunk<uint64_t, double>::Apply,

              // src_type = kF16:
              /* kI8 */ nullptr,
              /* kI16 */ nullptr,
              /* kI32 */ nullptr,
              /* kI64 */ nullptr,
              /* kF16 */ nullptr,
              /* kF32 */ nullptr,
              /* kF64 */ nullptr,

              // src_type = kF32:
              /* kI8 */ nullptr,
              /* kI16 */ nullptr,
              /* kI32 */ nullptr,
              /* kI64 */ nullptr,
              /* kF16 */ nullptr,
              /* kF32 */ nullptr,
              /* kF64 */ nullptr,

              // src_type = kF64:
              /* kI8 */ nullptr,
              /* kI16 */ nullptr,
              /* kI32 */ nullptr,
              /* kI64 */ nullptr,
              /* kF16 */ nullptr,
              /* kF32 */ nullptr,
              /* kF64 */ nullptr,
          };
      fn =
          kConversionTable[src_type_index * kBuiltinTypeCount + dst_type_index];
    } else if (!src_signed && !dst_signed) {
      // Unsigned -> unsigned.
      static const KernelFn
          kConversionTable[kBuiltinTypeCount * kBuiltinTypeCount] = {
              // src_type = kI8:
              /* kI8 */ Thunk<uint8_t, uint8_t>::Apply,
              /* kI16 */ Thunk<uint8_t, uint16_t>::Apply,
              /* kI32 */ Thunk<uint8_t, uint32_t>::Apply,
              /* kI64 */ Thunk<uint8_t, uint64_t>::Apply,
              /* kF16 */ nullptr,
              /* kF32 */ nullptr,
              /* kF64 */ nullptr,

              // src_type = kI16:
              /* kI8 */ Thunk<uint16_t, uint8_t>::Apply,
              /* kI16 */ Thunk<uint16_t, uint16_t>::Apply,
              /* kI32 */ Thunk<uint16_t, uint32_t>::Apply,
              /* kI64 */ Thunk<uint16_t, uint64_t>::Apply,
              /* kF16 */ nullptr,
              /* kF32 */ nullptr,
              /* kF64 */ nullptr,

              // src_type = kI32:
              /* kI8 */ Thunk<uint32_t, uint8_t>::Apply,
              /* kI16 */ Thunk<uint32_t, uint16_t>::Apply,
              /* kI32 */ Thunk<uint32_t, uint32_t>::Apply,
              /* kI64 */ Thunk<uint32_t, uint64_t>::Apply,
              /* kF16 */ nullptr,
              /* kF32 */ nullptr,
              /* kF64 */ nullptr,

              // src_type = kI64:
              /* kI8 */ Thunk<uint64_t, uint8_t>::Apply,
              /* kI16 */ Thunk<uint64_t, uint16_t>::Apply,
              /* kI32 */ Thunk<uint64_t, uint32_t>::Apply,
              /* kI64 */ Thunk<uint64_t, uint64_t>::Apply,
              /* kF16 */ nullptr,
              /* kF32 */ nullptr,
              /* kF64 */ nullptr,

              // src_type = kF16:
              /* kI8 */ nullptr,
              /* kI16 */ nullptr,
              /* kI32 */ nullptr,
              /* kI64 */ nullptr,
              /* kF16 */ nullptr,
              /* kF32 */ nullptr,
              /* kF64 */ nullptr,

              // src_type = kF32:
              /* kI8 */ nullptr,
              /* kI16 */ nullptr,
              /* kI32 */ nullptr,
              /* kI64 */ nullptr,
              /* kF16 */ nullptr,
              /* kF32 */ nullptr,
              /* kF64 */ nullptr,

              // src_type = kF64:
              /* kI8 */ nullptr,
              /* kI16 */ nullptr,
              /* kI32 */ nullptr,
              /* kI64 */ nullptr,
              /* kF16 */ nullptr,
              /* kF32 */ nullptr,
              /* kF64 */ nullptr,
          };
      fn =
          kConversionTable[src_type_index * kBuiltinTypeCount + dst_type_index];
    }
    if (!fn) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Unsupported conversion from " << src_type_index << " to "
             << dst_type_index;
    }
    return fn(src_local, dst_local, args...);
  }

  template <typename SRC, typename DST>
  struct Thunk {
    static Status Apply(BufferView* src_local, BufferView* dst_local,
                        ARGS... args) {
      ASSIGN_OR_RETURN(auto src_buffer,
                       src_local->buffer->MapMemory<SRC>(MemoryAccess::kRead));
      ASSIGN_OR_RETURN(auto dst_buffer, dst_local->buffer->MapMemory<DST>(
                                            MemoryAccess::kDiscardWrite));
      return KERNEL::Execute(src_buffer.contents(),
                             dst_buffer.mutable_contents(), args...);
    }
  };

// Disable F32/F64 conversions if they are not supported.
#if !defined(IREE_SUPPORT_F32)
  template <typename DST>
  struct Thunk<float, DST> {
    static Status Apply(BufferView* src_local, BufferView* dst_local,
                        ARGS... args) {
      return UnimplementedErrorBuilder(IREE_LOC) << "F32 not supported";
    }
  };
  template <typename SRC>
  struct Thunk<SRC, float> {
    static Status Apply(BufferView* src_local, BufferView* dst_local,
                        ARGS... args) {
      return UnimplementedErrorBuilder(IREE_LOC) << "F32 not supported";
    }
  };
#endif  // !IREE_SUPPORT_F32
#if !defined(IREE_SUPPORT_F64)
  template <typename DST>
  struct Thunk<double, DST> {
    static Status Apply(BufferView* src_local, BufferView* dst_local,
                        ARGS... args) {
      return UnimplementedErrorBuilder(IREE_LOC) << "F64 not supported";
    }
  };
  template <typename SRC>
  struct Thunk<SRC, double> {
    static Status Apply(BufferView* src_local, BufferView* dst_local,
                        ARGS... args) {
      return UnimplementedErrorBuilder(IREE_LOC) << "F64 not supported";
    }
  };
#endif  // !IREE_SUPPORT_F64
};

using ApplyConvertSS = ApplyConversionOp<kernels::Convert, /*src_signed=*/true,
                                         /*dst_signed=*/true>;
using ApplyConvertUU = ApplyConversionOp<kernels::Convert, /*src_signed=*/false,
                                         /*dst_signed=*/false>;
using ApplyConvertSU = ApplyConversionOp<kernels::Convert, /*src_signed=*/true,
                                         /*dst_signed=*/false>;
using ApplyConvertUS = ApplyConversionOp<kernels::Convert, /*src_signed=*/false,
                                         /*dst_signed=*/true>;

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_INTERPRETER_BYTECODE_DISPATCH_CONVERSION_H_
