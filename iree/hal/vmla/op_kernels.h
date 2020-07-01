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

// Defines kernel functions and provides their implementation via one (or more)
// included files.
//
// Kernels should do the simplest possible operation. Buffer validation is
// handled by the dispatch logic and need not be checked. Kernels may optionally
// accept arguments beyond just the buffers, depending on the required state
// and attributes.
//
// Kernels may optionally have runtime state. This is state that is allocated
// once for the entire Runtime (and stored on RuntimeState) and shared across
// all fibers. This enables kernels that may require thread pools or device
// handles to be shared while kernels that require transient storage to be safe
// to use from multiple fibers concurrently.
//
// All kernels are templated to enable specialization of particular types or
// type combinations. By default the op_kernels_generic.h will provide C++
// semantics as reference and platform-specific versions can be implemented
// as needed.

#ifndef IREE_HAL_VMLA_OP_KERNELS_H_
#define IREE_HAL_VMLA_OP_KERNELS_H_

#include <cstdint>

#include "absl/types/span.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"

namespace iree {
namespace hal {
namespace vmla {
namespace kernels {

using ShapeSpan = absl::Span<const int32_t>;

inline size_t GetElementCount(ShapeSpan shape) {
  size_t count = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    count *= shape[i];
  }
  return count;
}

struct CompareEQ {
  template <typename T>
  static Status Execute(absl::Span<const T> lhs_buffer,
                        absl::Span<const T> rhs_buffer,
                        absl::Span<uint8_t> dst_buffer);
};
struct CompareNE {
  template <typename T>
  static Status Execute(absl::Span<const T> lhs_buffer,
                        absl::Span<const T> rhs_buffer,
                        absl::Span<uint8_t> dst_buffer);
};
struct CompareLT {
  template <typename T>
  static Status Execute(absl::Span<const T> lhs_buffer,
                        absl::Span<const T> rhs_buffer,
                        absl::Span<uint8_t> dst_buffer);
};
struct CompareLE {
  template <typename T>
  static Status Execute(absl::Span<const T> lhs_buffer,
                        absl::Span<const T> rhs_buffer,
                        absl::Span<uint8_t> dst_buffer);
};
struct CompareGT {
  template <typename T>
  static Status Execute(absl::Span<const T> lhs_buffer,
                        absl::Span<const T> rhs_buffer,
                        absl::Span<uint8_t> dst_buffer);
};
struct CompareGE {
  template <typename T>
  static Status Execute(absl::Span<const T> lhs_buffer,
                        absl::Span<const T> rhs_buffer,
                        absl::Span<uint8_t> dst_buffer);
};

struct Conv2D {
  template <typename T>
  static Status Execute(absl::Span<const T> input_buffer, ShapeSpan input_shape,
                        absl::Span<const T> filter_buffer,
                        ShapeSpan filter_shape, absl::Span<T> dst_buffer,
                        ShapeSpan dst_shape, ShapeSpan strides, ShapeSpan pad_h,
                        ShapeSpan pad_w, ShapeSpan dilation,
                        const int32_t groups);
};

struct Copy {
  template <int element_size>
  static Status Execute(absl::Span<const uint8_t> src_buffer,
                        ShapeSpan src_shape,
                        absl::Span<const int32_t> src_indices,
                        absl::Span<uint8_t> dst_buffer, ShapeSpan dst_shape,
                        absl::Span<const int32_t> dst_indices,
                        absl::Span<const int32_t> lengths);
};

struct Select {
  template <typename T>
  static Status Execute(absl::Span<const uint8_t> cond_buffer,
                        absl::Span<const T> lhs_buffer,
                        absl::Span<const T> rhs_buffer,
                        absl::Span<T> dst_buffer);
};

struct Transpose {
  template <typename T>
  static Status Execute(absl::Span<const T> src_buffer,
                        absl::Span<T> dst_buffer, ShapeSpan src_shape,
                        absl::Span<const int32_t> perm);
};

struct Pad {
  template <typename T>
  static Status Execute(absl::Span<const T> src_buffer,
                        absl::Span<const T> padding_value,
                        absl::Span<T> dst_buffer, ShapeSpan src_shape,
                        ShapeSpan dst_shape,
                        absl::Span<const int32_t> edge_padding_low,
                        absl::Span<const int32_t> edge_padding_high,
                        absl::Span<const int32_t> interior_padding);
};

struct Gather {
  template <typename T>
  static Status Execute(absl::Span<const T> src_buffer,
                        absl::Span<const int32_t> indices_buffer,
                        absl::Span<T> dst_buffer, ShapeSpan src_shape,
                        ShapeSpan indices_shape, ShapeSpan dst_shape,
                        const int32_t dim, const int32_t batch_dims);
};

struct Scatter {
  template <typename T>
  static Status Execute(absl::Span<const T> src_buffer,
                        absl::Span<const int32_t> indices_buffer,
                        absl::Span<T> dst_buffer, ShapeSpan src_shape,
                        ShapeSpan indices_shape, ShapeSpan dst_shape);
};

struct Reverse {
  template <typename T>
  static Status Execute(absl::Span<const T> src_buffer,
                        absl::Span<T> dst_buffer, ShapeSpan src_shape,
                        absl::Span<const int32_t> dimensions);
};

struct Broadcast {
  template <typename T>
  static Status Execute(absl::Span<const T> src_buffer,
                        absl::Span<T> dst_buffer);
};

struct Tile {
  template <typename T>
  static Status Execute(absl::Span<const T> src_buffer,
                        absl::Span<T> dst_buffer, ShapeSpan src_shape,
                        ShapeSpan dst_shape);
};

struct Not {
  template <typename T>
  static Status Execute(absl::Span<const T> src_buffer,
                        absl::Span<T> dst_buffer);
};

struct And {
  template <typename T>
  static Status Execute(absl::Span<const T> lhs_buffer,
                        absl::Span<const T> rhs_buffer,
                        absl::Span<T> dst_buffer);
};

struct Or {
  template <typename T>
  static Status Execute(absl::Span<const T> lhs_buffer,
                        absl::Span<const T> rhs_buffer,
                        absl::Span<T> dst_buffer);
};

struct Xor {
  template <typename T>
  static Status Execute(absl::Span<const T> lhs_buffer,
                        absl::Span<const T> rhs_buffer,
                        absl::Span<T> dst_buffer);
};

struct ShiftLeft {
  template <typename T>
  static Status Execute(absl::Span<const T> lhs_buffer,
                        absl::Span<const T> rhs_buffer,
                        absl::Span<T> dst_buffer);
};

struct ShiftRight {
  template <typename T>
  static Status Execute(absl::Span<const T> lhs_buffer,
                        absl::Span<const T> rhs_buffer,
                        absl::Span<T> dst_buffer);
};

struct Add {
  template <typename T>
  static Status Execute(absl::Span<const T> lhs_buffer,
                        absl::Span<const T> rhs_buffer,
                        absl::Span<T> dst_buffer);
};

struct Sub {
  template <typename T>
  static Status Execute(absl::Span<const T> lhs_buffer,
                        absl::Span<const T> rhs_buffer,
                        absl::Span<T> dst_buffer);
};

struct Abs {
  template <typename T>
  static Status Execute(absl::Span<const T> src_buffer,
                        absl::Span<T> dst_buffer);
};

struct Neg {
  template <typename T>
  static Status Execute(absl::Span<const T> src_buffer,
                        absl::Span<T> dst_buffer);
};

struct Mul {
  template <typename T>
  static Status Execute(absl::Span<const T> lhs_buffer,
                        absl::Span<const T> rhs_buffer,
                        absl::Span<T> dst_buffer);
};

struct Div {
  template <typename T>
  static Status Execute(absl::Span<const T> lhs_buffer,
                        absl::Span<const T> rhs_buffer,
                        absl::Span<T> dst_buffer);
};

struct Rem {
  template <typename T>
  static Status Execute(absl::Span<const T> lhs_buffer,
                        absl::Span<const T> rhs_buffer,
                        absl::Span<T> dst_buffer);
};

struct Pow {
  template <typename T>
  static Status Execute(absl::Span<const T> lhs_buffer,
                        absl::Span<const T> rhs_buffer,
                        absl::Span<T> dst_buffer);
};

struct Exp {
  template <typename T>
  static Status Execute(absl::Span<const T> src_buffer,
                        absl::Span<T> dst_buffer);
};

struct Log {
  template <typename T>
  static Status Execute(absl::Span<const T> src_buffer,
                        absl::Span<T> dst_buffer);
};

struct Rsqrt {
  template <typename T>
  static Status Execute(absl::Span<const T> src_buffer,
                        absl::Span<T> dst_buffer);
};

struct Sqrt {
  template <typename T>
  static Status Execute(absl::Span<const T> src_buffer,
                        absl::Span<T> dst_buffer);
};

struct Cos {
  template <typename T>
  static Status Execute(absl::Span<const T> src_buffer,
                        absl::Span<T> dst_buffer);
};

struct Sin {
  template <typename T>
  static Status Execute(absl::Span<const T> src_buffer,
                        absl::Span<T> dst_buffer);
};

struct Tanh {
  template <typename T>
  static Status Execute(absl::Span<const T> src_buffer,
                        absl::Span<T> dst_buffer);
};

struct Atan2 {
  template <typename T>
  static Status Execute(absl::Span<const T> lhs_buffer,
                        absl::Span<const T> rhs_buffer,
                        absl::Span<T> dst_buffer);
};

struct Min {
  template <typename T>
  static Status Execute(absl::Span<const T> lhs_buffer,
                        absl::Span<const T> rhs_buffer,
                        absl::Span<T> dst_buffer);
};

struct Max {
  template <typename T>
  static Status Execute(absl::Span<const T> lhs_buffer,
                        absl::Span<const T> rhs_buffer,
                        absl::Span<T> dst_buffer);
};

struct Clamp {
  template <typename T>
  static Status Execute(absl::Span<const T> min_buffer,
                        absl::Span<const T> src_buffer,
                        absl::Span<const T> max_buffer,
                        absl::Span<T> dst_buffer);
};

struct Floor {
  template <typename T>
  static Status Execute(absl::Span<const T> src_buffer,
                        absl::Span<T> dst_buffer);
};

struct Ceil {
  template <typename T>
  static Status Execute(absl::Span<const T> src_buffer,
                        absl::Span<T> dst_buffer);
};

struct Convert {
  template <typename SRC, typename DST>
  static Status Execute(absl::Span<const SRC> src_buffer,
                        absl::Span<DST> dst_buffer);
};

struct MatMul {
  struct RuntimeState;

  static std::unique_ptr<RuntimeState> CreateRuntimeState();

  template <typename T, typename ACC>
  struct Buffers {
    ShapeSpan lhs_shape;
    absl::Span<const T> lhs_buffer;
    ShapeSpan rhs_shape;
    absl::Span<const T> rhs_buffer;
    ShapeSpan dst_shape;
    absl::Span<T> dst_buffer;

    // Optional bias buffer.
    absl::Span<const ACC> bias_buffer;

    // Fixed-point multiplier mantissa/exponent. May be a single value (for
    // uniform quantization) or one element per row of the destination matrix
    // for per-channel.
    absl::Span<const ACC> multiplier_mantissa_buffer;
    absl::Span<const int32_t> multiplier_exponent_buffer;
  };

  template <typename T, typename ACC>
  static Status Execute(RuntimeState* runtime_state,
                        const Buffers<T, ACC>& buffers);
};

struct RuntimeState {
  std::unique_ptr<MatMul::RuntimeState> mat_mul_state =
      MatMul::CreateRuntimeState();
};

struct ReduceSum {
  template <typename T>
  static Status Execute(absl::Span<const T> src_buffer,
                        absl::Span<const T> init_buffer,
                        absl::Span<T> dst_buffer, int32_t dimension,
                        ShapeSpan src_shape, ShapeSpan dst_shape);
};

struct ReduceMin {
  template <typename T>
  static Status Execute(absl::Span<const T> src_buffer,
                        absl::Span<const T> init_buffer,
                        absl::Span<T> dst_buffer, int32_t dimension,
                        ShapeSpan src_shape, ShapeSpan dst_shape);
};

struct ReduceMax {
  template <typename T>
  static Status Execute(absl::Span<const T> src_buffer,
                        absl::Span<const T> init_buffer,
                        absl::Span<T> dst_buffer, int32_t dimension,
                        ShapeSpan src_shape, ShapeSpan dst_shape);
};

struct PoolingSum {
  template <typename T>
  static Status Execute(absl::Span<const T> src_buffer,
                        absl::Span<const T> init_buffer,
                        absl::Span<T> dst_buffer, ShapeSpan src_shape,
                        ShapeSpan dst_shape, ShapeSpan window_dimensions,
                        ShapeSpan strides, ShapeSpan pad_low);
};

struct PoolingMin {
  template <typename T>
  static Status Execute(absl::Span<const T> src_buffer,
                        absl::Span<const T> init_buffer,
                        absl::Span<T> dst_buffer, ShapeSpan src_shape,
                        ShapeSpan dst_shape, ShapeSpan window_dimensions,
                        ShapeSpan strides, ShapeSpan pad_low);
};

struct PoolingMax {
  template <typename T>
  static Status Execute(absl::Span<const T> src_buffer,
                        absl::Span<const T> init_buffer,
                        absl::Span<T> dst_buffer, ShapeSpan src_shape,
                        ShapeSpan dst_shape, ShapeSpan window_dimensions,
                        ShapeSpan strides, ShapeSpan pad_low);
};

}  // namespace kernels
}  // namespace vmla
}  // namespace hal
}  // namespace iree

#include "iree/hal/vmla/op_kernels_generic.h"  // IWYU pragma: export
#include "iree/hal/vmla/op_kernels_ruy.h"  // IWYU pragma: export

#endif  // IREE_HAL_VMLA_OP_KERNELS_H_
