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

#ifndef IREE_HAL_INTERPRETER_BYTECODE_KERNELS_RUY_H_
#define IREE_HAL_INTERPRETER_BYTECODE_KERNELS_RUY_H_

#include "absl/base/thread_annotations.h"
#include "absl/memory/memory.h"
#include "iree/base/status.h"
#include "iree/hal/buffer_view.h"
#include "tensorflow/lite/experimental/ruy/context.h"
#include "tensorflow/lite/experimental/ruy/ruy.h"

namespace iree {
namespace hal {
namespace kernels {

// TODO(benvanik): something more clever for making this shareable.
// Maybe a factory fn based on the impl selected?
struct MatMul::RuntimeState {
  // TODO(benvanik): share the thread pool but keep context per-fiber?
  ruy::Context context;
};

inline std::unique_ptr<MatMul::RuntimeState> MatMul::CreateRuntimeState() {
  return absl::make_unique<RuntimeState>();
}

template <typename T>
void MatMul::Transpose2D(int d0, int d1, const T* input_data, T* output_data) {
  const int kLines = 4;
  const int kSkipSize = (kLines - 1) * d1;

  const T* input = input_data;

  int i = 0;
  for (; i <= d0 - kLines; i += kLines) {
    T* output = output_data + i;

    const T* input_ptr = input;
    preload_l1_keep(input_ptr);
    input_ptr += d1;
    preload_l1_keep(input_ptr);
    input_ptr += d1;
    preload_l1_keep(input_ptr);
    input_ptr += d1;
    preload_l1_keep(input_ptr);

    int j = 0;
    for (; j <= d1 - kLines; j += kLines) {
      input_ptr = input;
      const T a00 = input_ptr[0];
      const T a01 = input_ptr[1];
      const T a02 = input_ptr[2];
      const T a03 = input_ptr[3];
      input_ptr += d1;
      const T a10 = input_ptr[0];
      const T a11 = input_ptr[1];
      const T a12 = input_ptr[2];
      const T a13 = input_ptr[3];
      input_ptr += d1;
      const T a20 = input_ptr[0];
      const T a21 = input_ptr[1];
      const T a22 = input_ptr[2];
      const T a23 = input_ptr[3];
      input_ptr += d1;
      const T a30 = input_ptr[0];
      const T a31 = input_ptr[1];
      const T a32 = input_ptr[2];
      const T a33 = input_ptr[3];

      output[0] = a00;
      output[1] = a10;
      output[2] = a20;
      output[3] = a30;
      output += d0;

      output[0] = a01;
      output[1] = a11;
      output[2] = a21;
      output[3] = a31;
      output += d0;

      output[0] = a02;
      output[1] = a12;
      output[2] = a22;
      output[3] = a32;
      output += d0;

      output[0] = a03;
      output[1] = a13;
      output[2] = a23;
      output[3] = a33;
      output += d0;

      input += kLines;
    }
    if (j == d1) {
      input += kSkipSize;
    } else {
      for (int p = 0; p < kLines; ++p) {
        for (int q = 0; q < d1 - j; ++q) {
          *(output + q * d0 + p) = *(input + p * d1 + q);
        }
      }
      input += (d1 - j) + kSkipSize;
    }
  }
  for (; i < d0; ++i) {
    T* output = output_data + i;
    for (int j = 0; j < d1; ++j) {
      *output = *input;
      output += d0;
      ++input;
    }
  }
}

template <typename T, typename ACC>
Status MatMul::Execute(RuntimeState* runtime_state,
                       const Buffers<T, ACC>& buffers) {
  // Note that it is important to invoke RUY in RCC mode (LHS=Row Major,
  // RHS=Col Major, Result=Col Major), which necessitates some transposes. This
  // is not a long term solution and is just to get it on the optimized paths
  // until the compiler can reason properly about layout and pre-packing, which
  // is the anticipated future state. This is doing (A * B^T)^T.
  // If needed, it can also be re-arranged to (B * A^T).
  T* a_data;
  int a_d0, a_d1;
  T* b_data;
  int b_d0, b_d1;
  T* r_data;
  int r_d0, r_d1;

  bool transpose_dst = false;
  T* temp1_buffer = nullptr;
  T* temp2_buffer = nullptr;
  {
    // Do (A * B^T)^T
    IREE_TRACE_SCOPE0("MatMul#TransposeRhs");
    // A = LHS
    a_d0 = buffers.lhs_shape[0];
    a_d1 = buffers.lhs_shape[1];
    a_data = const_cast<T*>(buffers.lhs_buffer.data());
    // B = RHS^T
    b_d0 = buffers.rhs_shape[0];
    b_d1 = buffers.rhs_shape[1];
    b_data = temp1_buffer = new T[b_d0 * b_d1];
    Transpose2D(b_d0, b_d1, buffers.rhs_buffer.data(), b_data);
    // R
    transpose_dst = true;
    r_d0 = buffers.dst_shape[0];
    r_d1 = buffers.dst_shape[1];
    r_data = temp2_buffer = new T[r_d0 * r_d1];
  }

  ruy::Matrix<T> a_matrix;
  ruy::MakeSimpleLayout(a_d0, a_d1, ruy::Order::kRowMajor, &a_matrix.layout);
  a_matrix.data.set(a_data);

  ruy::Matrix<T> b_matrix;
  ruy::MakeSimpleLayout(b_d0, b_d1, ruy::Order::kColMajor, &b_matrix.layout);
  b_matrix.data.set(b_data);

  ruy::Matrix<T> r_matrix;
  ruy::MakeSimpleLayout(r_d0, r_d1, ruy::Order::kColMajor, &r_matrix.layout);
  r_matrix.data.set(r_data);

  ruy::BasicSpec<ACC, T> spec;
  spec.bias = buffers.bias_buffer.data();

  if (buffers.multiplier_mantissa_buffer.size() == 1) {
    spec.multiplier_fixedpoint = buffers.multiplier_mantissa_buffer[0];
    spec.multiplier_exponent = buffers.multiplier_exponent_buffer[0];
  } else {
    spec.multiplier_fixedpoint_perchannel =
        buffers.multiplier_mantissa_buffer.data();
    spec.multiplier_exponent_perchannel =
        buffers.multiplier_exponent_buffer.data();
  }

  ruy::Mul<ruy::kAllPaths>(a_matrix, b_matrix, spec, &runtime_state->context,
                           &r_matrix);

  if (transpose_dst) {
    IREE_TRACE_SCOPE0("MatMul#TransposeDst");
    // Dims reversed because it is written in col major and the transpose
    // treats the dims as row major.
    Transpose2D(r_d1, r_d0, r_data, buffers.dst_buffer.data());
  }
  delete[] temp1_buffer;
  delete[] temp2_buffer;

  return OkStatus();
}

}  // namespace kernels
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_INTERPRETER_BYTECODE_KERNELS_RUY_H_
