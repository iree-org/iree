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

template <typename T, typename ACC>
Status MatMul::Execute(RuntimeState* runtime_state,
                       const Buffers<T, ACC>& buffers) {
  ruy::Matrix<T> lhs_matrix;
  ruy::MakeSimpleLayout(buffers.lhs_shape[0], buffers.lhs_shape[1],
                        ruy::Order::kRowMajor, &lhs_matrix.layout);
  lhs_matrix.data.set(buffers.lhs_buffer.data());

  ruy::Matrix<T> rhs_matrix;
  ruy::MakeSimpleLayout(buffers.rhs_shape[0], buffers.rhs_shape[1],
                        ruy::Order::kRowMajor, &rhs_matrix.layout);
  rhs_matrix.data.set(buffers.rhs_buffer.data());

  ruy::Matrix<T> dst_matrix;
  ruy::MakeSimpleLayout(buffers.dst_shape[0], buffers.dst_shape[1],
                        ruy::Order::kRowMajor, &dst_matrix.layout);
  dst_matrix.data.set(buffers.dst_buffer.data());

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

  ruy::Mul<ruy::kAllPaths>(lhs_matrix, rhs_matrix, spec,
                           &runtime_state->context, &dst_matrix);

  return OkStatus();
}

}  // namespace kernels
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_INTERPRETER_BYTECODE_KERNELS_RUY_H_
