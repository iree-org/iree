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

#ifndef IREE_MODULES_VMLA_OP_KERNELS_FFT_H_
#define IREE_MODULES_VMLA_OP_KERNELS_FFT_H_

#include "absl/types/span.h"
#include "iree/base/logging.h"
#include "iree/base/status.h"
#include "pffft.h"

namespace iree {
namespace hal {
namespace vmla {
namespace kernels {

using ShapeSpan = absl::Span<const int32_t>;

struct Fft {
  template <typename T>
  static Status Execute(absl::Span<const T> real_src_buffer,
                        absl::Span<const T> imag_src_buffer,
                        absl::Span<T> real_dst_buffer,
                        absl::Span<T> imag_dst_buffer, ShapeSpan real_src_shape,
                        ShapeSpan imag_src_shape) {
    PFFFT_Setup* fft_state =
        pffft_new_setup(real_src_shape.back(), PFFFT_COMPLEX);
    int element_count = real_src_buffer.size();
    std::vector<T> complex_input;
    complex_input.reserve(element_count * 2);

    // pffft requires the input to be an array of interleaved complex numbers
    for (int i = 0; i < element_count; i++) {
      complex_input[i * 2] = real_src_buffer[i];
      complex_input[i * 2 + 1] = imag_src_buffer[i];
    }

    std::vector<T> complex_output;
    complex_output.reserve(element_count * 2);

    pffft_transform_ordered(fft_state, &complex_input[0], &complex_output[0],
                            NULL, PFFFT_FORWARD);

    // Split the interleaved array back into a real and imag vectors.
    for (int i = 0; i < element_count; i++) {
      real_dst_buffer[i] = complex_output[i * 2];
      imag_dst_buffer[i] = complex_output[i * 2 + 1];
    }
    pffft_destroy_setup(fft_state);
    return OkStatus();
  }
};

struct Ifft {
  template <typename T>
  static Status Execute(absl::Span<const T> real_src_buffer,
                        absl::Span<const T> imag_src_buffer,
                        absl::Span<T> real_dst_buffer,
                        absl::Span<T> imag_dst_buffer, ShapeSpan real_src_shape,
                        ShapeSpan imag_src_shape) {
    PFFFT_Setup* fft_state =
        pffft_new_setup(real_src_shape.back(), PFFFT_COMPLEX);
    int element_count = real_src_buffer.size();
    std::vector<T> complex_input;
    complex_input.reserve(element_count * 2);

    // pffft requires the input to be an array of interleaved complex numbers
    for (int i = 0; i < element_count; i++) {
      complex_input[i * 2] = real_src_buffer[i];
      complex_input[i * 2 + 1] = imag_src_buffer[i];
    }

    std::vector<T> complex_output;
    complex_output.reserve(element_count * 2);

    pffft_transform_ordered(fft_state, &complex_input[0], &complex_output[0],
                            NULL, PFFFT_BACKWARD);

    // Split the interleaved array back into a real and imag vectors and scale
    // them.
    for (int i = 0; i < element_count; i++) {
      real_dst_buffer[i] = complex_output[i * 2] / element_count;
      imag_dst_buffer[i] = complex_output[i * 2 + 1] / element_count;
    }
    pffft_destroy_setup(fft_state);
    return OkStatus();
  }
};

struct Rfft {
  template <typename T>
  static Status Execute(absl::Span<const T> real_src_buffer,
                        absl::Span<T> real_dst_buffer,
                        absl::Span<T> imag_dst_buffer,
                        ShapeSpan real_src_shape) {
    PFFFT_Setup* fft_state = pffft_new_setup(real_src_shape.back(), PFFFT_REAL);
    int element_count = real_src_buffer.size() / 2 + 1;

    std::vector<T> complex_output;
    complex_output.resize(element_count * 4);

    pffft_transform_ordered(fft_state, &real_src_buffer[0], &complex_output[0],
                            NULL, PFFFT_FORWARD);

    // Split the interleaved array back into a real and imag vectors and scale
    // them.
    for (int i = 0; i < element_count; i++) {
      real_dst_buffer[i] = complex_output[i * 2];
      imag_dst_buffer[i] = complex_output[i * 2 + 1];
    }
    auto temp = real_dst_buffer[element_count - 1];
    real_dst_buffer[element_count - 1] = imag_dst_buffer[0];
    imag_dst_buffer[0] = temp;
    pffft_destroy_setup(fft_state);
    return OkStatus();
  }
};

struct Irfft {
  template <typename T>
  static Status Execute(absl::Span<const T> real_src_buffer,
                        absl::Span<const T> imag_src_buffer,
                        absl::Span<T> real_dst_buffer, ShapeSpan real_src_shape,
                        ShapeSpan imag_src_shape) {
    PFFFT_Setup* fft_state = pffft_new_setup(real_src_shape.back(), PFFFT_REAL);
    int element_count = real_src_buffer.size();
    std::vector<T> complex_input;
    complex_input.reserve(element_count * 2);

    // pffft requires the input to be an array of interleaved complex numbers
    for (int i = 0; i < element_count; i++) {
      complex_input[i * 2] = real_src_buffer[i];
      complex_input[i * 2 + 1] = imag_src_buffer[i];
    }

    pffft_transform_ordered(fft_state, &complex_input[0], &real_dst_buffer[0],
                            NULL, PFFFT_BACKWARD);

    pffft_destroy_setup(fft_state);
    return OkStatus();
  }
};

}  // namespace kernels
}  // namespace vmla
}  // namespace hal
}  // namespace iree

#endif  // IREE_MODULES_VMLA_OP_KERNELS_FFT_H_
