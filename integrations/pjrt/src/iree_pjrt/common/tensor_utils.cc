// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_pjrt/common/tensor_utils.h"

#include <algorithm>
#include <vector>

namespace iree {
namespace pjrt {

void computeBroadcastArgs(int64_t ndims, int64_t element_size,
                          const int64_t* output_strides,
                          const int64_t* output_shape, int64_t* input_shape,
                          int64_t* perms) {
  // Find all dimensions that have non unary dimensions.
  int filtered = 0;
  for (int i = 0; i < ndims; i++) {
    if (output_strides[i] == 0 || output_shape[i] == 1) continue;
    perms[filtered] = i;
    ++filtered;
  }

  // Solve the order of these dimensions.
  std::stable_sort(perms, perms + filtered, [&](int64_t a, int64_t b) {
    return output_strides[a] > output_strides[b];
  });

  // Populate any unary dimensions to map to their dimension.
  std::vector<int64_t> reverse_perms;
  constexpr int64_t kempty = -1;
  reverse_perms.resize(ndims, kempty);
  for (int i = 0; i < ndims; ++i) {
    if (output_strides[i] == 0 || output_shape[i] == 1) {
      reverse_perms[i] = i;
    }
  }

  // Populate the reordered dimensions.
  int idx = 0;
  for (int i = 0; i < ndims; i++) {
    if (reverse_perms[i] == kempty) {
      reverse_perms[i] = perms[idx];
      idx++;
    }
  }

  for (int i = 0; i < ndims; i++) {
    perms[reverse_perms[i]] = i;

    int64_t dim = output_shape[reverse_perms[i]];
    int64_t stride = output_strides[reverse_perms[i]];
    input_shape[i] = stride == 0 ? 1 : dim;
  }
}

}  // namespace pjrt
}  // namespace iree
