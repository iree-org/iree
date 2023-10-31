// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>

namespace iree {
namespace pjrt {

// Compute the braodcasting equivalents for an output shape and striding
// behavior. This cannot guarantee a unique solution but will minimize
// moving dimensions when possible.
void computeBroadcastArgs(int64_t ndims, int64_t element_size,
                          const int64_t* strides, const int64_t* output_shape,
                          int64_t* input_shape, int64_t* perms);

}  // namespace pjrt
}  // namespace iree
