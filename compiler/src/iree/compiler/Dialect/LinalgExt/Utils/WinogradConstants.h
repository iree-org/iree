// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LINALGEXT_UTILS_WINOGRAD_CONSTANTS_H_
#define IREE_COMPILER_DIALECT_LINALGEXT_UTILS_WINOGRAD_CONSTANTS_H_

namespace mlir::iree_compiler::IREE::LinalgExt::Winograd {

// This file contains the Winograd constant matrices for different
// output tile sizes

//===----------------------------------------------------------------------===//
// Output tile size = 6, Kernel size = 3
//===----------------------------------------------------------------------===//
// These constants were obtained from this paper:
//
// Liu, J. et al (2021) Optimizing Winograd-Based Convolution with Tensor Cores.
// https://dl.acm.org/doi/abs/10.1145/3472456.3472473
//

// clang-format off

const float BT_6x6_3x3[] = {
  1.0f,       0.0f,  -21.0f/4.0f,         0.0f,  21.0f/4.0f,        0.0f, -1.0f, 0.0f,
  0.0f,       1.0f,         1.0f,  -17.0f/4.0f, -17.0f/4.0f,        1.0f,  1.0f, 0.0f,
  0.0f,      -1.0f,         1.0f,   17.0f/4.0f, -17.0f/4.0f,       -1.0f,  1.0f, 0.0f,
  0.0f,  1.0f/2.0f,    1.0f/4.0f,   -5.0f/2.0f,  -5.0f/4.0f,        2.0f,  1.0f, 0.0f,
  0.0f, -1.0f/2.0f,    1.0f/4.0f,    5.0f/2.0f,  -5.0f/4.0f,       -2.0f,  1.0f, 0.0f,
  0.0f,       2.0f,         4.0f,   -5.0f/2.0f,       -5.0f,   1.0f/2.0f,  1.0f, 0.0f,
  0.0f,      -2.0f,         4.0f,    5.0f/2.0f,       -5.0f,  -1.0f/2.0f,  1.0f, 0.0f,
  0.0f,      -1.0f,         0.0f,   21.0f/4.0f,        0.0f, -21.0f/4.0f,  0.0f, 1.0f
};

const float B_6x6_3x3[] = {
         1.0f,        0.0f,        0.0f,       0.0f,       0.0f,       0.0f,       0.0f,        0.0f,
         0.0f,        1.0f,       -1.0f,  1.0f/2.0f, -1.0f/2.0f,       2.0f,      -2.0f,       -1.0f,
  -21.0f/4.0f,        1.0f,        1.0f,  1.0f/4.0f,  1.0f/4.0f,       4.0f,       4.0f,        0.0f,
         0.0f, -17.0f/4.0f,  17.0f/4.0f, -5.0f/2.0f,  5.0f/2.0f, -5.0f/2.0f,  5.0f/2.0f,  21.0f/4.0f,
   21.0f/4.0f, -17.0f/4.0f, -17.0f/4.0f, -5.0f/4.0f, -5.0f/4.0f,      -5.0f,      -5.0f,        0.0f,
         0.0f,        1.0f,       -1.0f,       2.0f,      -2.0f,  1.0f/2.0f, -1.0f/2.0f, -21.0f/4.0f,
        -1.0f,        1.0f,        1.0f,       1.0f,       1.0f,       1.0f,       1.0f,        0.0f,
         0.0f,        0.0f,        0.0f,       0.0f,       0.0f,       0.0f,       0.0f,        1.0f
};

const float G_6x6_3x3[] = {
         1.0f,         0.0f,        0.0f,
   -2.0f/9.0f,   -2.0f/9.0f,  -2.0f/9.0f,
   -2.0f/9.0f,    2.0f/9.0f,  -2.0f/9.0f,
   1.0f/90.0f,   1.0f/45.0f,  2.0f/45.0f,
   1.0f/90.0f,  -1.0f/45.0f,  2.0f/45.0f,
  32.0f/45.0f,  16.0f/45.0f,  8.0f/45.0f,
  32.0f/45.0f, -16.0f/45.0f,  8.0f/45.0f,
         0.0f,         0.0f,        1.0f
};

const float AT_6x6_3x3[] = {
  1.0f,  1.0f,   1.0f,   1.0f,    1.0f,       1.0f,        1.0f,  0.0f,
  0.0f,  1.0f,  -1.0f,   2.0f,   -2.0f,  1.0f/2.0f,  -1.0f/2.0f,  0.0f,
  0.0f,  1.0f,   1.0f,   4.0f,    4.0f,  1.0f/4.0f,   1.0f/4.0f,  0.0f,
  0.0f,  1.0f,  -1.0f,   8.0f,   -8.0f,  1.0f/8.0f,  -1.0f/8.0f,  0.0f,
  0.0f,  1.0f,   1.0f,  16.0f,   16.0f, 1.0f/16.0f,  1.0f/16.0f,  0.0f,
  0.0f,  1.0f,  -1.0f,  32.0f,  -32.0f, 1.0f/32.0f, -1.0f/32.0f,  1.0f
};

const float A_6x6_3x3[] = {
  1.0f,       0.0f,      0.0f,       0.0f,       0.0f,        0.0f,
  1.0f,       1.0f,      1.0f,       1.0f,       1.0f,        1.0f,
  1.0f,      -1.0f,      1.0f,      -1.0f,       1.0f,       -1.0f,
  1.0f,       2.0f,      4.0f,       8.0f,      16.0f,       32.0f,
  1.0f,      -2.0f,      4.0f,      -8.0f,      16.0f,      -32.0f,
  1.0f,  1.0f/2.0f, 1.0f/4.0f,  1.0f/8.0f, 1.0f/16.0f,  1.0f/32.0f,
  1.0f, -1.0f/2.0f, 1.0f/4.0f, -1.0f/8.0f, 1.0f/16.0f, -1.0f/32.0f,
  0.0f,       0.0f,      0.0f,       0.0f,       0.0f,        1.0f
};

// clang-format on

} // namespace mlir::iree_compiler::IREE::LinalgExt::Winograd

#endif // IREE_COMPILER_DIALECT_LINALGEXT_UTILS_WINOGRAD_CONSTANTS_H_
