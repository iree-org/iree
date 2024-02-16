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
  1,      0, -21./4.,        0,  21./4.,       0, -1, 0,
  0,      1,       1,  -17./4., -17./4.,       1,  1, 0,
  0,     -1,       1,   17./4., -17./4.,      -1,  1, 0,
  0,   1./2,   1./4.,   -5./2.,  -5./4.,       2,  1, 0,
  0,  -1./2,   1./4.,    5./2.,  -5./4.,      -2,  1, 0,
  0,      2,       4,   -5./2.,      -5,   1./2.,  1, 0,
  0,     -2,       4,    5./2.,      -5,  -1./2.,  1, 0,
  0,     -1,       0,   21./4.,       0, -21./4.,  0, 1
};

const float B_6x6_3x3[] = {
        1,       0,       0,      0,      0,      0,      0,       0,
        0,       1,      -1,   1./2,  -1./2,      2,     -2,      -1,
  -21./4.,       1,       1,  1./4.,  1./4.,      4,      4,       0,
        0, -17./4.,  17./4., -5./2.,  5./2., -5./2.,  5./2.,  21./4.,
   21./4., -17./4., -17./4., -5./4., -5./4.,     -5,     -5,       0,
        0,       1,      -1,      2,     -2,  1./2., -1./2., -21./4.,
       -1,       1,       1,      1,      1,      1,      1,       0,
        0,       0,       0,      0,      0,      0,      0,       1
};

const float G_6x6_3x3[] = {
       1,       0,      0,
  -2./9.,  -2./9., -2./9.,
  -2./9.,   2./9., -2./9.,
   1./90,   1./45,  2./45,
   1./90,  -1./45,  2./45,
  32./45,  16./45,  8./45,
  32./45, -16./45,  8./45,
       0,       0,      1
};

const float AT_6x6_3x3[] = {
  1,  1,   1,   1,    1,     1,      1,  0,
  0,  1,  -1,   2,   -2,  1./2,  -1./2,  0,
  0,  1,   1,   4,    4,  1./4,   1./4,  0,
  0,  1,  -1,   8,   -8,  1./8,  -1./8,  0,
  0,  1,   1,  16,   16, 1./16,  1./16,  0,
  0,  1,  -1,  32,  -32, 1./32, -1./32,  1
};

const float A_6x6_3x3[] = {
  1,     0,    0,     0,     0,      0,
  1,     1,    1,     1,     1,      1,
  1,    -1,    1,    -1,     1,     -1,
  1,     2,    4,     8,    16,     32,
  1,    -2,    4,    -8,    16,    -32,
  1,  1./2, 1./4,  1./8, 1./16,  1./32,
  1, -1./2, 1./4, -1./8, 1./16, -1./32,
  0,     0,    0,     0,     0,      1
};

// clang-format on

} // namespace mlir::iree_compiler::IREE::LinalgExt::Winograd
#endif // IREE_COMPILER_DIALECT_LINALGEXT_UTILS_WINOGRAD_CONSTANTS_H_
