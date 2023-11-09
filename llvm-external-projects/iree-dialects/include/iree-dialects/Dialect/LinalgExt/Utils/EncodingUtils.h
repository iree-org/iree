// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_LINALGEXT_UTILS_ENCODING_UTILS_H_
#define IREE_DIALECTS_DIALECT_LINALGEXT_UTILS_ENCODING_UTILS_H_

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Utils/Utils.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace LinalgExt {

// Check if encoding user is one of matmul encodings.
bool isMatmulEncodingUser(EncodingUser user);

// Check if encoding user is one of batch matmul encodings.
bool isBatchMatmulEncodingUser(EncodingUser user);

struct TileMxNxK {
  int64_t M = 1;
  int64_t N = 1;
  int64_t K = 1;
};

MaterializeEncodingInfo getEncodingInfoForMatmul(EncodingUser user,
                                                 EncodingRole role,
                                                 TileMxNxK tileMxNxK);

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_DIALECTS_DIALECT_LINALGEXT_UTILS_ENCODING_UTILS_H_
