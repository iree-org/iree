// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/Utils/EncodingUtils.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace LinalgExt {

bool isMatmulEncodingUser(EncodingUser user) {
  return user == EncodingUser::MATMUL;
}

bool isBatchMatmulEncodingUser(EncodingUser user) {
  return user == EncodingUser::BATCH_MATMUL;
}

MaterializeEncodingInfo getEncodingInfoForMatmul(EncodingUser user,
                                                 EncodingRole role,
                                                 TileMxNxK tileMxNxK) {
  // Start dim of the MxK (LHS), KxN (RHS), or MxN (RESULT) 2D matrix.
  int64_t matmulDimBase = isBatchMatmulEncodingUser(user) ? 1 : 0;

  MaterializeEncodingInfo encodingInfo;
  encodingInfo.innerDimsPos = {matmulDimBase, matmulDimBase + 1};
  switch (role) {
  case (EncodingRole::LHS): {
    encodingInfo.innerTileSizes = {tileMxNxK.M, tileMxNxK.K};
    break;
  }
  case (EncodingRole::RHS): {
    encodingInfo.innerTileSizes = {tileMxNxK.N, tileMxNxK.K};
    encodingInfo.innerDimsPos = {matmulDimBase + 1, matmulDimBase};
    encodingInfo.outerDimsPerm =
        llvm::to_vector(llvm::seq<int64_t>(0, matmulDimBase));
    encodingInfo.outerDimsPerm.push_back(matmulDimBase + 1);
    encodingInfo.outerDimsPerm.push_back(matmulDimBase);
    break;
  }
  case (EncodingRole::RESULT): {
    encodingInfo.innerTileSizes = {tileMxNxK.M, tileMxNxK.N};
    break;
  }
  default: {
    assert(false);
    return {};
  }
  }
  return encodingInfo;
}

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
