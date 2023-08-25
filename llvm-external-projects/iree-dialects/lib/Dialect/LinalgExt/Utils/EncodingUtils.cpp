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

bool isBatchMatmulEncodingUser(EncodingUser user) {
  switch (user) {
  case EncodingUser::BATCH_MATMUL_F32F32F32:
  case EncodingUser::BATCH_MATMUL_F16F16F32:
  case EncodingUser::BATCH_MATMUL_F16F16F16:
  case EncodingUser::BATCH_MATMUL_BF16BF16F32:
  case EncodingUser::BATCH_MATMUL_BF16BF16BF16:
  case EncodingUser::BATCH_MATMUL_I8I8I32:
    return true;
  default:
    return false;
  }
}

MaterializeEncodingInfo
chooseEncodingInfoForMatmul(EncodingUser user, EncodingRole role,
                            MatmulTileParams tileParams) {
  // Start dim of the MxK (LHS), KxN (RHS), or MxN (RESULT) 2D matrix.
  int64_t matmulDimBase = isBatchMatmulEncodingUser(user) ? 1 : 0;

  MaterializeEncodingInfo encodingInfo;
  encodingInfo.innerDimsPos = {matmulDimBase, matmulDimBase + 1};
  switch (role) {
  case (EncodingRole::LHS): {
    encodingInfo.innerTileSizes = {tileParams.M, tileParams.K};
    break;
  }
  case (EncodingRole::RHS): {
    encodingInfo.innerTileSizes = {tileParams.N, tileParams.K};
    encodingInfo.innerDimsPos = {matmulDimBase + 1, matmulDimBase};
    encodingInfo.outerDimsPerm =
        llvm::to_vector(llvm::seq<int64_t>(0, matmulDimBase));
    encodingInfo.outerDimsPerm.push_back(matmulDimBase + 1);
    encodingInfo.outerDimsPerm.push_back(matmulDimBase);
    break;
  }
  case (EncodingRole::RESULT): {
    encodingInfo.innerTileSizes = {tileParams.M, tileParams.N};
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
