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

bool isVecmatEncodingUser(EncodingUser user) {
  return user == EncodingUser::VECMAT;
}

bool isMatvecEncodingUser(EncodingUser user) {
  return user == EncodingUser::MATVEC;
}

bool isBatchMatvecEncodingUser(EncodingUser user) {
  return user == EncodingUser::BATCH_MATVEC;
}

bool isVectorEncoding(EncodingAttr encoding) {
  return (isVecmatVector(encoding) || isMatvecVector(encoding) ||
          isBatchMatvecVector(encoding));
}

bool isVecmatVector(EncodingAttr encoding) {
  if (!encoding)
    return false;
  auto user = encoding.getUser().getValue();
  auto role = encoding.getRole().getValue();
  if (user == EncodingUser::VECMAT &&
      (role == EncodingRole::LHS || role == EncodingRole::RESULT)) {
    return true;
  }
  return false;
}

bool isMatvecVector(EncodingAttr encoding) {
  if (!encoding)
    return false;
  auto user = encoding.getUser().getValue();
  auto role = encoding.getRole().getValue();
  if (user == EncodingUser::MATVEC &&
      (role == EncodingRole::RHS || role == EncodingRole::RESULT)) {
    return true;
  }
  return false;
}

bool isBatchMatvecVector(EncodingAttr encoding) {
  if (!encoding)
    return false;
  auto user = encoding.getUser().getValue();
  auto role = encoding.getRole().getValue();
  if (user == EncodingUser::BATCH_MATVEC &&
      (role == EncodingRole::RHS || role == EncodingRole::RESULT)) {
    return true;
  }
  return false;
}

int64_t getExpandedDimIndex(EncodingAttr encoding) {
  if (isVecmatVector(encoding))
    return 0;
  if (isMatvecVector(encoding))
    return 1;
  if (isBatchMatvecVector(encoding))
    return 2;
  return -1;
}

SmallVector<ReassociationIndices>
getReassociationMapsForVectors(EncodingAttr encoding) {
  SmallVector<ReassociationIndices> ri = {};
  if (isVecmatVector(encoding) || isMatvecVector(encoding))
    ri = {{0, 1}};
  else if (isBatchMatvecVector(encoding))
    ri = {{0}, {1, 2}};
  return ri;
}

RankedTensorType createNewTypeForVectors(RankedTensorType inputType,
                                         EncodingAttr encoding,
                                         bool expanding) {
  RankedTensorType newType = inputType;
  Type eType = inputType.getElementType();
  if (isVecmatVector(encoding)) {
    if (expanding)
      newType = RankedTensorType::get({1, inputType.getDimSize(0)}, eType);
    else
      newType = RankedTensorType::get({inputType.getDimSize(1)}, eType);
  } else if (isMatvecVector(encoding)) {
    if (expanding)
      newType = RankedTensorType::get({inputType.getDimSize(0), 1}, eType);
    else
      newType = RankedTensorType::get({inputType.getDimSize(0)}, eType);
  } else if (isBatchMatvecVector(encoding)) {
    if (expanding)
      newType = RankedTensorType::get(
          {inputType.getDimSize(0), inputType.getDimSize(1), 1}, eType);
    else
      newType = RankedTensorType::get(
          {inputType.getDimSize(0), inputType.getDimSize(1)}, eType);
  }
  return newType;
}

MaterializeEncodingInfo
chooseEncodingInfoForMatmul(EncodingUser user, EncodingRole role,
                            MatmulTileParams tileParams) {
  // Start dim of the MxK (LHS), KxN (RHS), or MxN (RESULT) 2D matrix.
  int64_t matmulDimBase =
      (isBatchMatmulEncodingUser(user) || isBatchMatvecEncodingUser(user)) ? 1
                                                                           : 0;

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
