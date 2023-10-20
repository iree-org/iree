// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_LINALGEXT_UTILS_ENCODING_UTILS_H_
#define IREE_DIALECTS_DIALECT_LINALGEXT_UTILS_ENCODING_UTILS_H_

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace LinalgExt {

// Check if encoding user is one of matmul encodings.
bool isMatmulEncodingUser(EncodingUser user);

// Check if encoding user is one of batch matmul encodings.
bool isBatchMatmulEncodingUser(EncodingUser user);

// Check if encoding user is one of vecmat encodings.
bool isVecmatEncodingUser(EncodingUser user);

// Check if encoding user is one of matvec encodings.
bool isMatvecEncodingUser(EncodingUser user);

// Check if encoding user is one of batch matvec encodings.
bool isBatchMatvecEncodingUser(EncodingUser user);

// Check if encoding belongs to a vector in a matrix/vector operation.
bool isVectorEncoding(EncodingAttr encoding);

// Check if encoding user is a vector in a vecmat operation.
bool isVecmatVector(EncodingAttr encoding);

// Check if encoding user is a vector in a matvec operation.
bool isMatvecVector(EncodingAttr encoding);

// Check if encoding user is a vector in a batch_matvec operation.
bool isBatchMatvecVector(EncodingAttr encoding);

// Get the dimension that is being expanded when provided a vector/matrix
// operation encoding.
int64_t getExpandedDimIndex(EncodingAttr encoding);

// Get the reassociation maps for expanding/collapsing vectors in vector/matrix
// operations based on their encoding.
SmallVector<ReassociationIndices>
getReassociationMapsForVectors(EncodingAttr encoding);

// Based on the encoding, deduce the new type of a vector after
// expanding/collapsing it in a vector/matrix operation.
RankedTensorType createNewTypeForVectors(RankedTensorType inputType,
                                         EncodingAttr encoding, bool expanding);

struct MatmulTileParams {
  int64_t M = 1;
  int64_t K = 1;
  int64_t N = 1;
};

MaterializeEncodingInfo
chooseEncodingInfoForMatmul(EncodingUser user, EncodingRole role,
                            MatmulTileParams tileParams);

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_DIALECTS_DIALECT_LINALGEXT_UTILS_ENCODING_UTILS_H_
