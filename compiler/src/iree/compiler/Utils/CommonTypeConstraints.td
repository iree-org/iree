// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_UTILS_COMMONTYPECONSTRAINTS
#define IREE_CODEGEN_UTILS_COMMONTYPECONSTRAINTS

include "mlir/IR/CommonTypeConstraints.td"

def AnyRankedTensorOrMemRef : AnyTypeOf<[AnyRankedTensor, AnyMemRef],
  "ranked tensor or memref of any type", "::mlir::ShapedType">;
def AnyRankedTensorVectorOrMemRef
    : AnyTypeOf<[AnyRankedTensor, AnyVectorOfAnyRank, AnyMemRef],
                "ranked tensor, vector, or memref of any type",
                "::mlir::ShapedType">;

class RankedTensorOrMemRefOf<list<Type> allowedTypes, list<int> ranks>
    : AnyTypeOf<[TensorRankOf<allowedTypes, ranks>,
        MemRefRankOf<allowedTypes, ranks>],
        "tensor or memref of "
          # !interleave(!foreach(t, allowedTypes, t.summary), "/")
          # " values and rank "
          # !interleave(ranks, "/"),
        "::mlir::ShapedType">;

class RankedTensorOrVectorOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes,
      Or<[IsVectorOfNonZeroRankTypePred, And<[IsTensorTypePred, HasRankPred]>]>,
  "ranked tensor or vector", "::mlir::ShapedType">;

def AnyRankedTensorOrVector : RankedTensorOrVectorOf<[AnyType]>;

class RankedTensorOrVectorOrMemRefOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes,
      Or<[IsVectorOfAnyRankTypePred,
          And<[IsTensorTypePred, HasRankPred]>,
          And<[IsMemRefTypePred, HasRankPred]>]>,
  "ranked tensor or vector or memref", "::mlir::ShapedType">;

#endif // IREE_CODEGEN_UTILS_COMMONTYPECONSTRAINTS
