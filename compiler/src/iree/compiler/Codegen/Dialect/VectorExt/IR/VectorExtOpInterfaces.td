// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_VECTOREXT_OP_INTERFACES
#define IREE_DIALECT_VECTOREXT_OP_INTERFACES

include "mlir/IR/OpBase.td"

def IndexedVectorOpInterface : OpInterface<"IndexedVectorOpInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::VectorExt";

  let description = [{
    Interface for ops that access a shaped base (memref or tensor) using an
    affine indexing map whose symbols are bound to index vectors. This covers
    the common structure shared by `transfer_gather` (read) and
    `transfer_scatter` (write).

    An implementing op has:
    - A **base** shaped operand (source for reads, destination for writes).
    - A **vector** value (the result for reads, an operand for writes).
    - Zero or more **index vectors** that provide gathered/scattered indices.
    - An **indexing_maps** attribute describing how the vector iteration space
      maps to the base dimensions and to each index vector.
    - An optional **mask** vector.

    The `indexing_maps` attribute is an array of affine maps. Every map has
    `numDims = vector rank` and `numSymbols = number of index vecs`:

    - Map 0 (base map): `(vector_dims)[symbols] -> (base_dims)`.
      A dim expr means the base dimension is contiguous (iterated in lockstep
      with the vector dimension). A symbol expr means the base dimension is
      gathered/scattered (looked up via the corresponding index vector).
      A constant 0 means the base dimension is broadcast.
    - Maps 1..N (index vec maps): `(vector_dims)[symbols] -> (index_vec_dims)`.
      Describes how each index vector is indexed from the vector iteration
      space. Only dim exprs are allowed.
    - Optional last map (mask map): present only when a mask operand is
      provided. Only dim exprs are allowed.
  }];

  let methods = [
    InterfaceMethod<
      /*description=*/"Get the base shaped operand (source for gather, dest for scatter).",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getBase"
    >,
    InterfaceMethod<
      /*description=*/"Get the index vector operands.",
      /*retTy=*/"::mlir::Operation::operand_range",
      /*methodName=*/"getIndexVecs"
    >,
    InterfaceMethod<
      /*description=*/"Get the optional mask operand.",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getMask"
    >,
    InterfaceMethod<
      /*description=*/"Get the indexing maps attribute.",
      /*retTy=*/"::mlir::ArrayAttr",
      /*methodName=*/"getIndexingMaps"
    >,
    InterfaceMethod<
      /*description=*/"Get the vector value (result for gather, operand for scatter).",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getVector"
    >,
  ];

  let verify = [{
    return detail::verifyIndexedVectorOpInterface($_op);
  }];

  let extraSharedClassDeclaration = [{
    /// Return the vector type.
    VectorType getVectorType() {
      return cast<VectorType>($_op.getVector().getType());
    }

    /// Return indexing maps as a vector of AffineMap.
    SmallVector<AffineMap> getIndexingMapsArray() {
      return llvm::to_vector(
          $_op.getIndexingMaps().template getAsValueRange<AffineMapAttr>());
    }

    /// Return the indexing map for the base operand (first map).
    AffineMap getBaseIndexingMap() {
      return cast<AffineMapAttr>($_op.getIndexingMaps()[0]).getValue();
    }

    /// Return the indexing maps for each index vector.
    SmallVector<AffineMap> getIndexVecIndexingMaps() {
      SmallVector<AffineMap> maps = $_op.getIndexingMapsArray();
      return SmallVector<AffineMap>(
          maps.begin() + 1, maps.begin() + 1 + $_op.getIndexVecs().size());
    }

    /// Return the mask indexing map, if a mask is present.
    std::optional<AffineMap> getMaskIndexingMap() {
      if (!$_op.getMask()) {
        return std::nullopt;
      }
      ArrayAttr maps = $_op.getIndexingMaps();
      return cast<AffineMapAttr>(maps[maps.size() - 1]).getValue();
    }

    /// Invert the base indexing map to get a permutation map suitable for
    /// vector.transfer_read/write. The base map is
    /// (vector_dims)[symbols] -> (base_dims). This returns
    /// (base_dims) -> (vector_dims), where non-dim results (gathered/scattered
    /// symbols, broadcast constants) become constant 0.
    /// Note: conceptually the same as inverseAndBroadcastProjectedPermutation,
    /// but that utility asserts symbol-free maps. Our base map has symbols for
    /// gathered/scattered dims, which we treat identically to broadcast (map to
    /// constant 0 in the inverse).
    AffineMap getBasePermutationMap() {
      AffineMap baseMap = $_op.getBaseIndexingMap();
      MLIRContext *ctx = $_op->getContext();
      SmallVector<AffineExpr> exprs(baseMap.getNumDims(),
                                    getAffineConstantExpr(0, ctx));
      for (auto [i, expr] : llvm::enumerate(baseMap.getResults())) {
        if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
          exprs[dimExpr.getPosition()] = getAffineDimExpr(i, ctx);
        }
      }
      return AffineMap::get(baseMap.getNumResults(), /*symbolCount=*/0,
                            exprs, ctx);
    }

    /// Return true if the base operand has tensor semantics.
    bool hasTensorSemantics() {
      return isa<RankedTensorType>($_op.getBase().getType());
    }
  }];
}

#endif // IREE_DIALECT_VECTOREXT_OP_INTERFACES
