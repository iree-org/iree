// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_VECTOR_EXT_TRANSFORMS_TRANSFORMS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_VECTOR_EXT_TRANSFORMS_TRANSFORMS_H_

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::VectorExt {

LogicalResult vectorizeGatherLikeGenericToTransferGather(
    RewriterBase &rewriter, linalg::GenericOp linalgOp,
    ArrayRef<int64_t> vectorSizes = {}, ArrayRef<bool> scalableVecDims = {},
    bool vectorizeNDExtract = false);

/// Vectorizes iree_linalg_ext.gather to iree_vector_ext.transfer_gather.
/// Currently, this pattern only works when the index_depth and batch rank of
/// the gather is 1.
///
/// %gather = iree_linalg_ext.gather dimension_map=[0] ins(%source, %indices)
///                                                    outs(%output)
///
///  vectorizes to:
///
/// %indices_vec = vector.transfer_read %indices
/// %gather_vec = iree_vector_ext.gather %source[...][%indices_vec...]
/// %gather = vector.transfer_write %gather_vec, %output
LogicalResult
vectorizeLinalgExtGatherToTransferGather(RewriterBase &rewriter,
                                         IREE::LinalgExt::GatherOp gatherOp,
                                         ArrayRef<int64_t> vectorSizes = {});

/// Vectorizes iree_linalg_ext.arg_compare to iree_vector_ext.arg_compare.
///
/// This transformation wraps the tensor operation with vector.transfer_read
/// and vector.transfer_write operations, creating a vectorized version.
///
/// %result:2 = iree_linalg_ext.arg_compare
///   dimension(1)
///   ins(%input : tensor<4x128xf32>)
///   outs(%out_val, %out_idx : tensor<4xf32>, tensor<4xi32>) {
/// ^bb0(%a: f32, %b: f32):
///   %cmp = arith.cmpf ogt, %a, %b : f32
///   iree_linalg_ext.yield %cmp : i1
/// }
///
///  vectorizes to:
///
/// %input_vec = vector.transfer_read %input
/// %out_val_vec = vector.transfer_read %out_val
/// %out_idx_vec = vector.transfer_read %out_idx
/// %result_vec:2 = iree_vector_ext.arg_compare
///   dimension(1)
///   ins(%input_vec : vector<4x128xf32>)
///   inits(%out_val_vec, %out_idx_vec : vector<4xf32>, vector<4xi32>) {
/// ^bb0(%a: f32, %b: f32):
///   %cmp = arith.cmpf ogt, %a, %b : f32
///   iree_vector_ext.yield %cmp : i1
/// }
/// %write_val = vector.transfer_write %result_vec#0, %out_val
/// %write_idx = vector.transfer_write %result_vec#1, %out_idx
LogicalResult
vectorizeLinalgExtArgCompare(RewriterBase &rewriter,
                             IREE::LinalgExt::ArgCompareOp argCompareOp,
                             ArrayRef<int64_t> vectorSizes = {});

void populateVectorTransferGatherLoweringPatterns(RewritePatternSet &patterns);

void populateVectorMaskLoweringPatterns(RewritePatternSet &patterns);

}; // namespace mlir::iree_compiler::IREE::VectorExt

#endif // IREE_COMPILER_CODEGEN_DIALECT_VECTOR_EXT_TRANSFORMS_TRANSFORMS_H_
