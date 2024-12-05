// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_IR_GPULOWERINGCONFIGUTILS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_IR_GPULOWERINGCONFIGUTILS_H_

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"

namespace mlir::iree_compiler::IREE::GPU {

/// Helper to retrieve/set a target mma intrinsic.
MmaInterfaceAttr getMmaKind(LoweringConfigAttr config);
void setMmaKind(MLIRContext *context, SmallVectorImpl<NamedAttribute> &attrs,
                MmaInterfaceAttr kind);

// TODO: Merge subgroup counts functionality into subgroup tiling level
//       lowering, when we have it implemented.
/// Helper to retrieve/set a target subgroup M/N counts.
std::optional<int64_t> getSubgroupMCount(LoweringConfigAttr config);
std::optional<int64_t> getSubgroupNCount(LoweringConfigAttr config);
void setSubgroupMCount(MLIRContext *context,
                       SmallVectorImpl<NamedAttribute> &attrs,
                       int64_t subgroupMCount);
void setSubgroupNCount(MLIRContext *context,
                       SmallVectorImpl<NamedAttribute> &attrs,
                       int64_t subgroupNCount);

// Helper to retrieve/set distribution basis.
LogicalResult getBasis(IREE::GPU::LoweringConfigAttr config,
                       IREE::GPU::TilingLevel level,
                       SmallVector<int64_t> &basis,
                       SmallVector<int64_t> &mapping);
void setBasis(MLIRContext *context, SmallVector<NamedAttribute> &attrs,
              IREE::GPU::TilingLevel level, ArrayRef<int64_t> basis,
              ArrayRef<int64_t> mapping);

/// Helper to retrieve/set a list of operand indices to promote.
std::optional<SmallVector<int64_t>>
getPromotedOperandList(LoweringConfigAttr config);
void setPromotedOperandList(MLIRContext *context,
                            SmallVectorImpl<NamedAttribute> &attrs,
                            ArrayRef<int64_t> operands);

/// Helper to retrieve  list of operand to pad.
std::optional<SmallVector<int64_t>> getPaddingList(LoweringConfigAttr config);

} // namespace mlir::iree_compiler::IREE::GPU

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_IR_GPULOWERINGCONFIGUTILS_H_
