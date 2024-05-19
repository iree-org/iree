// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"

namespace mlir::mesh {
class MeshOp;
using MeshAxis = int16_t;
} // namespace mlir::mesh

namespace mlir::iree_compiler::IREE::Flow {

// Returns a `!flow.channel` value from |builder| that represents the given
// mesh axes. If no mesh axes are provided then all will be used.
using LookupMeshChannelFn = std::function<Value(
    Location loc, mesh::MeshOp meshOp,
    std::optional<ArrayRef<mesh::MeshAxis>> meshAxes, OpBuilder &builder)>;

void populateMeshToFlowCollectivesPatterns(
    RewritePatternSet &patterns, SymbolTableCollection &symbolTableCollection,
    LookupMeshChannelFn lookupChannel);

} // namespace mlir::iree_compiler::IREE::Flow
