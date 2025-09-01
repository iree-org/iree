// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Dialect/Shard/IR/ShardOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"

namespace mlir::shard {
class ShardOp;
using ShardAxis = int16_t;
} // namespace mlir::shard

namespace mlir::iree_compiler::IREE::Flow {

// Returns a `!flow.channel` value from |builder| that represents the given
// grid axes. If no grid axes are provided then all will be used.
using LookupShardChannelFn = std::function<Value(
    Location loc, shard::GridOp gridOp,
    std::optional<ArrayRef<shard::GridAxis>> gridAxes, OpBuilder &builder)>;

void populateShardToFlowCollectivesPatterns(
    RewritePatternSet &patterns, SymbolTableCollection &symbolTableCollection,
    LookupShardChannelFn lookupChannel);

} // namespace mlir::iree_compiler::IREE::Flow
