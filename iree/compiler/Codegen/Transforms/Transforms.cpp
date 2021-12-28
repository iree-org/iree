// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- Transforms.cpp - Transformations common to all backends ------------===//
//
// Implements transformations that are common to all backends.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Transforms/Transforms.h"

#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {

}  // namespace

LogicalResult defineWorkgroupCountRegion(
    OpBuilder &builder, FuncOp funcOp,
    WorkgroupCountRegionBuilder regionBuilder) {
  IREE::HAL::ExecutableEntryPointOp entryPointOp = getEntryPoint(funcOp);
  if (!entryPointOp) {
    return funcOp.emitOpError("unable to find corresponding entry point op");
  }
  Location loc = entryPointOp.getLoc();

  OpBuilder::InsertionGuard guard(builder);
  // Create the cloned operation but with a single region.
  builder.setInsertionPoint(entryPointOp);

  auto clonedOp = builder.create<IREE::HAL::ExecutableEntryPointOp>(
      loc, entryPointOp.sym_nameAttr(), entryPointOp.ordinalAttr(),
      entryPointOp.layoutAttr(), entryPointOp.workgroup_sizeAttr(),
      entryPointOp.workgroup_local_memoryAttr(), 1);
  // Copy over all attributes
  for (auto attr : entryPointOp->getAttrs()) {
    if (attr.getName() != entryPointOp.sym_nameAttrName() &&
        attr.getName() != entryPointOp.ordinalAttrName() &&
        attr.getName() != entryPointOp.layoutAttr() &&
        attr.getName() != entryPointOp.workgroup_sizeAttrName() &&
        attr.getName() != entryPointOp.workgroup_local_memoryAttrName()) {
      clonedOp->setAttr(attr.getName(), attr.getValue());
    }
  }
  Region *region = clonedOp.getBody();
  Block *entryBlock = builder.createBlock(region);
  // Add 3 index arguments for the workload.
  auto indexType = builder.getIndexType();
  std::array<Value, 3> workload = {entryBlock->addArgument(indexType),
                                   entryBlock->addArgument(indexType),
                                   entryBlock->addArgument(indexType)};
  std::array<Value, 3> workgroupCount = regionBuilder(builder, loc, workload);
  builder.create<IREE::HAL::ReturnOp>(loc, workgroupCount);
  entryPointOp.erase();
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
