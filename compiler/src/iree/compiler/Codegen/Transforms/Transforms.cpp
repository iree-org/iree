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

namespace {}  // namespace

FailureOr<IREE::HAL::ExecutableExportOp> defineWorkgroupCountRegion(
    OpBuilder &builder, IREE::HAL::ExecutableExportOp exportOp,
    WorkgroupCountRegionBuilder regionBuilder) {
  Location loc = exportOp.getLoc();

  OpBuilder::InsertionGuard guard(builder);
  // Create the cloned operation but with a single region.
  builder.setInsertionPoint(exportOp);

  auto clonedOp = builder.create<IREE::HAL::ExecutableExportOp>(
      loc, exportOp.sym_nameAttr(), exportOp.ordinalAttr(),
      exportOp.layoutAttr(), exportOp.workgroup_sizeAttr(),
      exportOp.workgroup_local_memoryAttr());
  // Copy over all attributes
  for (auto attr : exportOp->getAttrs()) {
    if (attr.getName() != exportOp.sym_nameAttrName() &&
        attr.getName() != exportOp.ordinalAttrName() &&
        attr.getName() != exportOp.layoutAttr() &&
        attr.getName() != exportOp.workgroup_sizeAttrName() &&
        attr.getName() != exportOp.workgroup_local_memoryAttrName()) {
      clonedOp->setAttr(attr.getName(), attr.getValue());
    }
  }
  Region &region = clonedOp.workgroup_count();
  Block *entryBlock = builder.createBlock(&region);
  // Add 3 index arguments for the workload.
  auto indexType = builder.getIndexType();
  auto device =
      entryBlock->addArgument(builder.getType<IREE::HAL::DeviceType>(), loc);
  // NOTE: this code currently assumes workloads are always defined as 3D.
  std::array<Value, 3> workload = {entryBlock->addArgument(indexType, loc),
                                   entryBlock->addArgument(indexType, loc),
                                   entryBlock->addArgument(indexType, loc)};
  std::array<Value, 3> workgroupCount =
      regionBuilder(builder, loc, device, workload);
  builder.create<IREE::HAL::ReturnOp>(loc, workgroupCount);
  exportOp.erase();
  return clonedOp;
}

}  // namespace iree_compiler
}  // namespace mlir
