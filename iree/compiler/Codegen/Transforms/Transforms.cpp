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
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {

/// Sets the hal.interace.workgroup.size operation to the constant value passed
/// in as `workloadPerWorkgroup`. The number of entries in
/// `workloadPerWorkgroup` is at least as much as the dimensionality of the
/// workgroup. It is assumed that the inner-most loop is mapped to the fastest
/// varying dimension in flow.dispatch.workgroup_size.
class SetWorkgroupSizePattern
    : public OpRewritePattern<IREE::HAL::InterfaceWorkgroupSizeOp> {
 public:
  SetWorkgroupSizePattern(MLIRContext *context,
                          ArrayRef<int64_t> workloadPerWorkgroupRef,
                          PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit),
        workloadPerWorkgroup(llvm::to_vector<4>(
            workloadPerWorkgroupRef.size() > kNumMaxParallelDims
                ? workloadPerWorkgroupRef.take_front(kNumMaxParallelDims)
                : workloadPerWorkgroupRef)) {}

  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceWorkgroupSizeOp workgroupSizeOp,
      PatternRewriter &rewriter) const override {
    int64_t dim = workgroupSizeOp.dimension().getSExtValue();
    if (dim >= workloadPerWorkgroup.size()) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<ConstantIndexOp>(workgroupSizeOp,
                                                 workloadPerWorkgroup[dim]);
    return success();
  }

 private:
  SmallVector<int64_t, 4> workloadPerWorkgroup;
};
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
      entryPointOp.interfaceAttr(), entryPointOp.workgroup_sizeAttr(),
      entryPointOp.workgroup_local_memoryAttr(), 1);
  // Copy over all attributes
  for (auto attr : entryPointOp->getAttrs()) {
    if (attr.first != entryPointOp.sym_nameAttrName() &&
        attr.first != entryPointOp.ordinalAttrName() &&
        attr.first != entryPointOp.interfaceAttrName() &&
        attr.first != entryPointOp.workgroup_sizeAttrName() &&
        attr.first != entryPointOp.workgroup_local_memoryAttrName()) {
      clonedOp->setAttr(attr.first, attr.second);
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

LogicalResult materializeStaticLaunchInformation(
    FuncOp funcOp, ArrayRef<int64_t> workloadPerWorkgroup) {
  OwningRewritePatternList patterns(funcOp.getContext());
  patterns.insert<SetWorkgroupSizePattern>(funcOp.getContext(),
                                           workloadPerWorkgroup);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    return failure();
  }
  assert(workloadPerWorkgroup.size() <= kNumMaxParallelDims &&
         "workloadPerWorkgroup size greater than max num parallel dims");
  WorkgroupCountRegionBuilder regionBuilder =
      [&workloadPerWorkgroup](
          OpBuilder &b, Location loc,
          std::array<Value, 3> workload) -> std::array<Value, 3> {
    Value one = b.create<ConstantIndexOp>(loc, 1);
    std::array<Value, 3> returnValues = {one, one, one};
    for (auto ts : llvm::enumerate(workloadPerWorkgroup)) {
      returnValues[ts.index()] = linalg::applyMapToValues(
          b, loc,
          AffineMap::get(0, 1, b.getAffineSymbolExpr(0).ceilDiv(ts.value())),
          workload[ts.index()])[0];
    }
    return returnValues;
  };
  OpBuilder builder(funcOp.getContext());
  return defineWorkgroupCountRegion(builder, funcOp, regionBuilder);
}

/// Return a fused vector::ContractionOp which represents a patterns such as:
///
/// ```mlir
///    %c0 = vector.constant 0: ...
///    %c = vector.contract %a, %b, %c0: ...
///    %e = add %c, %d: ...
/// ```
///
/// by:
///
/// ```mlir
///    %e = vector.contract %a, %b, %d: ...
/// ```
///
/// Return null if the canonicalization does not apply.
// TODO: This should be a folding of Add into Contract in core but while they
// live in different dialects, it is not possible without unnatural
// dependencies.
vector::ContractionOp canonicalizeContractionAdd(Operation *op) {
  if (!isa<AddIOp, AddFOp>(op)) return nullptr;

  OpBuilder builder(op);
  auto canonicalize = [](OpBuilder &b, Value maybeContraction,
                         Value otherOperand) -> vector::ContractionOp {
    vector::ContractionOp contractionOp =
        dyn_cast_or_null<vector::ContractionOp>(
            maybeContraction.getDefiningOp());
    if (!contractionOp) return nullptr;
    if (auto maybeZero =
            dyn_cast_or_null<ConstantOp>(contractionOp.acc().getDefiningOp())) {
      if (maybeZero.value() == b.getZeroAttr(contractionOp.acc().getType())) {
        BlockAndValueMapping bvm;
        bvm.map(contractionOp.acc(), otherOperand);
        return cast<vector::ContractionOp>(b.clone(*contractionOp, bvm));
      }
    }
    return nullptr;
  };

  Value a = op->getOperand(0), b = op->getOperand(1);
  vector::ContractionOp contract = canonicalize(builder, a, b);
  contract = contract ? contract : canonicalize(builder, b, a);
  return contract;
}

}  // namespace iree_compiler
}  // namespace mlir
