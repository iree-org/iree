// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/Utils.h"

#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace iree_compiler {

bool isEntryPoint(FuncOp func) { return func.isPublic(); }

unsigned getNumOuterParallelLoops(linalg::LinalgOp op) {
  return op.iterator_types()
      .getValue()
      .take_while(
          [](Attribute attr) -> bool { return isParallelIterator(attr); })
      .size();
}

IREE::HAL::ExecutableEntryPointOp getEntryPoint(FuncOp funcOp) {
  auto variantOp = funcOp->getParentOfType<IREE::HAL::ExecutableVariantOp>();
  for (auto op : variantOp.getOps<IREE::HAL::ExecutableEntryPointOp>()) {
    if (op.sym_name() == funcOp.getName()) {
      return op;
    }
  }
  return nullptr;
}

llvm::StringMap<IREE::HAL::ExecutableEntryPointOp> getAllEntryPoints(
    ModuleOp module) {
  auto variantOp = module->getParentOfType<IREE::HAL::ExecutableVariantOp>();
  llvm::StringMap<IREE::HAL::ExecutableEntryPointOp> entryPointOps;
  for (auto op : variantOp.getOps<IREE::HAL::ExecutableEntryPointOp>()) {
    entryPointOps[op.sym_name()] = op;
  }
  return entryPointOps;
}

void setTranslationInfo(FuncOp entryPointFn,
                        IREE::HAL::DispatchLoweringPassPipeline passPipeline,
                        ArrayRef<int64_t> workgroupSize,
                        ArrayRef<int64_t> workloadPerWorkgroup) {
  auto entryPointOp = getEntryPoint(entryPointFn);
  auto translationInfo = buildTranslationInfo(
      passPipeline, workloadPerWorkgroup, entryPointFn.getContext());
  setTranslationInfo(entryPointOp, translationInfo, workgroupSize);
}

SmallVector<unsigned> getPartitionedLoops(Operation *op) {
  SmallVector<unsigned> partitionedLoops;
  if (auto mmt4dOp = dyn_cast<linalg::Mmt4DOp>(op)) {
    return {0, 1};
  }
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    size_t numOuterParallelLoops = getNumOuterParallelLoops(linalgOp);
    partitionedLoops =
        llvm::to_vector<4>(llvm::seq<unsigned>(0, numOuterParallelLoops));
    if (partitionedLoops.size() > kNumMaxParallelDims) {
      partitionedLoops.erase(
          partitionedLoops.begin(),
          std::next(partitionedLoops.begin(),
                    numOuterParallelLoops - kNumMaxParallelDims));
    }
    return partitionedLoops;
  }
  if (auto tilableOp = dyn_cast<linalg_ext::TiledOpInterface>(op)) {
    return tilableOp.getPartitionableLoops(kNumMaxParallelDims);
  }
  return {};
}

LogicalResult setOpConfigAndEntryPointFnTranslation(
    FuncOp entryPointFn, Operation *op, IREE::HAL::LoweringConfig config,
    IREE::HAL::DispatchLoweringPassPipeline passPipeline,
    ArrayRef<int64_t> workgroupSize) {
  auto partitionedLoops = getPartitionedLoops(op);
  SmallVector<int64_t, 3> workloadPerWorkgroup;
  auto tileSizes = getTileSizes(config, 0);
  if (!tileSizes.empty() && !partitionedLoops.empty()) {
    for (unsigned depth : partitionedLoops) {
      if (depth >= tileSizes.size()) {
        return op->emitOpError(
                   "illegal configuration for lowering op, expect first level "
                   "tile size to contain at least ")
               << partitionedLoops.back() << " elements";
      }
      if (tileSizes[depth] == 0) {
        return op->emitOpError("illegal to set tilesize of loop ")
               << depth
               << " to zero since it is set to be partitioned at the flow "
                  "level";
      }
      workloadPerWorkgroup.push_back(tileSizes[depth]);
    }
    if (!workloadPerWorkgroup.empty()) {
      workloadPerWorkgroup =
          llvm::to_vector<3>(llvm::reverse(workloadPerWorkgroup));
    }
  }
  auto entryPointOp = getEntryPoint(entryPointFn);
  if (!entryPointOp) {
    return entryPointFn.emitOpError(
        "unable to find entry point op for entry point function");
  }
  IREE::HAL::TranslationInfo translationInfo = buildTranslationInfo(
      passPipeline, workloadPerWorkgroup, entryPointOp->getContext());
  setTranslationInfo(entryPointOp, translationInfo, workgroupSize);
  return success();
}

/// Walk up the defs of the view, to get the untiled value. Either walks up
/// `ViewOpInterface` op-chains or the `subtensor` op-chains.
static Value getViewSource(Value view) {
  while (true) {
    Operation *definingOp = view.getDefiningOp();
    if (!definingOp) break;
    if (auto viewOp = view.getDefiningOp<ViewLikeOpInterface>()) {
      view = viewOp.getViewSource();
      continue;
    }
    if (auto subTensorOp = view.getDefiningOp<tensor::ExtractSliceOp>()) {
      view = subTensorOp.source();
      continue;
    }
    if (auto dispatchTensorLoadOp =
            view.getDefiningOp<IREE::Flow::DispatchTensorLoadOp>()) {
      view = dispatchTensorLoadOp.source();
      continue;
    }
    break;
  }
  return view;
}

Type getUntiledType(Value tiledView) {
  Value viewSource = getViewSource(tiledView);
  return viewSource.getType();
}

ArrayRef<int64_t> getUntiledShape(Value tiledView) {
  auto type = getUntiledType(tiledView);
  return TypeSwitch<Type, ArrayRef<int64_t>>(type)
      .Case<ShapedType, IREE::Flow::DispatchTensorType>(
          [&](auto shapedType) { return shapedType.getShape(); })
      .Default([&](Type type) { return ArrayRef<int64_t>{}; });
}

/// Returns the untiled shape of the output of a `LinalgOp`.
// TODO(ravishankarm): Using the result shape for vectorization should be
// avoided. Ideally the tile size is enough. But there is a phase ordering issue
// which prevents the tile size from being known at this point.
ArrayRef<int64_t> getUntiledResultShape(linalg::LinalgOp linalgOp,
                                        unsigned resultNum) {
  // Check the shape of the `outs` operand.
  auto outputShape = getUntiledShape(linalgOp.outputs()[resultNum]);
  if (llvm::none_of(outputShape, ShapedType::isDynamic)) return outputShape;

  // For Linalg ops with buffer semantics, there won't exist op results and
  // hence IR users. Also directly return.
  if (linalgOp.hasBufferSemantics()) return outputShape;

  // Try to use the result value and check if the untiled shape can be obtained
  // based on the uses.
  Value result = linalgOp->getResult(resultNum);
  for (Operation *user : result.getUsers()) {
    if (auto storeOp = dyn_cast<IREE::Flow::DispatchTensorStoreOp>(user)) {
      return storeOp.target()
          .getType()
          .cast<IREE::Flow::DispatchTensorType>()
          .getShape();
    }
  }
  return result.getType().cast<ShapedType>().getShape();
}

LogicalResult getFilteredOps(FuncOp funcOp, RootOpFilteringFn filteringFn,
                             SmallVectorImpl<Operation *> &filteredOps,
                             SmallVectorImpl<Operation *> &tiledLoops) {
  Region &region = funcOp.body();
  if (!llvm::hasSingleElement(region)) {
    return funcOp.emitError("unable dispatch function with multiple blocks");
  }
  Block *body = &region.front();
  auto forOps = body->getOps<scf::ForOp>();
  while (!forOps.empty()) {
    if (!llvm::hasSingleElement(forOps)) return failure();
    scf::ForOp forOp = *(forOps.begin());
    tiledLoops.push_back(forOp.getOperation());
    body = forOp.getBody();
    forOps = body->getOps<scf::ForOp>();
  }
  for (Operation &op : body->getOperations()) {
    if (filteringFn(&op)) {
      filteredOps.push_back(&op);
    }
  }
  return success();
}

LogicalResult getComputeOps(FuncOp funcOp,
                            SmallVectorImpl<Operation *> &computeOps,
                            SmallVectorImpl<Operation *> &tiledLoops) {
  if (failed(getFilteredOps(
          funcOp,
          [](Operation *op) {
            return isa<linalg::LinalgOp, linalg_ext::TiledOpInterface>(op);
          },
          computeOps, tiledLoops))) {
    return failure();
  }

  // Propagate markers to all ops. If one of the ops has a marker all ops in
  // this loop need to have marker since body of the loop maps to a workgroup.
  // TODO(ravishankarm): Temporary WAR till a better story w.r.t markers is
  // figured out.
  Optional<StringRef> marker = llvm::None;
  for (auto op : computeOps) {
    if (hasMarker(op)) {
      assert(!marker || marker.getValue() == getMarkerOrNull(op) &&
                            "expected all markers within op to be the same");
      marker = getMarkerOrNull(op);
    }
  }
  if (!marker.hasValue() && !tiledLoops.empty()) {
    marker = getWorkgroupMarker();
  }
  if (marker.hasValue()) {
    for (auto op : computeOps) {
      setMarker(op, marker.getValue());
    }
  }
  return success();
}

LogicalResult getLinalgOps(FuncOp funcOp,
                           SmallVectorImpl<linalg::LinalgOp> &linalgOps,
                           SmallVectorImpl<Operation *> &tiledLoops) {
  {
    SmallVector<Operation *> computeOps;
    if (failed(getFilteredOps(
            funcOp, [](Operation *op) { return isa<linalg::LinalgOp>(op); },
            computeOps, tiledLoops))) {
      return failure();
    }
    for (auto op : computeOps) {
      linalgOps.push_back(cast<linalg::LinalgOp>(op));
    }
  }

  // Propagate markers to all ops. If one of the ops has a marker all ops in
  // this loop need to have marker since body of the loop maps to a workgroup.
  // TODO(ravishankarm): Temporary WAR till a better story w.r.t markers is
  // figured out.
  Optional<StringRef> marker = llvm::None;
  for (auto op : linalgOps) {
    if (hasMarker(op)) {
      assert(!marker || marker.getValue() == getMarkerOrNull(op) &&
                            "expected all markers within op to be the same");
      marker = getMarkerOrNull(op);
    }
  }
  if (marker.hasValue()) {
    for (auto op : linalgOps) {
      setMarker(op, marker.getValue());
    }
  }
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
