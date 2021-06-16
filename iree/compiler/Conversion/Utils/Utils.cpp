// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Conversion/Utils/Utils.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace iree_compiler {

bool isEntryPoint(FuncOp func) { return func.isPublic(); }

unsigned getNumOuterParallelLoops(linalg::LinalgOp op) {
  return op.iterator_types()
      .getValue()
      .take_while([](Attribute attr) -> bool {
        return linalg::isParallelIteratorType(attr);
      })
      .size();
}

IREE::HAL::ExecutableEntryPointOp getEntryPoint(FuncOp funcOp) {
  auto targetOp =
      funcOp.getOperation()->getParentOfType<IREE::HAL::ExecutableTargetOp>();
  for (auto op : targetOp.getOps<IREE::HAL::ExecutableEntryPointOp>()) {
    if (op.sym_name() == funcOp.getName()) {
      return op;
    }
  }
  return nullptr;
}

llvm::StringMap<IREE::HAL::ExecutableEntryPointOp> getAllEntryPoints(
    ModuleOp module) {
  auto targetOp =
      module.getOperation()->getParentOfType<IREE::HAL::ExecutableTargetOp>();
  llvm::StringMap<IREE::HAL::ExecutableEntryPointOp> entryPointOps;
  for (auto op : targetOp.getOps<IREE::HAL::ExecutableEntryPointOp>()) {
    entryPointOps[op.sym_name()] = op;
  }
  return entryPointOps;
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
    if (auto subTensorOp = view.getDefiningOp<SubTensorOp>()) {
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

}  // namespace iree_compiler
}  // namespace mlir
