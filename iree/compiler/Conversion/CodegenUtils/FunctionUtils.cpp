// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace iree_compiler {

bool isEntryPoint(FuncOp func) { return func.isPublic(); }

FailureOr<FuncOp> getSingleEntryPointFunction(ModuleOp module) {
  auto entryPointFns = llvm::to_vector<1>(llvm::make_filter_range(
      module.getOps<FuncOp>(), [&](FuncOp op) { return isEntryPoint(op); }));
  if (!llvm::hasSingleElement(entryPointFns)) {
    module.emitError(
        "cannot handle modules with multiple entry point functions.");
    return {};
  }
  return entryPointFns[0];
}

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
