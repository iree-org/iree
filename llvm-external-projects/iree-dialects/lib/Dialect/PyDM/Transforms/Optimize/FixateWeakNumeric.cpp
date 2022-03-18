// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../PassDetail.h"
#include "iree-dialects/Dialect/PyDM/IR/PyDMOps.h"
#include "iree-dialects/Dialect/PyDM/Transforms/Passes.h"

using namespace mlir;
namespace PYDM = mlir::iree_compiler::IREE::PYDM;
using namespace PYDM;

namespace {

struct FixateWeakNumericPass
    : public FixateWeakNumericBase<FixateWeakNumericPass> {
  void runOnOperation() override {
    Operation *rootOp = getOperation();
    rootOp->walk([&](Operation *op) {
      convertOperation(op);
      return WalkResult::advance();
    });
  }

  void convertOperation(Operation *op) {
    // Process all regions/blocks to rewrite block arguments.
    for (auto &region : op->getRegions()) {
      for (auto &block : region) {
        for (BlockArgument blockArg : block.getArguments()) {
          convertValue(blockArg);
        }
      }
    }

    // And all results.
    for (Value result : op->getResults()) {
      convertValue(result);
    }

    // Special cases for operations.
    if (auto funcOp = llvm::dyn_cast<PYDM::FuncOp>(op)) {
      FunctionType existingFt = funcOp.getType();
      FunctionType newFt = convertFunctionType(existingFt);
      if (newFt != existingFt) {
        funcOp.setType(newFt);
      }
    }
  }

  void convertValue(Value value) {
    value.setType(convertType(value.getType()));
  }

  Type convertType(Type type) {
    // TODO: The specific types we promote to need to be configured by the
    // lowering options.
    if (auto integerType = type.dyn_cast<PYDM::IntegerType>()) {
      if (integerType.isWeak()) {
        return PYDM::IntegerType::get(type.getContext(), 32);
      }
    } else if (auto realType = type.dyn_cast<PYDM::RealType>()) {
      if (realType.isWeak()) {
        return PYDM::RealType::get(type.getContext(),
                                   mlir::Float32Type::get(type.getContext()));
      }
    } else if (auto objectType = type.dyn_cast<PYDM::ObjectType>()) {
      Type primitiveType = objectType.getPrimitiveType();
      if (primitiveType) {
        Type newPrimitiveType = convertType(primitiveType);
        if (newPrimitiveType != primitiveType) {
          return PYDM::ObjectType::get(
              type.getContext(), newPrimitiveType.cast<PYDM::PrimitiveType>());
        }
      }
    }

    return type;
  }

  FunctionType convertFunctionType(FunctionType ft) {
    SmallVector<Type> inputs(ft.getInputs().begin(), ft.getInputs().end());
    SmallVector<Type> results(ft.getResults().begin(), ft.getResults().end());
    bool modified = false;
    for (Type &type : inputs) {
      Type newType = convertType(type);
      if (type != newType) {
        type = newType;
        modified = true;
      }
    }
    for (Type &type : results) {
      Type newType = convertType(type);
      if (type != newType) {
        type = newType;
        modified = true;
      }
    }

    if (!modified)
      return ft;

    return FunctionType::get(ft.getContext(), inputs, results);
  }
};

} // namespace

std::unique_ptr<OperationPass<>> PYDM::createFixateWeakNumericPass() {
  return std::make_unique<FixateWeakNumericPass>();
}
