// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- SetStrictSymbolicShapes.cpp - Pass to set strict symbolic shapes -=====//
//
// Adds an attribute to all functions in the module indicating all contained
// operations can be treated as if the symbolic shapes are strict, thereby
// eliminating the need for special dynamic size-1 broadcast handling.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "torch-iree/InputConversion/PassDetail.h"
#include "torch-iree/InputConversion/Passes.h"
// #include "mlir/IR/BuiltinAttributes.h"
// #include "mlir/Pass/Pass.h"

// TODO: This should be set and imported from upstream.
static const llvm::StringLiteral kStrictSymbolsMarker =
    "torch.assume_strict_symbolic_shapes";

namespace mlir {
namespace iree_compiler {
namespace TorchInput {

namespace {
struct SetStrictSymbolicShapesPass
    : public SetStrictSymbolicShapesPassBase<SetStrictSymbolicShapesPass> {

  void runOnOperation() override {
    getOperation()->setAttr(kStrictSymbolsMarker, UnitAttr::get(&getContext()));
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createSetStrictSymbolicShapesPass() {
  return std::make_unique<SetStrictSymbolicShapesPass>();
}

} // namespace TorchInput
} // namespace iree_compiler
} // namespace mlir
