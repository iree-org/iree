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

#include "compiler/plugins/input/Torch/InputConversion/PassDetail.h"
#include "compiler/plugins/input/Torch/InputConversion/Passes.h"
#include "llvm/ADT/StringRef.h"

static const llvm::StringLiteral kStrictSymbolsMarker =
    "torch.assume_strict_symbolic_shapes";

namespace mlir::iree_compiler::TorchInput {

namespace {
struct SetStrictSymbolicShapesPass
    : public SetStrictSymbolicShapesPassBase<SetStrictSymbolicShapesPass> {

  void runOnOperation() override {
    getOperation()->setAttr(kStrictSymbolsMarker, UnitAttr::get(&getContext()));
  }
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createSetStrictSymbolicShapesPass() {
  return std::make_unique<SetStrictSymbolicShapesPass>();
}

} // namespace mlir::iree_compiler::TorchInput
