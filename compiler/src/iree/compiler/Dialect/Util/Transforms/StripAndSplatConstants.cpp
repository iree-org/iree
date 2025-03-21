// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Util {

#define GEN_PASS_DEF_STRIPANDSPLATCONSTANTSPASS
#include "iree/compiler/Dialect/Util/Transforms/Passes.h.inc"

namespace {

class StripAndSplatConstantsPass
    : public impl::StripAndSplatConstantsPassBase<StripAndSplatConstantsPass> {
public:
  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Give each splatted value a module-unique byte value so that it's easier
    // to track back to where it came from in the final output.
    int replaceIndex = 1;
    auto getSplatAttr = [&](ShapedType type) {
      return IREE::Util::BytePatternAttr::get(moduleOp.getContext(), type,
                                              replaceIndex++);
    };

    for (auto globalOp : moduleOp.getOps<Util::GlobalOp>()) {
      auto initialValue = globalOp.getInitialValueAttr();
      if (!initialValue) {
        continue;
      }
      auto shapedType = dyn_cast<ShapedType>(initialValue.getType());
      if (!shapedType) {
        continue;
      }
      globalOp.setInitialValueAttr(getSplatAttr(shapedType));
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Util
