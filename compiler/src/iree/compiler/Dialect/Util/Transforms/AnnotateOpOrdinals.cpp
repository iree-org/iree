// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"

namespace mlir::iree_compiler::IREE::Util {

#define GEN_PASS_DEF_ANNOTATEOPORDINALSPASS
#include "iree/compiler/Dialect/Util/Transforms/Passes.h.inc"

namespace {

class AnnotateOpOrdinalsPass
    : public impl::AnnotateOpOrdinalsPassBase<AnnotateOpOrdinalsPass> {
public:
  void runOnOperation() override {
    auto *context = &getContext();
    auto attrName = StringAttr::get(context, "util.ordinal");
    auto indexType = IndexType::get(context);
    int64_t globalOrdinal = 0;
    getOperation().walk<WalkOrder::PreOrder>([&](Operation *op) {
      op->setAttr(attrName, IntegerAttr::get(indexType, globalOrdinal++));
    });
  }
};

} // namespace
} // namespace mlir::iree_compiler::IREE::Util
