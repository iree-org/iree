// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::VectorExt;

namespace {

static const StringRef kIteratorMarker = "__test_iterator_layout__";
static const StringRef kFrozenIteratorMarker =
    "__test_frozen_iterator_layout__";

struct TestVectorExtIteratorPass
    : public PassWrapper<TestVectorExtIteratorPass, Pass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestVectorExtIteratorPass)
  TestVectorExtIteratorPass() = default;
  TestVectorExtIteratorPass(const TestVectorExtIteratorPass &other)
      : PassWrapper(other) {}
  StringRef getArgument() const final { return "test-vector-ext-iterators"; }
  StringRef getDescription() const final {
    return "Test VectorExt Iterator pass.";
  }
  bool canScheduleOn(RegisteredOperationName opName) const override {
    return true;
  }
  // Prints the layout so that LIT can test it for correctness.
  static void printFn(const LayoutIterator::State &state) {
    for (const auto &[dim, it] : state) {
      llvm::outs() << stringifyLayoutDimension(dim).str() + ":" +
                          std::to_string(*it) + ", ";
    }
    llvm::outs() << "\n";
  }
  void testIterator(Operation *op) {
    auto layout = dyn_cast_or_null<LayoutAttr>(op->getAttr(kIteratorMarker));
    DenseMap<LayoutDimension, int64_t> strides;
    LayoutIterator iterator(layout, strides);
    iterator.apply(printFn);
  }
  LayoutDimensionAttr createLayoutDimensionAttr(MLIRContext *ctx,
                                                LayoutDimension dim) {
    return LayoutDimensionAttr::get(ctx, dim);
  }
  LayoutIterator
  createFrozenIterator(MLIRContext *ctx,
                       DenseMap<LayoutDimension, int64_t> &strides) {
    SmallVector<LayoutDimensionAttr> labels{
        createLayoutDimensionAttr(ctx, LayoutDimension::VECTORZ),
        createLayoutDimensionAttr(ctx, LayoutDimension::VECTORX)};
    auto newLayout =
        LayoutAttr::get(ctx, {PerDimLayoutAttr::get(ctx, labels[0], {1}),
                              PerDimLayoutAttr::get(ctx, labels[1], {1})});
    return LayoutIterator(newLayout, strides);
  }
  void testFrozenIterator(Operation *op) {
    auto layout =
        dyn_cast_or_null<LayoutAttr>(op->getAttr(kFrozenIteratorMarker));
    DenseMap<LayoutDimension, int64_t> strides;
    LayoutIterator iterator(layout, strides);
    auto frozenIterator = createFrozenIterator(op->getContext(), strides);
    iterator.maybeFreezeAndConcatenate(frozenIterator);
    iterator.apply(printFn);
  }
  void runOnOperation() override {
    getOperation()->walk([&](Operation *op) {
      if (op->hasAttr(kIteratorMarker)) {
        return testIterator(op);
      }
      if (op->hasAttr(kFrozenIteratorMarker)) {
        return testFrozenIterator(op);
      }
    });
  }
};

} // namespace

namespace mlir::test_ext {
void registerVectorExtTestPasses() {
  PassRegistration<TestVectorExtIteratorPass>();
}
} // namespace mlir::test_ext
