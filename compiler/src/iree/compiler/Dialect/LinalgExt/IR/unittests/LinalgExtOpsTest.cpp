// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

namespace mlir::iree_compiler::IREE::LinalgExt {
namespace {

using ::testing::ElementsAre;

class LinalgExtOpsTest : public ::testing::Test {
protected:
  LinalgExtOpsTest() {
    registry.insert<IREELinalgExtDialect, func::FuncDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  }

  MLIRContext *getContext() { return &context; }

private:
  MLIRContext context;
  DialectRegistry registry;
};

static CustomOp findCustomOp(ModuleOp module, StringRef functionName) {
  func::FuncOp function = module.lookupSymbol<func::FuncOp>(functionName);
  if (!function) {
    return {};
  }
  auto customOps = function.getOps<CustomOp>();
  return customOps.empty() ? CustomOp{} : *customOps.begin();
}

TEST_F(LinalgExtOpsTest, GetStaticLoopRanges) {
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
      R"mlir(
module {
  func.func @symbol_dimensions(
      %input : tensor<4x16xf32>, %output : tensor<4x16xf32>)
      -> tensor<4x16xf32> {
    %0 = iree_linalg_ext.custom_op {
        indexing_maps = [affine_map<(d0)[s0] -> (d0, s0)>,
                         affine_map<(d0)[s0] -> (d0, s0)>],
        iterator_types = [#iree_linalg_ext.iterator_type<parallel>]}
        ins(%input : tensor<4x16xf32>)
        outs(%output : tensor<4x16xf32>) {
      ^bb0(%input_slice : tensor<?x?xf32>,
           %output_slice : tensor<?x?xf32>):
        iree_linalg_ext.yield %output_slice : tensor<?x?xf32>
    } -> tensor<4x16xf32>
    return %0 : tensor<4x16xf32>
  }

  func.func @non_invertible(
      %input : tensor<7xf32>, %output : tensor<4xf32>) -> tensor<4xf32> {
    %0 = iree_linalg_ext.custom_op {
        indexing_maps = [affine_map<(d0, d1) -> (d0 + d1)>,
                         affine_map<(d0, d1) -> (d0)>],
        iterator_types = [#iree_linalg_ext.iterator_type<parallel>,
                          #iree_linalg_ext.iterator_type<reduction>]}
        ins(%input : tensor<7xf32>) outs(%output : tensor<4xf32>) {
      ^bb0(%input_slice : tensor<?xf32>, %output_slice : tensor<?xf32>):
        iree_linalg_ext.yield %output_slice : tensor<?xf32>
    } -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  func.func @dynamic_and_static(
      %input : tensor<?xf32>, %output : tensor<4xf32>) -> tensor<4xf32> {
    %0 = iree_linalg_ext.custom_op {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = [#iree_linalg_ext.iterator_type<parallel>]}
        ins(%input : tensor<?xf32>) outs(%output : tensor<4xf32>) {
      ^bb0(%input_slice : tensor<?xf32>, %output_slice : tensor<?xf32>):
        iree_linalg_ext.yield %output_slice : tensor<?xf32>
    } -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  func.func @conflicting_static(
      %input : tensor<4xf32>, %output : tensor<8xf32>) -> tensor<8xf32> {
    %0 = iree_linalg_ext.custom_op {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = [#iree_linalg_ext.iterator_type<parallel>]}
        ins(%input : tensor<4xf32>) outs(%output : tensor<8xf32>) {
      ^bb0(%input_slice : tensor<?xf32>, %output_slice : tensor<?xf32>):
        iree_linalg_ext.yield %output_slice : tensor<?xf32>
    } -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }

  func.func @empty_map(%input : tensor<4xf32>,
      %ignored : tensor<5x6xf32>, %output : tensor<4xf32>)
      -> tensor<4xf32> {
    %0 = iree_linalg_ext.custom_op {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<() -> ()>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = [#iree_linalg_ext.iterator_type<parallel>]}
        ins(%input, %ignored : tensor<4xf32>, tensor<5x6xf32>)
        outs(%output : tensor<4xf32>) {
      ^bb0(%input_slice : tensor<?xf32>,
           %ignored_slice : tensor<?x?xf32>,
           %output_slice : tensor<?xf32>):
        iree_linalg_ext.yield %output_slice : tensor<?xf32>
    } -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
)mlir",
      getContext());

  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  EXPECT_THAT(findCustomOp(*module, "symbol_dimensions").getStaticLoopRanges(),
              ElementsAre(4));
  EXPECT_THAT(findCustomOp(*module, "non_invertible").getStaticLoopRanges(),
              ElementsAre(4, ShapedType::kDynamic));
  EXPECT_THAT(findCustomOp(*module, "dynamic_and_static").getStaticLoopRanges(),
              ElementsAre(4));
  EXPECT_THAT(findCustomOp(*module, "conflicting_static").getStaticLoopRanges(),
              ElementsAre(ShapedType::kDynamic));
  EXPECT_THAT(findCustomOp(*module, "empty_map").getStaticLoopRanges(),
              ElementsAre(4));
}

} // namespace
} // namespace mlir::iree_compiler::IREE::LinalgExt
