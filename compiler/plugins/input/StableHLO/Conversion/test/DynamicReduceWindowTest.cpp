// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/StableHLO/Conversion/Passes.h"

#include <gtest/gtest.h>

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {
namespace {

TEST(DynamicReduceWindowTest, ScanThroughDeepShapeProducerChain) {
  MLIRContext context;
  context
      .loadDialect<arith::ArithDialect, func::FuncDialect,
                   tensor::TensorDialect, mlir::stablehlo::StablehloDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @prefix(%input: tensor<?xi32>) -> tensor<?xi32> {
      %axis = arith.constant 0 : index
      %init = stablehlo.constant dense<0> : tensor<i32>
      %zero = arith.constant 0 : i32
      %one = arith.constant 1 : i32
      %ones = stablehlo.constant dense<1> : tensor<1xi32>
      %dim = tensor.dim %input, %axis : tensor<?xi32>
      %size = arith.index_cast %dim : index to i32
      %low = arith.subi %size, %one : i32
      %window = tensor.from_elements %size : tensor<1xi32>
      %padding = tensor.from_elements %low, %zero : tensor<1x2xi32>
      %result = stablehlo.custom_call @stablehlo.dynamic_reduce_window(%input, %init, %window, %ones, %ones, %ones, %padding) {api_version = 2 : i32, called_computations = [@sum]} : (tensor<?xi32>, tensor<i32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1x2xi32>) -> tensor<?xi32>
      return %result : tensor<?xi32>
    }
    func.func private @sum(%lhs: tensor<i32>, %rhs: tensor<i32>) -> tensor<i32> {
      %sum = stablehlo.add %lhs, %rhs : tensor<i32>
      return %sum : tensor<i32>
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  {
    mlir::stablehlo::CustomCallOp call;
    module->walk([&call](mlir::stablehlo::CustomCallOp op) { call = op; });
    ASSERT_TRUE(call);

    // Shape producers form an SSA chain, independent of region nesting. A scan
    // proof must reach its constant leaf without one stack frame per producer.
    OpBuilder builder(call);
    Value ones = call.getOperand(3);
    for (int i = 0; i < 40000; ++i) {
      ones = mlir::stablehlo::ReshapeOp::create(builder, call.getLoc(),
                                                ones.getType(), ones);
    }
    call->setOperand(3, ones);
  }
  ASSERT_TRUE(succeeded(verify(*module)));

  PassManager manager(&context);
  manager.addPass(createLegalizeDynamicReduceWindow());
  ASSERT_TRUE(succeeded(manager.run(*module)));
  EXPECT_TRUE(succeeded(verify(*module)));
  int scans = 0;
  int remainingCalls = 0;
  module->walk([&scans](IREE::LinalgExt::ScanOp) { ++scans; });
  module->walk(
      [&remainingCalls](mlir::stablehlo::CustomCallOp) { ++remainingCalls; });
  EXPECT_EQ(scans, 1);
  EXPECT_EQ(remainingCalls, 0);
}

} // namespace
} // namespace mlir::iree_compiler::stablehlo
