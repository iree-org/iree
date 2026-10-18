// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/StableHLO/Conversion/TargetPreprocessing.h"
#include "compiler/plugins/input/StableHLO/Conversion/StableHLOCustomCallMetadata.h"
#include "compiler/plugins/input/StableHLO/Conversion/TargetRegistry.h"

#include <gtest/gtest.h>
#include <atomic>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir::iree_compiler::stablehlo {
namespace {

constexpr StringLiteral kCustomCallModule = R"mlir(
module {
  func.func @run() -> tensor<i32> {
    %result = stablehlo.custom_call @example.constant() {
      api_version = 4 : i32, backend_config = {configuration = "preserved"},
      has_side_effect = false
    } : () -> tensor<i32>
    return %result : tensor<i32>
  }
}
)mlir";

class TargetPreprocessingTest : public ::testing::Test {
protected:
  TargetPreprocessingTest() {
    context.loadDialect<func::FuncDialect, mlir::stablehlo::StablehloDialect>();
  }

  void checkCustomCall(bool serializeToVhlo) {
    auto module = parseSourceString<ModuleOp>(kCustomCallModule, &context);
    ASSERT_TRUE(module);
    if (serializeToVhlo) {
      PassManager serialization(&context);
      serialization.addPass(
          mlir::stablehlo::createStablehloLegalizeToVhloPass());
      ASSERT_TRUE(succeeded(serialization.run(*module)));
      bool foundVhlo = false;
      module->walk([&](Operation *op) {
        foundVhlo |= op->getName().getDialectNamespace() == "vhlo";
      });
      ASSERT_TRUE(foundVhlo);
    }

    auto calls = std::make_shared<std::atomic<unsigned>>(0);
    TargetRegistry registry;
    ASSERT_TRUE(succeeded(registry.addCustomCall(
        "example.constant",
        [calls](mlir::stablehlo::CustomCallOp op,
                PatternRewriter &rewriter) -> LogicalResult {
          EXPECT_EQ(op->getAttrOfType<DictionaryAttr>("backend_config")
                        .getAs<StringAttr>("configuration")
                        .getValue(),
                    "preserved");
          EXPECT_EQ(op->getAttrOfType<IntegerAttr>("api_version").getInt(), 4);
          EXPECT_FALSE(hasEmptyCustomCallMetadata(op));
          ++*calls;
          auto type = cast<RankedTensorType>(op.getResult(0).getType());
          auto constant = arith::ConstantOp::create(
              rewriter, op.getLoc(), type,
              DenseIntElementsAttr::get(type, APInt(32, 7)));
          rewriter.replaceOp(op, constant.getResult());
          return success();
        },
        [](DialectRegistry &dialects) {
          dialects.insert<arith::ArithDialect>();
        })));

    PassManager preprocessing(&context);
    buildTargetPreprocessingPassPipeline(preprocessing, registry);
    ASSERT_TRUE(succeeded(preprocessing.run(*module)));
    EXPECT_EQ(calls->load(), 1u);
    EXPECT_TRUE(succeeded(verify(*module)));
    module->walk([&](Operation *op) {
      EXPECT_NE(op->getName().getDialectNamespace(), "vhlo");
      EXPECT_FALSE(isa<mlir::stablehlo::CustomCallOp>(op));
    });
    auto func = module->lookupSymbol<func::FuncOp>("run");
    ASSERT_TRUE(func);
    auto result = cast<func::ReturnOp>(func.getBody().front().getTerminator())
                      .getOperands()
                      .front();
    auto constant = result.getDefiningOp<arith::ConstantOp>();
    ASSERT_TRUE(constant);
    auto value = cast<DenseIntElementsAttr>(constant.getValue());
    EXPECT_EQ(value.getSplatValue<APInt>().getSExtValue(), 7);
  }

  MLIRContext context;
};

TEST_F(TargetPreprocessingTest, HandlesStablehloBeforeInputConversion) {
  checkCustomCall(false);
}

TEST_F(TargetPreprocessingTest, NormalizesVhloBeforeTargetLookup) {
  checkCustomCall(true);
}

TEST_F(TargetPreprocessingTest, LeavesOtherInputDialectsUnchanged) {
  context.loadDialect<arith::ArithDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @run(%x: i32) -> i32 {
        %c = arith.constant 1 : i32
        %result = arith.addi %x, %c : i32
        return %result : i32
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto original = module->clone();
  TargetRegistry registry;
  PassManager preprocessing(&context);
  buildTargetPreprocessingPassPipeline(preprocessing, registry);
  ASSERT_TRUE(succeeded(preprocessing.run(*module)));
  std::string before, after;
  llvm::raw_string_ostream beforeStream(before), afterStream(after);
  original->print(beforeStream);
  module->print(afterStream);
  EXPECT_EQ(before, after);
  original->destroy();
}

} // namespace
} // namespace mlir::iree_compiler::stablehlo
