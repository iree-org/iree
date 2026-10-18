// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/StableHLO/Conversion/TargetRegistry.h"

#include <gtest/gtest.h>

#include "llvm/ADT/StringSet.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir::iree_compiler::stablehlo {
namespace {

TargetRegistry::CustomCallHandler replaceWithI32(int32_t value) {
  return [value](mlir::stablehlo::CustomCallOp op,
                 PatternRewriter &rewriter) -> LogicalResult {
    auto type = cast<RankedTensorType>(op.getResult(0).getType());
    auto attr = DenseIntElementsAttr::get(type, APInt(32, value));
    auto constant =
        arith::ConstantOp::create(rewriter, op.getLoc(), type, attr);
    rewriter.replaceOp(op, constant.getResult());
    return success();
  };
}

class TargetRegistryTest : public ::testing::Test {
protected:
  TargetRegistryTest() {
    context.loadDialect<func::FuncDialect, arith::ArithDialect,
                        mlir::stablehlo::StablehloDialect>();
  }

  OwningOpRef<ModuleOp> parseModule(StringRef source) {
    auto module = parseSourceString<ModuleOp>(source, &context);
    if (module) {
      EXPECT_TRUE(succeeded(verify(*module)));
    }
    return module;
  }

  bool runRewritePass(ModuleOp module, const TargetRegistry &registry) {
    PassManager manager(&context);
    manager.addPass(createRewriteRegisteredTargetsPass(registry));
    return succeeded(manager.run(module));
  }

  std::string print(ModuleOp module) {
    std::string result;
    llvm::raw_string_ostream stream(result);
    module.print(stream);
    return result;
  }

  MLIRContext context;
};

TEST_F(TargetRegistryTest, RejectsEmptyNamesAndNullHandlers) {
  TargetRegistry registry;
  auto compositeHandler = [](mlir::stablehlo::CompositeOp, PatternRewriter &) {
    return failure();
  };
  EXPECT_FALSE(succeeded(registry.addCustomCall("", replaceWithI32(1))));
  EXPECT_FALSE(succeeded(registry.addCustomCall("empty_handler", {})));
  EXPECT_FALSE(succeeded(registry.addComposite("", 0, compositeHandler)));
  EXPECT_FALSE(succeeded(registry.addComposite("empty_handler", 0, {})));
}

TEST_F(TargetRegistryTest, ExternalPatternsDeclineAliasesByDefault) {
  TargetRegistry registry;
  int calls = 0;
  ASSERT_TRUE(succeeded(registry.addCustomCall(
      "test.alias",
      [&calls](mlir::stablehlo::CustomCallOp op, PatternRewriter &rewriter) {
        ++calls;
        rewriter.replaceOp(op, op.getOperand(0));
        return success();
      })));
  auto module = parseModule(R"mlir(
    module {
      func.func @run(%buffer: memref<4xi32>) -> memref<4xi32> {
        %r = stablehlo.custom_call @test.alias(%buffer) {
          api_version = 4 : i32, backend_config = {},
          has_side_effect = false,
          output_operand_aliases = [#stablehlo.output_operand_alias<
            output_tuple_indices = [], operand_index = 0,
            operand_tuple_indices = []>]
        } : (memref<4xi32>) -> memref<4xi32>
        return %r : memref<4xi32>
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  RewritePatternSet patterns(&context);
  populateTargetRewritePatterns(patterns, registry);
  ASSERT_TRUE(succeeded(applyPatternsGreedily(*module, std::move(patterns))));
  EXPECT_EQ(calls, 0);
  EXPECT_NE(print(*module).find("test.alias"), std::string::npos);
}

TEST_F(TargetRegistryTest, PureHandlersDeclineEffectsAndTokens) {
  TargetRegistry registry;
  int calls = 0;
  ASSERT_TRUE(succeeded(registry.addCustomCall(
      "test.identity",
      [&calls](mlir::stablehlo::CustomCallOp op, PatternRewriter &rewriter) {
        ++calls;
        rewriter.replaceOp(op, op.getOperand(0));
        return success();
      })));
  auto module = parseModule(R"mlir(
    module {
      func.func @effect(%input: tensor<i32>) -> tensor<i32> {
        %result = stablehlo.custom_call @test.identity(%input) {
          has_side_effect = true
        } : (tensor<i32>) -> tensor<i32>
        return %result : tensor<i32>
      }
      func.func @token(%input: !stablehlo.token) -> !stablehlo.token {
        %result = stablehlo.custom_call @test.identity(%input) {
          has_side_effect = false
        } : (!stablehlo.token) -> !stablehlo.token
        return %result : !stablehlo.token
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  std::string before = print(*module);
  ASSERT_TRUE(runRewritePass(*module, registry));
  EXPECT_EQ(calls, 0);
  EXPECT_EQ(print(*module), before);
}

TEST_F(TargetRegistryTest, DuplicateCustomCallDoesNotReplaceHandler) {
  TargetRegistry registry;
  ASSERT_TRUE(succeeded(registry.addCustomCall("target", replaceWithI32(1))));
  const auto *first = registry.lookupCustomCall("target");
  ASSERT_NE(first, nullptr);
  EXPECT_FALSE(succeeded(registry.addCustomCall("target", replaceWithI32(2))));
  EXPECT_EQ(registry.lookupCustomCall("target"), first);

  auto module = parseModule(R"mlir(
    module {
      func.func @run() -> tensor<i32> {
        %0 = stablehlo.custom_call @target() {has_side_effect = false}
            : () -> tensor<i32>
        return %0 : tensor<i32>
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(runRewritePass(*module, registry));
  EXPECT_NE(print(*module).find("dense<1>"), std::string::npos);
}

TEST_F(TargetRegistryTest, CompositeVersionsAndCustomCallNamesAreIndependent) {
  TargetRegistry registry;
  auto compositeHandler = [](mlir::stablehlo::CompositeOp, PatternRewriter &) {
    return failure();
  };
  ASSERT_TRUE(succeeded(registry.addComposite("shared", 0, compositeHandler)));
  EXPECT_FALSE(succeeded(registry.addComposite("shared", 0, compositeHandler)));
  EXPECT_TRUE(succeeded(registry.addComposite("shared", 1, compositeHandler)));
  ASSERT_TRUE(succeeded(registry.addCustomCall("shared", replaceWithI32(3))));
  EXPECT_NE(registry.lookupComposite("shared", 0), nullptr);
  EXPECT_NE(registry.lookupComposite("shared", 1), nullptr);
  EXPECT_NE(registry.lookupCustomCall("shared"), nullptr);
}

TEST_F(TargetRegistryTest, OwnsTemporaryStringKeys) {
  TargetRegistry registry;
  {
    std::string key = "temporary_target";
    ASSERT_TRUE(succeeded(registry.addCustomCall(key, replaceWithI32(4))));
  }
  EXPECT_NE(registry.lookupCustomCall("temporary_target"), nullptr);
}

TEST_F(TargetRegistryTest, PassSnapshotsRegistryBeforeLaterRegistration) {
  TargetRegistry registry;
  PassManager manager(&context);
  manager.addPass(createRewriteRegisteredTargetsPass(registry));
  ASSERT_TRUE(
      succeeded(registry.addCustomCall("late_target", replaceWithI32(5))));

  auto module = parseModule(R"mlir(
    module {
      func.func @run() -> tensor<i32> {
        %0 = stablehlo.custom_call @late_target() {has_side_effect = false}
            : () -> tensor<i32>
        return %0 : tensor<i32>
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(manager.run(*module)));
  EXPECT_NE(print(*module).find("stablehlo.custom_call @late_target"),
            std::string::npos);
}

TEST_F(TargetRegistryTest, ClonedPassOutlivesRegistry) {
  auto clonedPipeline = [&]() -> std::unique_ptr<OpPassManager> {
    TargetRegistry registry;
    if (failed(registry.addCustomCall("owned_target", replaceWithI32(7)))) {
      return nullptr;
    }
    OpPassManager original(ModuleOp::getOperationName());
    original.addPass(createRewriteRegisteredTargetsPass(registry));
    return std::make_unique<OpPassManager>(original);
  }();
  ASSERT_NE(clonedPipeline, nullptr);
  auto module = parseModule(R"mlir(
    module {
      func.func @run() -> tensor<i32> {
        %0 = stablehlo.custom_call @owned_target() {has_side_effect = false}
            : () -> tensor<i32>
        return %0 : tensor<i32>
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  PassManager manager(&context);
  static_cast<OpPassManager &>(manager) = *clonedPipeline;
  ASSERT_TRUE(succeeded(manager.run(*module)));
  EXPECT_NE(print(*module).find("dense<7>"), std::string::npos);
}

TEST_F(TargetRegistryTest, HandlerCanUseDependentDialect) {
  DialectRegistry dialectRegistry;
  dialectRegistry
      .insert<func::FuncDialect, mlir::stablehlo::StablehloDialect>();
  MLIRContext context(dialectRegistry);
  TargetRegistry registry;
  auto handler = [](mlir::stablehlo::CustomCallOp op,
                    PatternRewriter &rewriter) -> LogicalResult {
    auto type = cast<RankedTensorType>(op.getResult(0).getType());
    auto attr = DenseIntElementsAttr::get(type, APInt(32, 9));
    auto constant =
        arith::ConstantOp::create(rewriter, op.getLoc(), type, attr);
    rewriter.replaceOp(op, constant.getResult());
    return success();
  };
  ASSERT_TRUE(succeeded(registry.addCustomCall(
      "arith_target", handler, [](DialectRegistry &dialects) {
        dialects.insert<arith::ArithDialect>();
      })));
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @run() -> tensor<i32> {
        %0 = stablehlo.custom_call @arith_target() {has_side_effect = false}
            : () -> tensor<i32>
        return %0 : tensor<i32>
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  PassManager manager(&context);
  manager.addPass(createRewriteRegisteredTargetsPass(registry));
  ASSERT_TRUE(succeeded(manager.run(*module)));
  EXPECT_NE(context.getLoadedDialect<arith::ArithDialect>(), nullptr);
  std::string printed;
  llvm::raw_string_ostream stream(printed);
  module->print(stream);
  EXPECT_NE(printed.find("arith.constant dense<9>"), std::string::npos);
}

TEST_F(TargetRegistryTest, CompositeDispatchFallsBackByVersionAndAttributes) {
  TargetRegistry registry;
  int handlerCalls = 0;
  auto handler = [&handlerCalls](mlir::stablehlo::CompositeOp op,
                                 PatternRewriter &rewriter) -> LogicalResult {
    EXPECT_EQ(op.getVersion(), 1);
    ++handlerCalls;
    auto mode = op.getCompositeAttributes().getAs<StringAttr>("mode");
    if (!mode || mode.getValue() != "identity") {
      return failure();
    }
    rewriter.replaceOp(op, op.getInputs());
    return success();
  };
  ASSERT_TRUE(succeeded(registry.addComposite("test.identity", 1, handler)));
  auto module = parseModule(R"mlir(
    module {
      func.func @run(%arg0: tensor<4xf32>)
          -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
        %exact = stablehlo.composite "test.identity" %arg0 {
          decomposition = @reference_identity,
          composite_attributes = {mode = "identity"},
          version = 1 : i32
        } : (tensor<4xf32>) -> tensor<4xf32>
        %unknown_version = stablehlo.composite "test.identity" %arg0 {
          decomposition = @reference_identity,
          composite_attributes = {mode = "identity"},
          version = 99 : i32
        } : (tensor<4xf32>) -> tensor<4xf32>
        %unsupported_mode = stablehlo.composite "test.identity" %arg0 {
          decomposition = @reference_identity,
          composite_attributes = {mode = "unsupported"},
          version = 1 : i32
        } : (tensor<4xf32>) -> tensor<4xf32>
        return %exact, %unknown_version, %unsupported_mode
            : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
      }
      func.func private @reference_identity(%arg0: tensor<4xf32>)
          -> tensor<4xf32> {
        return %arg0 : tensor<4xf32>
      }
    }
  )mlir");
  ASSERT_TRUE(module);

  ASSERT_TRUE(runRewritePass(*module, registry));
  EXPECT_GE(handlerCalls, 2);
  int remainingComposites = 0;
  module->walk([&](mlir::stablehlo::CompositeOp) { ++remainingComposites; });
  EXPECT_EQ(remainingComposites, 2);

  PassManager manager(&context);
  manager.addNestedPass<func::FuncOp>(
      mlir::stablehlo::createStablehloLegalizeCompositeToCallPass(
          mlir::stablehlo::StablehloLegalizeCompositeToCallPassOptions{
              /*exceptListOption=*/{}}));
  ASSERT_TRUE(succeeded(manager.run(*module)));
  remainingComposites = 0;
  module->walk([&](mlir::stablehlo::CompositeOp) { ++remainingComposites; });
  EXPECT_EQ(remainingComposites, 0);
  int referenceCalls = 0;
  module->walk([&](func::CallOp call) {
    if (call.getCallee() == "reference_identity") {
      ++referenceCalls;
    }
  });
  EXPECT_EQ(referenceCalls, 2);
}

// JAX passes its dimension variables, and the platform index of a
// multi-platform export, as leading composite operands that the decomposition
// marks with jax.global_constant; handlers see only the composite's own.
TEST_F(TargetRegistryTest, CompositeDispatchSkipsGlobalConstantOperands) {
  TargetRegistry registry;
  std::vector<size_t> inputCounts;
  auto handler = [&inputCounts](mlir::stablehlo::CompositeOp op,
                                PatternRewriter &rewriter) -> LogicalResult {
    inputCounts.push_back(op.getInputs().size());
    auto mode = op.getCompositeAttributes().getAs<StringAttr>("mode");
    if (!mode || mode.getValue() != "identity") {
      return failure();
    }
    rewriter.replaceOp(op, op.getInputs());
    return success();
  };
  ASSERT_TRUE(succeeded(registry.addComposite("test.identity", 1, handler)));
  auto module = parseModule(R"mlir(
    module {
      func.func @run(%s: tensor<i64>, %p: tensor<i32>, %x: tensor<?x7xf32>)
          -> (tensor<?x7xf32>, tensor<?x7xf32>, tensor<?x7xf32>) {
        %one = stablehlo.composite "test.identity" %s, %x {
          decomposition = @dims_identity,
          composite_attributes = {mode = "identity"},
          version = 1 : i32
        } : (tensor<i64>, tensor<?x7xf32>) -> tensor<?x7xf32>
        %two = stablehlo.composite "test.identity" %p, %s, %x {
          decomposition = @platform_dims_identity,
          composite_attributes = {mode = "identity"},
          version = 1 : i32
        } : (tensor<i32>, tensor<i64>, tensor<?x7xf32>) -> tensor<?x7xf32>
        %declined = stablehlo.composite "test.identity" %s, %x {
          decomposition = @dims_identity,
          composite_attributes = {mode = "unsupported"},
          version = 1 : i32
        } : (tensor<i64>, tensor<?x7xf32>) -> tensor<?x7xf32>
        return %one, %two, %declined
            : tensor<?x7xf32>, tensor<?x7xf32>, tensor<?x7xf32>
      }
      func.func private @dims_identity(
          %arg0: tensor<i64> {jax.global_constant = "s"},
          %arg1: tensor<?x7xf32>) -> tensor<?x7xf32> {
        return %arg1 : tensor<?x7xf32>
      }
      func.func private @platform_dims_identity(
          %arg0: tensor<i32> {jax.global_constant = "_platform_index"},
          %arg1: tensor<i64> {jax.global_constant = "s"},
          %arg2: tensor<?x7xf32>) -> tensor<?x7xf32> {
        return %arg2 : tensor<?x7xf32>
      }
    }
  )mlir");
  ASSERT_TRUE(module);

  ASSERT_TRUE(runRewritePass(*module, registry));
  ASSERT_FALSE(inputCounts.empty());
  for (size_t count : inputCounts) {
    EXPECT_EQ(count, 1u);
  }
  auto run = module->lookupSymbol<func::FuncOp>("run");
  auto returned = cast<func::ReturnOp>(run.getBody().front().getTerminator());
  EXPECT_EQ(returned.getOperand(0), run.getArgument(2));
  EXPECT_EQ(returned.getOperand(1), run.getArgument(2));
  // The declined composite keeps every operand for its decomposition.
  SmallVector<mlir::stablehlo::CompositeOp> remaining;
  module->walk(
      [&](mlir::stablehlo::CompositeOp op) { remaining.push_back(op); });
  ASSERT_EQ(remaining.size(), 1u);
  EXPECT_EQ(remaining.front().getInputs().size(), 2u);
  EXPECT_EQ(returned.getOperand(2), remaining.front().getResult(0));
}

} // namespace
} // namespace mlir::iree_compiler::stablehlo
