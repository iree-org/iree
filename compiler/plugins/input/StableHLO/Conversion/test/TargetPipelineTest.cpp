// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/StableHLO/Conversion/Passes.h"
#include "compiler/plugins/input/StableHLO/Conversion/StableHLOCustomCallMetadata.h"
#include "compiler/plugins/input/StableHLO/Conversion/TargetPreprocessing.h"
#include "compiler/plugins/input/StableHLO/Conversion/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Pipelines/Pipelines.h"
#include "iree/compiler/PluginAPI/Client.h"

#include <gtest/gtest.h>
#include <atomic>

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir::iree_compiler::stablehlo {
namespace {
class TargetExtension final : public PipelineExtensions {
public:
  TargetExtension() : rewrites(std::make_shared<std::atomic<unsigned>>(0)) {
    auto count = rewrites;
    auto result = registry.addComposite(
        "test.identity", 1,
        [count](mlir::stablehlo::CompositeOp op,
                PatternRewriter &rewriter) -> LogicalResult {
          ++*count;
          rewriter.replaceOp(op, op.getInputs());
          return success();
        });
    EXPECT_TRUE(succeeded(result));
  }
  void extendInputConversionPreprocessingPassPipeline(
      OpPassManager &manager, InputDialectOptions::Type inputType) override {
    ++preprocessingCalls;
    buildTargetPreprocessingPassPipeline(manager, registry);
  }

  bool
  extendCustomInputConversionPassPipeline(OpPassManager &manager,
                                          std::string_view mnemonic) override {
    if (mnemonic != "stablehlo" && mnemonic != "vhlo") {
      return false;
    }
    ++conversionCalls;
    buildStableHLOInputConversionPassPipeline(manager, StableHloOptions{});
    return true;
  }

  void populateDetectedCustomInputConversionTypes(
      ModuleOp &module, llvm::StringSet<> &mnemonics) override {
    ++detectionCalls;
    module.walk([&](Operation *op) {
      if (op->getName().getDialectNamespace() == "stablehlo") {
        mnemonics.insert("stablehlo");
      }
    });
  }

  TargetRegistry registry;
  std::shared_ptr<std::atomic<unsigned>> rewrites;
  unsigned preprocessingCalls = 0;
  unsigned conversionCalls = 0;
  unsigned detectionCalls = 0;
};

// Runs the precompile pipeline from its start to input conversion, or to
// |end|.
static LogicalResult
runInputPipeline(ModuleOp module, TargetExtension &extension,
                 StringRef mnemonic = "stablehlo",
                 const IREE::HAL::TargetOptions &halOptions = {},
                 IREEVMPipelinePhase end = IREEVMPipelinePhase::Input) {
  IREEVMPipelineHooks hooks;
  hooks.pipelineExtensions = &extension;
  InputDialectOptions inputOptions;
  inputOptions.inputTypeMnemonic = mnemonic.str();
  IREE::HAL::TargetRegistry targets;
  PassManager manager(module.getContext());
  buildIREEPrecompileTransformPassPipeline(
      targets, {}, {}, inputOptions, {}, {}, {}, {}, {}, halOptions, hooks,
      manager, IREEVMPipelinePhase::Start, end);
  return manager.run(module);
}

class TargetPipelineTest : public ::testing::Test {
protected:
  void runPipeline(StringRef mnemonic, bool rawVhlo, bool includeFallback) {
    DialectRegistry dialects;
    func::registerInlinerExtension(dialects);
    MLIRContext context(dialects);
    context.loadDialect<func::FuncDialect, mlir::stablehlo::StablehloDialect>();
    std::string source = R"mlir(
      module {
        func.func @registered(%input: tensor<2xf32>) -> tensor<2xf32> {
          %result = stablehlo.composite "test.identity" %input {
            decomposition = @reference, version = 1 : i32
          } : (tensor<2xf32>) -> tensor<2xf32>
          return %result : tensor<2xf32>
        }
        func.func private @reference(%input: tensor<2xf32>) -> tensor<2xf32> {
          return %input : tensor<2xf32>
        }
      }
    )mlir";
    auto module = parseSourceString<ModuleOp>(source, &context);
    ASSERT_TRUE(module);
    if (includeFallback) {
      auto registered = module->lookupSymbol<func::FuncOp>("registered");
      auto fallback = registered.clone();
      fallback.setName("fallback");
      fallback.walk([](mlir::stablehlo::CompositeOp op) { op.setVersion(99); });
      module->push_back(fallback);
    }
    if (rawVhlo) {
      PassManager serialization(&context);
      serialization.addPass(
          mlir::stablehlo::createStablehloLegalizeToVhloPass());
      ASSERT_TRUE(succeeded(serialization.run(*module)));
    }

    TargetExtension extension;
    ASSERT_TRUE(succeeded(runInputPipeline(*module, extension, mnemonic)));
    EXPECT_EQ(extension.preprocessingCalls, 1u);
    EXPECT_EQ(extension.rewrites->load(), 1u);
    EXPECT_EQ(extension.detectionCalls, mnemonic == "auto" ? 1u : 0u);
    EXPECT_EQ(extension.conversionCalls,
              mnemonic == "auto" && !includeFallback ? 0u : 1u);
    module->walk([&](Operation *op) {
      EXPECT_NE(op->getName().getDialectNamespace(), "vhlo");
      EXPECT_FALSE(isa<mlir::stablehlo::CompositeOp>(op));
    });
    for (StringRef name : {StringRef("registered"), StringRef("fallback")}) {
      auto function = module->lookupSymbol<func::FuncOp>(name);
      if (!function) {
        continue;
      }
      auto returned =
          cast<func::ReturnOp>(function.getBody().front().getTerminator());
      EXPECT_EQ(returned.getOperands().front(), function.getArgument(0));
    }
  }
};

TEST_F(TargetPipelineTest, ExplicitStablehloRunsPreprocessingBeforeFallback) {
  runPipeline("stablehlo", false, true);
}
TEST_F(TargetPipelineTest, ExplicitVhloRunsPreprocessingBeforeFallback) {
  runPipeline("vhlo", true, true);
}
TEST_F(TargetPipelineTest, AutoDetectsRemainingFallbackAfterPreprocessing) {
  runPipeline("auto", false, true);
}
TEST_F(TargetPipelineTest, AutoAcceptsFullyRewrittenGenericInput) {
  runPipeline("auto", false, false);
}
TEST_F(TargetPipelineTest, AutoNormalizesVhloBeforeDetection) {
  runPipeline("auto", true, true);
}

} // namespace
} // namespace mlir::iree_compiler::stablehlo
