// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TileSizeSelectionPattern.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/Target/LoweringStrategy.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace mlir::iree_compiler::IREE::HAL {

namespace {

class X86Mmt4DOpPattern : public OpTileSizeSelectionPattern<linalg::Mmt4DOp> {
  FailureOr<TileSizeAndPipelineConfig> matchAndConfig(
      FunctionOpInterface funcOp, linalg::Mmt4DOp op) const override {
    auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
    if (!isX86(targetAttr)) {
      return failure();
    }

    Value lhs = op.getDpsInputs()[0];
    Value rhs = op.getDpsInputs()[1];
    Value out = op.getDpsInits()[0];
    ShapedType lhsType = cast<ShapedType>(lhs.getType());
    ShapedType rhsType = cast<ShapedType>(rhs.getType());
    ShapedType outType = cast<ShapedType>(out.getType());

    llvm::dbgs() << "mmt4d: " << lhsType << " " << rhsType << " -> " << outType
                 << "\n";

    return failure();
  }
};

class X86GenericOpPattern
    : public OpTileSizeSelectionPattern<linalg::GenericOp> {
  FailureOr<TileSizeAndPipelineConfig> matchAndConfig(
      FunctionOpInterface funcOp, linalg::GenericOp op) const override {
    auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
    if (!isX86(targetAttr)) {
      return failure();
    }

    auto tileOp = cast<TilingInterface>(op.getOperation());

    llvm::dbgs() << "generic: [";
    for (auto iterType : tileOp.getLoopIteratorTypes()) {
      llvm::dbgs() << "," << iterType;
    }
    llvm::dbgs() << "]";
    for (auto input : op.getDpsInputs()) {
      llvm::dbgs() << " " << cast<ShapedType>(input.getType());
    }
    llvm::dbgs() << " ->";
    for (auto output : op.getDpsInits()) {
      llvm::dbgs() << " " << cast<ShapedType>(output.getType());
    }
    llvm::dbgs() << "\n";

    return failure();
  }
};

class X86PackOpPattern : public OpTileSizeSelectionPattern<tensor::PackOp> {
  FailureOr<TileSizeAndPipelineConfig> matchAndConfig(
      FunctionOpInterface funcOp, tensor::PackOp op) const override {
    auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
    if (!isX86(targetAttr)) {
      return failure();
    }

    llvm::dbgs() << "pack: " << op.getSourceType() << " -> " << op.getDestType()
                 << "\n";

    return failure();
  }
};

class LLVMCPUPatternLoweringStrategy : public IREE::HAL::LoweringStrategy {
  LogicalResult matchAndSetTranslationInfo(
      FunctionOpInterface funcOp) override {
    SmallVector<std::unique_ptr<TileSizeSelectionPattern>> patterns;
    patterns.push_back(std::make_unique<X86Mmt4DOpPattern>());
    patterns.push_back(std::make_unique<X86GenericOpPattern>());
    patterns.push_back(std::make_unique<X86PackOpPattern>());

    SmallVector<Operation *> computeOps = getComputeOps(funcOp);
    for (Operation *op : computeOps) {
      for (auto &pattern : patterns) {
        FailureOr<TileSizeAndPipelineConfig> config =
            pattern->matchAndConfig(funcOp, op);
        if (succeeded(config)) {
          break;
        }
      }
    }
    return failure();
  }
};

class PluginRegistration
    : public PluginSession<PluginRegistration, EmptyPluginOptions,
                           PluginActivationPolicy::DefaultActivated> {
  void configureHALTargetBackends(
      IREE::HAL::TargetRegistry &registry) override {
    auto backend = registry.getTargetBackend("llvm-cpu");
    backend->addLoweringStrategy(
        std::make_unique<LLVMCPUPatternLoweringStrategy>());
  }
};

}  // namespace

}  // namespace mlir::iree_compiler::IREE::HAL

extern "C" bool
iree_register_compiler_plugin_hal_lowering_strategy_llvmcpu_pattern_lowering_strategy(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar
      ->registerPlugin<::mlir::iree_compiler::IREE::HAL::PluginRegistration>(
          "hal_lowering_strategy_llvmcpu_pattern_lowering_strategy");
  return true;
}
