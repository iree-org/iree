// Copyright 2023 Nod Labs, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/Pass/PassManager.h"
#include "torch-iree/InputConversion/Passes.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorDialect.h"
#include "torch-mlir/Conversion/Passes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"

namespace mlir::iree_compiler {

namespace {

struct TorchOptions {
  bool strictSymbolicShapes = true;
  void bindOptions(OptionsBinder &binder) {
    static llvm::cl::OptionCategory category("Torch Input");
    binder.opt<bool>(
        "iree-torch-use-strict-symbolic-shapes", strictSymbolicShapes,
        llvm::cl::cat(category),
        llvm::cl::desc("Forces dynamic shapes to be treated as strict"));
  }
};

// The shark-turbine plugin provides dialects, passes and opt-in options.
// Therefore, it is appropriate for default activation.
struct TorchSession
    : public PluginSession<TorchSession, TorchOptions,
                           PluginActivationPolicy::DefaultActivated> {
  static void registerPasses() {
    mlir::torch::registerTorchPasses();
    mlir::torch::registerTorchConversionPasses();
    mlir::torch::registerConversionPasses();
    TorchInput::registerTMTensorConversionPasses();
  }

  void onRegisterDialects(DialectRegistry &registry) override {
    registry.insert<torch::Torch::TorchDialect>();
    registry.insert<torch::TorchConversion::TorchConversionDialect>();
    registry.insert<mlir::torch::TMTensor::TMTensorDialect>();
  }

  bool extendCustomInputConversionPassPipeline(
      OpPassManager &passManager, std::string_view typeMnemonic) override {
    if (typeMnemonic == "torch") {
      TorchInput::TorchToIREELoweringPipelineOptions torchOptions;
      torchOptions.strictSymbolicShapes = options.strictSymbolicShapes;
      TorchInput::createTorchToIREEPipeline(passManager, torchOptions);
      passManager.addNestedPass<func::FuncOp>(
          TorchInput::createConvertTMTensorToLinalgExtPass());
      return true;
    }
    // TODO: Retire the tm_tensor input pipeline once we are fully switched
    // to the 'torch' pipeline, which handles everything from the 'torch'
    // dialect down (vs just 'tm_tensor' which was converting a couple of
    // ops to linalg).
    if (typeMnemonic == "tm_tensor") {
      passManager.addNestedPass<func::FuncOp>(
          TorchInput::createConvertTMTensorToLinalgExtPass());
      return true;
    }
    return false;
  }

  void populateCustomInputConversionTypes(StringSet<> &typeMnemonics) override {
    typeMnemonics.insert("tm_tensor");
    typeMnemonics.insert("torch");
  }
};

} // namespace

} // namespace mlir::iree_compiler

IREE_DEFINE_COMPILER_OPTION_FLAGS(::mlir::iree_compiler::TorchOptions);

extern "C" bool iree_register_compiler_plugin_torch_iree(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<::mlir::iree_compiler::TorchSession>("torch_iree");
  return true;
}
