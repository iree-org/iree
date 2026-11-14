// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/StableHLO/Conversion/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir::iree_compiler::stablehlo {

namespace {

struct StableHLOOptions {
  void bindOptions(OptionsBinder &binder) {}
};

// StableHLO (https://github.com/openxla/stablehlo) support plugin.
//
// The StableHLO plugin provides dialects, passes and opt-in options.
// Therefore, it is appropriate for default activation.
struct StableHLOSession
    : PluginSession<StableHLOSession, StableHLOOptions,
                    PluginActivationPolicy::DefaultActivated> {
  static void registerPasses() {
    // TODO(scotttodd): register other StableHLO passes?
    registerStableHLOConversionPasses();
    mlir::stablehlo::registerChloLegalizeToStablehloPass();
  }

  void onRegisterDialects(DialectRegistry &registry) override {
    registry.insert<mlir::shape::ShapeDialect>();
    registry.insert<mlir::chlo::ChloDialect>();
    registry.insert<mlir::stablehlo::StablehloDialect>();
    registry.insert<mlir::vhlo::VhloDialect>();
    registry.insert<IREE::Flow::FlowDialect>();

    // Automatic input conversion constructs these pipelines after the outer
    // pass manager has loaded its dependent dialects. Register their union
    // here so a VHLO-only input does not rely on dialects loaded by parsing.
    StableHloOptions stableHloOptions;
    OpPassManager passManager(ModuleOp::getOperationName());
    buildStableHLOInputConversionPassPipeline(passManager, stableHloOptions);
    buildStableHLOXLAInputConversionPassPipeline(passManager, stableHloOptions);
    passManager.getDependentDialects(registry);
  }

  bool extendCustomInputConversionPassPipeline(
      OpPassManager &passManager, std::string_view typeMnemonic) override {
    StableHloOptions stableHloOptions;

    // VHLO is converted to StableHLO. The conversion function is called
    // automatically, and if the input is fully stablehlo the function
    // acts as Nop.
    if (typeMnemonic == "stablehlo" || typeMnemonic == "vhlo") {
      buildStableHLOInputConversionPassPipeline(passManager, stableHloOptions);
      return true;
    } else if (typeMnemonic == "stablehlo_xla") {
      buildStableHLOXLAInputConversionPassPipeline(passManager,
                                                   stableHloOptions);
      return true;
    }

    return false;
  }

  void populateCustomInputConversionTypes(StringSet<> &typeMnemonics) override {
    typeMnemonics.insert("stablehlo");
    typeMnemonics.insert("stablehlo_xla");
    typeMnemonics.insert("vhlo");
  }

  // Detects the input conversion a module needs:
  //   input dialect operations, types or attributes with tuples
  //       --> only "stablehlo_xla"
  //   input dialect operations, types or attributes without tuples
  //       --> only "stablehlo"
  //   no input dialect use --> no types
  void populateDetectedCustomInputConversionTypes(
      ModuleOp &module, StringSet<> &typeMnemonics) override {
    InputDialectScan scan(module.getContext());
    scan.run(module);
    if (scan.hasStableHLO && scan.hasTuples) {
      typeMnemonics.insert("stablehlo_xla");
    } else if (scan.hasStableHLO) {
      typeMnemonics.insert("stablehlo");
    }
  }

private:
  // Finds uses of the CHLO, StableHLO and VHLO dialects and of tuples. Input
  // dialect types and tensor encodings also require conversion when the
  // module has no input dialect operations. Any tuples in such a module
  // require the tuple-flattening pipeline.
  class InputDialectScan {
  public:
    explicit InputDialectScan(MLIRContext *ctx)
        : chloDialect(ctx->getLoadedDialect("chlo")),
          stablehloDialect(ctx->getLoadedDialect("stablehlo")),
          vhloDialect(ctx->getLoadedDialect("vhlo")) {
      typeWalker.addWalk([this](Type type) {
        hasStableHLO |= isInputDialect(&type.getDialect());
        hasTuples |= isa<TupleType>(type);
      });
      typeWalker.addWalk([this](Attribute attr) {
        hasStableHLO |= isInputDialect(&attr.getDialect());
      });
    }

    // The type walker refers to this scan.
    InputDialectScan(const InputDialectScan &) = delete;
    InputDialectScan &operator=(const InputDialectScan &) = delete;

    // Scans until both the input dialect and tuple requirements are known,
    // even if they occur in different operations.
    void run(ModuleOp module) {
      module.walk([this](Operation *op) {
        scanOperation(op);
        return hasStableHLO && hasTuples ? WalkResult::interrupt()
                                         : WalkResult::advance();
      });
    }

    bool hasStableHLO = false;
    bool hasTuples = false;

  private:
    bool isInputDialect(const Dialect *dialect) const {
      return dialect && (dialect == chloDialect ||
                         dialect == stablehloDialect || dialect == vhloDialect);
    }

    void walkTypes(TypeRange types) {
      for (Type type : types) {
        typeWalker.walk(type);
      }
    }

    void scanOperation(Operation *op) {
      hasStableHLO |= isInputDialect(op->getDialect());
      walkTypes(op->getOperandTypes());
      walkTypes(op->getResultTypes());
      if (auto funcOp = dyn_cast<FunctionOpInterface>(op)) {
        // Defined functions expose their inputs through entry block arguments.
        if (funcOp.isExternal()) {
          walkTypes(funcOp.getArgumentTypes());
        }
        walkTypes(funcOp.getResultTypes());
      }
      for (Region &region : op->getRegions()) {
        for (Block &block : region) {
          walkTypes(block.getArgumentTypes());
        }
      }
    }

    const Dialect *chloDialect;
    const Dialect *stablehloDialect;
    const Dialect *vhloDialect;
    AttrTypeWalker typeWalker;
  };
};

} // namespace

} // namespace mlir::iree_compiler::stablehlo

IREE_DEFINE_COMPILER_OPTION_FLAGS(
    ::mlir::iree_compiler::stablehlo::StableHLOOptions);

extern "C" bool iree_register_compiler_plugin_input_stablehlo(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<::mlir::iree_compiler::stablehlo::StableHLOSession>(
      "input_stablehlo");
  return true;
}
