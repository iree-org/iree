// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/StableHLO/Conversion/Passes.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"

namespace mlir::iree_compiler::stablehlo {

namespace {

struct StableHLOOptions {
  void bindOptions(OptionsBinder &binder) {}
};

static bool checkOpForTuples(Operation *op) {
  if (auto funcOp = dyn_cast<mlir::FunctionOpInterface>(op)) {
    for (auto t : funcOp.getArgumentTypes()) {
      if (isa<TupleType>(t)) {
        return true;
      }
    }
    for (auto t : funcOp.getResultTypes()) {
      if (isa<TupleType>(t)) {
        return true;
      }
    }
  }

  // Check for tuple operands or results.
  for (auto t : op->getOperandTypes()) {
    if (isa<TupleType>(t)) {
      return true;
    }
  }
  for (auto t : op->getResultTypes()) {
    if (isa<TupleType>(t)) {
      return true;
    }
  }

  return false;
}

// StableHLO (https://github.com/openxla/stablehlo) support plugin.
//
// The StableHLO plugin provides dialects, passes and opt-in options.
// Therefore, it is appropriate for default activation.
struct StableHLOSession
    : public PluginSession<StableHLOSession, StableHLOOptions,
                           PluginActivationPolicy::DefaultActivated> {
  static void registerPasses() {
    // TODO(scotttodd): register other StableHLO passes?
    registerStableHLOConversionPasses();
  }

  void onRegisterDialects(DialectRegistry &registry) override {
    registry.insert<mlir::shape::ShapeDialect>();
    registry.insert<mlir::chlo::ChloDialect>();
    registry.insert<mlir::stablehlo::StablehloDialect>();
    registry.insert<mlir::vhlo::VhloDialect>();
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

  void populateDetectedCustomInputConversionTypes(
      ModuleOp &module, StringSet<> &typeMnemonics) override {

    auto *ctx = module.getContext();
    const Dialect *chloDialect = ctx->getLoadedDialect("chlo");
    const Dialect *stablehloDialect = ctx->getLoadedDialect("stablehlo");
    const Dialect *vhloDialect = ctx->getLoadedDialect("vhlo");

    // stablehlo ops _with tuples_    --> only "stablehlo_xla" type
    // stablehlo ops _without tuples_ --> only "stablehlo" type
    // no stablehlo ops --> no types

    bool hasStableHLO = false;
    bool hasTuples = false;
    module.walk([&](Operation *op) {
      Dialect *d = op->getDialect();
      if (d == chloDialect || d == stablehloDialect || d == vhloDialect) {
        hasStableHLO = true;
        if (checkOpForTuples(op)) {
          hasTuples = true;
          // Early exit, no need to continue scanning.
          return WalkResult::interrupt();
        }
        // Keep scanning in case a future op contains tuples.
      }
      return WalkResult::advance();
    });

    if (hasTuples) {
      typeMnemonics.insert("stablehlo_xla");
    } else if (hasStableHLO) {
      typeMnemonics.insert("stablehlo");
    }
  }
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
