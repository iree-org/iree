// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Translation/HALExecutable.h"

#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Utils/TracingUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-translate/Translation.h"

namespace mlir {
namespace iree_compiler {

// Converts from a module containing a hal.executable into the serialized form.
static LogicalResult translateFromMLIRToHALExecutable(
    mlir::ModuleOp moduleOp, IREE::HAL::TargetOptions executableOptions) {
  auto executableOps =
      llvm::to_vector<4>(moduleOp.getOps<IREE::HAL::ExecutableOp>());
  auto sourceOps =
      llvm::to_vector<4>(moduleOp.getOps<IREE::HAL::ExecutableSourceOp>());
  size_t usableOpCount = executableOps.size() + sourceOps.size();
  if (usableOpCount != 1) {
    return moduleOp.emitError()
           << "HAL executable translation requires "
              "exactly 1 top level hal.executable/hal.executable.source op";
  }

  PassManager passManager(moduleOp.getContext());
  mlir::applyPassManagerCLOptions(passManager);
  mlir::applyDefaultTimingPassManagerCLOptions(passManager);
  passManager.addInstrumentation(std::make_unique<PassTracing>());

  IREE::HAL::buildHALTransformPassPipeline(passManager, executableOptions);

  if (failed(passManager.run(moduleOp))) {
    return moduleOp.emitError() << "conversion from source -> HAL failed";
  }
  return success();
}

// Translates an MLIR module containing a single hal.executable into a
// target-specific binary form (such as an ELF file or a FlatBuffer containing
// a SPIR-V blob).
//
// Exposed via the --iree-mlir-to-hal-executable translation.
static LogicalResult translateFromMLIRToHALExecutableWithFlags(
    mlir::ModuleOp moduleOp, llvm::raw_ostream &output) {
  mlir::registerPassManagerCLOptions();

  // Convert into the final target-specific module definition and serialize it.
  auto executableOptions = IREE::HAL::TargetOptions::FromFlags::get();
  auto result = translateFromMLIRToHALExecutable(moduleOp, executableOptions);
  if (failed(result)) {
    return result;
  }

  // Extract the serialized binary representation from the executable.
  auto executableOp = *moduleOp.getOps<IREE::HAL::ExecutableOp>().begin();
  auto binaryOps =
      llvm::to_vector<4>(executableOp.getOps<IREE::HAL::ExecutableBinaryOp>());
  if (binaryOps.size() != 1) {
    return executableOp.emitError()
           << "executable translation failed to produce exactly 1 binary for "
              "the input executable";
  }
  auto binaryOp = binaryOps.front();
  auto rawData = binaryOp.data().getRawData();
  output.write(rawData.data(), rawData.size());
  return success();
}

void registerHALExecutableTranslation() {
  TranslateFromMLIRRegistration toHALExecutableWithFlags(
      "iree-mlir-to-hal-executable", translateFromMLIRToHALExecutableWithFlags);
}

}  // namespace iree_compiler
}  // namespace mlir
