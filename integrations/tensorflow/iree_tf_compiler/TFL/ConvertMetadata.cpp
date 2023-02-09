// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_tf_compiler/TFL/PassDetail.h"
#include "iree_tf_compiler/TFL/Passes.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_integrations {
namespace TFL {
namespace {

// Extract the input and output names
static void splitFunctionIONames(StringAttr namesAttr,
                                 llvm::SmallVectorImpl<std::string> &names) {
  SmallVector<StringRef, 4> namesRef;
  llvm::SplitString(namesAttr.getValue(), namesRef, ",");
  for (auto nameRef : namesRef) {
    names.push_back(nameRef.str());
  }
}

class ConvertModuleMetadataPass
    : public ConvertModuleMetadataBase<ConvertModuleMetadataPass> {
 public:
  void runOnOperation() override {
    // None currently handled.
  }
};

class ConvertFunctionMetadataPass
    : public ConvertFunctionMetadataBase<ConvertFunctionMetadataPass> {
 public:
  void runOnOperation() override {
    auto funcOp = getOperation();

    // Setup TF entry functions as an IREE entry point and preserve the
    // associated metadata. Note that TFLite uses `tf.entry_function`.
    auto entryFunctionAttr =
        funcOp->getAttrOfType<DictionaryAttr>("tf.entry_function");
    if (entryFunctionAttr) {
      setupEntryPointAttrs(funcOp, entryFunctionAttr);
    }
  }

 private:
  // TF/TFL pack their I/O names on an annoying dictionary. We want our shape
  // names to match up with those for readability so we extract them here.
  // Is this ugly? Yeah - but such is what we have to deal with here.
  void setupEntryPointAttrs(func::FuncOp funcOp,
                            DictionaryAttr entryFunctionAttr) {
    funcOp.setPublic();
    bool nonZeroInputs = funcOp.getNumArguments() > 0;
    bool nonZeroOutputs = funcOp.getNumResults() > 0;

    if (nonZeroInputs) {
      auto inputsAttr = entryFunctionAttr.get("inputs")
                            .template dyn_cast_or_null<StringAttr>();
      if (!inputsAttr) {
        funcOp.emitError() << "functions with tf.entry_function must have "
                              "input names to be handled by IREE";
        signalPassFailure();
        return;
      }
      SmallVector<std::string, 4> inputNames;
      splitFunctionIONames(inputsAttr, inputNames);
      if (inputNames.size() != funcOp.getNumArguments()) {
        funcOp.emitError()
            << "tf.entry_function attribute malformed: inputs don't "
               "match the function signature";
        signalPassFailure();
        return;
      }
      for (unsigned i = 0; i < inputNames.size(); ++i) {
        funcOp.setArgAttr(i, "iree.identifier",
                          StringAttr::get(&getContext(), inputNames[i]));
      }
    }
    if (nonZeroOutputs) {
      auto outputsAttr = entryFunctionAttr.get("outputs")
                             .template dyn_cast_or_null<StringAttr>();
      if (!outputsAttr) {
        funcOp.emitError() << "functions with tf.entry_function must have "
                              "output names to be handled by IREE";
        signalPassFailure();
        return;
      }
      SmallVector<std::string, 4> outputNames;
      splitFunctionIONames(outputsAttr, outputNames);
      if (outputNames.size() != funcOp.getNumResults()) {
        funcOp.emitError()
            << "tf.entry_function attribute malformed: outputs don't "
               "match the function signature";
        signalPassFailure();
        return;
      }
      for (unsigned i = 0; i < outputNames.size(); ++i) {
        funcOp.setResultAttr(i, "iree.identifier",
                             StringAttr::get(&getContext(), outputNames[i]));
      }
    }
  }
};
}  // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> createConvertModuleMetadataPass() {
  return std::make_unique<ConvertModuleMetadataPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
createConvertFunctionMetadataPass() {
  return std::make_unique<ConvertFunctionMetadataPass>();
}

}  // namespace TFL
}  // namespace iree_integrations
}  // namespace mlir
