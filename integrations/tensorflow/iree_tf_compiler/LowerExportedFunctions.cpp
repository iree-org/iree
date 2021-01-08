// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree_tf_compiler/Passes.h"
#include "iree/base/signature_mangle.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"

namespace mlir {
namespace iree_integrations {
namespace TF {

using ::iree::SipSignatureMangler;

static LogicalResult setRawSignatureIndex(FuncOp funcOp,
                                          SipSignatureMangler &mangler,
                                          int rawIndex,
                                          ArrayAttr indexPathAttr) {
  llvm::SmallVector<SipSignatureMangler::Key, 8> indexKeys;
  for (auto &indexAttr : indexPathAttr) {
    if (auto stringAttr = indexAttr.dyn_cast<StringAttr>()) {
      auto stringRef = stringAttr.getValue();
      indexKeys.emplace_back(
          absl::string_view(stringRef.data(), stringRef.size()));
    } else if (auto intAttr = indexAttr.dyn_cast<IntegerAttr>()) {
      indexKeys.emplace_back(intAttr.getInt());
    } else {
      return funcOp.emitError()
             << "Each index path component must be a string or integer";
    }
  }

  if (!mangler.SetRawSignatureIndex(rawIndex, indexKeys)) {
    return funcOp.emitError()
           << "Unable to generate mangled form for index path";
  }

  return success();
}

class LowerExportedFunctionsPass
    : public PassWrapper<LowerExportedFunctionsPass, OperationPass<ModuleOp>> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<iree_compiler::IREE::Flow::FlowDialect,
                    mlir::tf_saved_model::TensorFlowSavedModelDialect>();
  }

  void runOnOperation() override {
    if (failed(run())) {
      signalPassFailure();
    }
  }

  LogicalResult run() {
    mlir::Builder builder(getOperation());
    const Identifier savedModelIndexPathIdent =
        builder.getIdentifier("tf_saved_model.index_path");
    const Identifier sipIdent = builder.getIdentifier("sip");
    const Identifier abiIdent = builder.getIdentifier("abi");
    const Identifier abiVersionIdent = builder.getIdentifier("abiv");

    // Handle saved model exported functions.
    for (auto func : getOperation().getOps<FuncOp>()) {
      // Transfer exported names to IREE.
      auto exported_names = mlir::tf_saved_model::GetExportedNames(func);
      if (exported_names.empty()) continue;

      // TODO(laurenzo): After VM rework, we should just keep the
      // function name as-is and create explicit export ops for each exported
      // function.
      if (exported_names.size() > 1) {
        return func.emitError() << "Multiple exported names not supported yet";
      }
      func.setName(exported_names.front());

      // Function level reflection attributes.
      SipSignatureMangler inputsMangler;
      SipSignatureMangler resultsMangler;
      SmallVector<NamedAttribute, 3> funcReflectAttrs;
      funcReflectAttrs.push_back(
          builder.getNamedAttr(abiIdent, builder.getStringAttr(sipIdent)));
      funcReflectAttrs.push_back(
          builder.getNamedAttr(abiVersionIdent, builder.getI32IntegerAttr(1)));

      // Tag it as an IREE exported function.
      func->setAttr("iree.module.export", builder.getUnitAttr());

      // Process per-argument attrs and generate reflection metadata.
      for (int i = 0, e = func.getNumArguments(); i < e; i++) {
        auto indexPathAttr =
            func.getArgAttrOfType<mlir::ArrayAttr>(i, savedModelIndexPathIdent);
        if (!indexPathAttr) {
          return func.emitError()
                 << "Missing argument attribute: " << savedModelIndexPathIdent;
        }
        func.removeArgAttr(i, savedModelIndexPathIdent);

        if (failed(
                setRawSignatureIndex(func, inputsMangler, i, indexPathAttr))) {
          return failure();
        }
      }

      // Process per-result attrs and generate reflection metadata.
      for (int i = 0, e = func.getNumResults(); i < e; i++) {
        auto indexPathAttr = func.getResultAttrOfType<mlir::ArrayAttr>(
            i, savedModelIndexPathIdent);
        if (!indexPathAttr) {
          return func.emitError()
                 << "Missing result attribute: " << savedModelIndexPathIdent;
        }
        func.removeResultAttr(i, savedModelIndexPathIdent);

        if (failed(
                setRawSignatureIndex(func, resultsMangler, i, indexPathAttr))) {
          return failure();
        }
      }

      // Add the function level reflection attribute.
      auto functionSignature = SipSignatureMangler::ToFunctionSignature(
          inputsMangler, resultsMangler);
      if (!functionSignature) {
        return func.emitError() << "Unable to generate sip function signature";
      }
      funcReflectAttrs.push_back(builder.getNamedAttr(
          sipIdent, builder.getStringAttr(functionSignature->encoded())));

      if (!funcReflectAttrs.empty()) {
        func->setAttr("iree.reflection",
                      builder.getDictionaryAttr(funcReflectAttrs));
      }

      // Remove its designation as a saved model export.
      func.removeAttr("tf_saved_model.exported_names");
    }

    // We should have now removed anything requiring saved model semantics.
    getOperation().removeAttr("tf_saved_model.semantics");
    return success();
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createLowerExportedFunctionsPass() {
  return std::make_unique<LowerExportedFunctionsPass>();
}

static PassRegistration<LowerExportedFunctionsPass> pass(
    "iree-tf-saved-model-lower-exported-functions",
    "Lower tf_saved_model exported functions to ones with IREE SIP metadata");

}  // namespace TF
}  // namespace iree_integrations
}  // namespace mlir
