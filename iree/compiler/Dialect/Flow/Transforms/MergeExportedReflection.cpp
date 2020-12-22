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

#include "iree/base/signature_mangle.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using iree::SignatureBuilder;
using iree::SignatureParser;

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

class MergeExportedReflectionPass
    : public PassWrapper<MergeExportedReflectionPass, FunctionPass> {
  void runOnFunction() override {
    auto func = getFunction();
    Builder builder(&getContext());
    Identifier reflectionIdent = builder.getIdentifier("iree.reflection");

    // Only process exported functions.
    if (!func->getAttr("iree.module.export")) return;
    if (func->getAttr("iree.abi.none")) return;

    // Accumulate input and results into these.
    std::string inputsAccum;
    std::string resultsAccum;

    auto addItem = [&](Attribute partialAttr, llvm::StringRef operandType,
                       int operandIndex) {
      auto partialString = partialAttr.dyn_cast_or_null<StringAttr>();
      if (!partialString) {
        // Ignore it - it is normal for synthetic args/results to not be
        // annotated so long as the ABI wrappers expect the convention.
        return success();
      }

      SignatureParser p(absl::string_view(partialString.getValue().begin(),
                                          partialString.getValue().size()));
      char tag = p.tag();
      if (p.type() == SignatureParser::Type::kSpan && tag == 'I') {
        inputsAccum.append(p.sval().data(), p.sval().size());
        return success();
      } else if (p.type() == SignatureParser::Type::kSpan && tag == 'R') {
        resultsAccum.append(p.sval().data(), p.sval().size());
        return success();
      } else {
        llvm::StringRef sval_stringref(p.sval().data(), p.sval().size());
        func.emitError() << "Illegal partial reflection attribute: '"
                         << sval_stringref << "' on " << operandType << " "
                         << operandIndex;
        signalPassFailure();
        return failure();
      }
    };

    // Arguments (note that if run late, this can include arguments that
    // have been promoted to inputs but they should still be tagged correctly).
    auto fPartialIdent = builder.getIdentifier("f_partial");
    for (int i = 0, e = func.getNumArguments(); i < e; ++i) {
      DictionaryAttr l(
          func.getArgAttrOfType<DictionaryAttr>(i, reflectionIdent));
      if (l && failed(addItem(l.get(fPartialIdent), "argument", i))) {
        return;
      }
      NamedAttrList lAttrList(l);
      lAttrList.erase(fPartialIdent);
      if (!lAttrList.empty()) {
        func.setArgAttr(i, reflectionIdent,
                        lAttrList.getDictionary(&getContext()));
      } else {
        func.removeArgAttr(i, reflectionIdent);
      }
    }

    // Results.
    for (int i = 0, e = func.getNumResults(); i < e; ++i) {
      DictionaryAttr l(
          func.getResultAttrOfType<DictionaryAttr>(i, reflectionIdent));
      if (l && failed(addItem(l.get(fPartialIdent), "result", i))) {
        return;
      }
      NamedAttrList lAttrList(l);
      lAttrList.erase(fPartialIdent);
      if (!lAttrList.empty()) {
        func.setResultAttr(i, reflectionIdent,
                           lAttrList.getDictionary(&getContext()));
      } else {
        func.removeResultAttr(i, reflectionIdent);
      }
    }

    // And then merge and update the function level attribute.
    auto fIdent = builder.getIdentifier("f");
    auto fVersionIdent = builder.getIdentifier("fv");
    SignatureBuilder functionSignature;
    functionSignature.Span(inputsAccum, 'I');
    functionSignature.Span(resultsAccum, 'R');
    NamedAttrList l(func->getAttrOfType<DictionaryAttr>(reflectionIdent));
    l.set(fIdent, builder.getStringAttr(functionSignature.encoded()));
    l.set(fVersionIdent, builder.getStringAttr("1"));
    func->setAttr(reflectionIdent, l.getDictionary(&getContext()));
  }
};

std::unique_ptr<OperationPass<FuncOp>> createMergeExportedReflection() {
  return std::make_unique<MergeExportedReflectionPass>();
}

static PassRegistration<MergeExportedReflectionPass> pass(
    "iree-flow-merge-exported-reflection",
    "Merges per arg and result exported reflection metadata into function "
    "level reflection attributes.");

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
