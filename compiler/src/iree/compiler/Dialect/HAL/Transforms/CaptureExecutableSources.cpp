// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include "mlir/IR/AsmState.h"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_CAPTUREEXECUTABLESOURCESPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-hal-capture-executable-sources
//===----------------------------------------------------------------------===//

static bool hasDictionaryAttrEntry(Operation *op, StringRef dictionaryName,
                                   StringRef key) {
  auto dictionaryAttr = op->getAttrOfType<DictionaryAttr>(dictionaryName);
  return dictionaryAttr && dictionaryAttr.get(key);
}

static void insertDictionaryAttrEntry(Operation *op, StringRef dictionaryName,
                                      StringRef key, Attribute value) {
  NamedAttrList attrs;
  auto dictionaryAttr = op->getAttrOfType<DictionaryAttr>(dictionaryName);
  if (dictionaryAttr)
    attrs.assign(dictionaryAttr.getValue());
  attrs.set(key, value);
  op->setAttr(dictionaryName, DictionaryAttr::get(op->getContext(), attrs));
}

static Attribute getSourceAttr(MLIRContext *context, StringRef fileName,
                               StringRef source) {
  // TODO(benvanik): use our own resource attribute that allows us to store the
  // source string verbatim (and out-of-band) in the file. Today only element
  // attrs have resource equivalents upstream (no string resource attr).
  Builder b(context);
  auto blob = HeapAsmResourceBlob::allocateAndCopyInferAlign(
      ArrayRef<char>(source.data(), source.size()));
  return DenseI8ResourceElementsAttr::get(
      VectorType::get({static_cast<int64_t>(source.size())}, b.getI8Type()),
      fileName, std::move(blob));
}

struct CaptureExecutableSourcesPass
    : public IREE::HAL::impl::CaptureExecutableSourcesPassBase<
          CaptureExecutableSourcesPass> {
  using IREE::HAL::impl::CaptureExecutableSourcesPassBase<
      CaptureExecutableSourcesPass>::CaptureExecutableSourcesPassBase;
  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto moduleName = moduleOp.getName().value_or("module");

    for (auto executableOp : moduleOp.getOps<IREE::HAL::ExecutableOp>()) {
      for (auto variantOp :
           executableOp.getOps<IREE::HAL::ExecutableVariantOp>()) {
        // Skip externally defined variants as there's no source to capture.
        if (variantOp.isExternal())
          continue;

        // Ignore if there is already source assigned.
        auto fileName = (moduleName + "_" + executableOp.getName() + "_" +
                         variantOp.getName() + "." + stage + ".mlir")
                            .str();
        if (hasDictionaryAttrEntry(variantOp, "sources", fileName))
          continue;

        // Create a standalone executable with just the variant being captured.
        // This allows the source to be passed to iree-compile in the
        // hal-executable compilation mode.
        auto clonedExecutableOp = executableOp.cloneWithoutRegions();
        clonedExecutableOp.setVisibility(SymbolTable::Visibility::Public);
        OpBuilder clonedBuilder = OpBuilder::atBlockBegin(
            &clonedExecutableOp.getBody().emplaceBlock());
        auto clonedVariantOp = cast<IREE::HAL::ExecutableVariantOp>(
            clonedBuilder.clone(*variantOp));
        clonedBuilder.create<IREE::HAL::ExecutableEndOp>(
            clonedBuilder.getUnknownLoc());

        // Capture the source contents and update the locations in the IR to
        // reference it.
        std::string source;
        llvm::raw_string_ostream os(source);
        OpPrintingFlags flags;
        flags.useLocalScope();
        mlir::generateLocationsFromIR(os, fileName, clonedExecutableOp, flags);
        os << "\n"; // newline at end of file

        // Wrap up the contents and attach them to the variant.
        auto sourceAttr =
            getSourceAttr(variantOp.getContext(), fileName, source);
        insertDictionaryAttrEntry(variantOp, "sources", fileName, sourceAttr);

        // Extract the new locations of the exported functions and attach them
        // to the original.
        SymbolTable symbolTable(variantOp.getInnerModule());
        SymbolTable clonedSymbolTable(clonedVariantOp.getInnerModule());
        for (auto [exportOp, clonedExportOp] : llvm::zip_equal(
                 variantOp.getExportOps(), clonedVariantOp.getExportOps())) {
          // Attach the cloned function location that was updated to point into
          // the source file and attach it to the original function.
          auto clonedFuncOp =
              clonedSymbolTable.lookup(clonedExportOp.getSymName());
          if (clonedFuncOp) {
            insertDictionaryAttrEntry(exportOp, "source_locs", stage,
                                      clonedFuncOp->getLoc());
          }
        }

        clonedExecutableOp.erase();
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
