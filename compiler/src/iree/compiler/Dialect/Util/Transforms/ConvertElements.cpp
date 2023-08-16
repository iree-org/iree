// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Util/IR/CASResources.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-util-import-resources"

namespace mlir::iree_compiler::IREE::Util {

namespace {

class ConvertElementsPass : public ConvertElementsBase<ConvertElementsPass> {
public:
  using Converter = std::function<ElementsAttr(Location, ElementsAttr)>;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<BuiltinDialect>();
    registry.insert<UtilDialect>();
  }

  void runOnOperation() override {
    llvm::DenseMap<Attribute, Attribute> replacements;
    Converter converter;
    if (mode == "CASElements") {
      converter = &ConvertElementsPass::convertElementsAttrToCAS;
    } else if (mode == "DenseElements") {
      converter = &ConvertElementsPass::convertElementsAttrToDense;
    } else {
      emitError(UnknownLoc::get(&getContext()))
          << "unknown pass 'mode' = '" << mode << "'";
      signalPassFailure();
      return;
    }

    getOperation()->walk([&](Operation *op) {
      bool updated = false;
      SmallVector<NamedAttribute> attrs(op->getAttrs());
      for (auto &attr : attrs) {
        if (auto elements = llvm::dyn_cast<ElementsAttr>(attr.getValue())) {
          // Already seen?
          auto it = replacements.find(elements);
          if (it != replacements.end()) {
            LLVM_DEBUG(llvm::dbgs()
                       << ":: Replacing already encountered attr of "
                       << elements.getType() << "\n");
            attr.setValue(it->second);
            updated = true;
            continue;
          }

          // Convert.
          ElementsAttr replacement = converter(op->getLoc(), elements);
          if (replacement) {
            attr.setValue(replacement);
            replacements[elements] = replacement;
            updated = true;
          }
        }
      }
      if (updated)
        op->setAttrs(attrs);
    });
    LLVM_DEBUG(llvm::dbgs() << "DONE CONVERTING RESOURCES\n");
  }

  static ElementsAttr convertElementsAttrToCAS(Location loc,
                                               ElementsAttr elementsAttr) {
    if (llvm::isa<DenseElementsAttr>(elementsAttr) && elementsAttr.isSplat()) {
      // DenseElementsAttr encodes arbitrary dimension
      // splats whereas DenseResourceElementsAttr does not.
      // TODO: Also extend this to possibly be size threshold based.
      return {};
    }
    if (llvm::isa<CASElementsAttr>(elementsAttr)) {
      // Don't self convert.
      return {};
    }
    auto serializable = llvm::dyn_cast<SerializableAttrInterface>(elementsAttr);
    if (!serializable) {
      LLVM_DEBUG(llvm::dbgs() << "Cannot convert (not serializable) "
                              << elementsAttr.getType() << "\n");
      return {};
    }

    LLVM_DEBUG(llvm::dbgs() << "Converting elements attr "
                            << elementsAttr.getType() << "\n");
    int64_t storageSize = serializable.getStorageSize();
    if (storageSize <= 0) {
      LLVM_DEBUG(llvm::dbgs() << "Cannot convert elements attr "
                              << elementsAttr.getType() << "\n");
      return {};
    }

    CASResourceBuilder builder =
        CASResourceBuilder::allocateHeap(static_cast<size_t>(storageSize));
    if (failed(serializable.serializeToBuffer(
            loc, llvm::support::endianness::native, builder.getData()))) {
      return {};
    }

    CASManagerDialectInterface &casManager =
        CASManagerDialectInterface::get(loc.getContext());
    PopulatedCASResource::Reference ref =
        casManager.internGlobalResource(std::move(builder));
    return CASElementsAttr::get(elementsAttr.getShapedType(),
                                ref->getGlobalResource());
  }

  static ElementsAttr convertElementsAttrToDense(Location loc,
                                                 ElementsAttr elementsAttr) {
    if (llvm::isa<DenseElementsAttr>(elementsAttr)) {
      // Do not self convert.
      return {};
    }

    // Switch based on float vs int because builtin variants don't
    // support bag of bits access.
    ShapedType st = elementsAttr.getShapedType();
    Type elementType = st.getElementType();
    int64_t numElements = st.getNumElements();

    // TODO: We could fast-path this with various access to raw data,
    // but this is a utility pass and it is testing that elements
    // conversions works.
    if (llvm::isa<FloatType>(elementType)) {
      std::vector<APFloat> newValues;
      newValues.reserve(numElements);
      auto range = elementsAttr.tryGetValues<APFloat>();
      if (!range) {
        emitWarning(loc) << "elements attr does not support APFloat iteration";
        return {};
      }
      for (APFloat value : elementsAttr.getValues<APFloat>()) {
        newValues.push_back(value);
      }
      return DenseElementsAttr::get(st, newValues);
    } else if (llvm::isa<IntegerType>(elementType)) {
      std::vector<APInt> newValues;
      newValues.reserve(numElements);
      auto range = elementsAttr.tryGetValues<APInt>();
      if (!range) {
        emitWarning(loc) << "elements attr does not support APInt iteration";
        return {};
      }
      for (APInt value : *range) {
        newValues.push_back(value);
      }
      return DenseElementsAttr::get(st, newValues);
    } else {
      // Not supported.
      return {};
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<void>> createConvertElementsPass() {
  return std::make_unique<ConvertElementsPass>();
}

} // namespace mlir::iree_compiler::IREE::Util
