// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Util/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-util-import-resources"

namespace mlir::iree_compiler::IREE::Util {

namespace {

template <typename ElementType, unsigned numBits = sizeof(ElementType) * 8>
static void copyIntAttrIntoBlob(AsmResourceBlob &blob,
                                DenseIntElementsAttr attr) {
  ArrayRef<ElementType> data = blob.getDataAs<ElementType>();
  MutableArrayRef<ElementType> rwData = MutableArrayRef<ElementType>(
      const_cast<ElementType *>(data.data()), data.size());
  ArrayRef<char> rawSrcData = attr.getRawData();
  if (rawSrcData.size() == blob.getData().size()) {
    // Memcpy.
    std::memcpy(rwData.data(), rawSrcData.data(), rawSrcData.size());
  } else {
    // Slow.
    size_t index = 0;
    for (APInt value : attr.getValues<APInt>()) {
      rwData[index++] = value.extractBitsAsZExtValue(numBits, 0);
    }
  }
}

template <typename ElementType, unsigned numBits = sizeof(ElementType) * 8>
static void copyFPAttrIntoBlob(AsmResourceBlob &blob,
                               DenseFPElementsAttr attr) {
  ArrayRef<ElementType> data = blob.getDataAs<ElementType>();
  MutableArrayRef<ElementType> rwData = MutableArrayRef<ElementType>(
      const_cast<ElementType *>(data.data()), data.size());
  ArrayRef<char> rawSrcData = attr.getRawData();
  if (rawSrcData.size() == blob.getData().size()) {
    // Memcpy.
    std::memcpy(rwData.data(), rawSrcData.data(), rawSrcData.size());
  } else {
    // Slow.
    size_t index = 0;
    for (APFloat value : attr.getValues<APFloat>()) {
      rwData[index++] =
          value.bitcastToAPInt().extractBitsAsZExtValue(numBits, 0);
    }
  }
}

class ImportResourcesPass : public ImportResourcesBase<ImportResourcesPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<BuiltinDialect>();
  }

  void runOnOperation() override {
    llvm::DenseMap<Attribute, Attribute> replacements;

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
          if (shouldConvertElements(elements)) {
            LLVM_DEBUG(llvm::dbgs() << ":: Converting elements attr of "
                                    << elements.getType() << "\n");
            if (auto replacement = convertElementsAttr(elements)) {
              attr.setValue(replacement);
              replacements[elements] = replacement;
              updated = true;
            } else {
              LLVM_DEBUG(llvm::dbgs() << "  Failed to convert\n");
            }
          }
        }
      }
      if (updated)
        op->setAttrs(attrs);
    });
    LLVM_DEBUG(llvm::dbgs() << "DONE CONVERTING RESOURCES\n");
  }

  static bool shouldConvertElements(ElementsAttr attr) {
    if (llvm::isa<DenseElementsAttr>(attr)) {
      // DenseElementsAttr encodes arbitrary dimension
      // splats whereas DenseResourceElementsAttr does not.
      return !attr.isSplat();
    }

    return false;
  }

  static ElementsAttr convertElementsAttr(ElementsAttr elementsAttr) {
    auto st = llvm::cast<ShapedType>(elementsAttr.getType());
    auto elementType = st.getElementType();
    auto numElements = elementsAttr.getNumElements();
    auto bitWidth = elementType.getIntOrFloatBitWidth();
    AsmResourceBlob blob;
    if (auto attr = llvm::dyn_cast<DenseIntElementsAttr>(elementsAttr)) {
      switch (bitWidth) {
      case 1:
        blob = HeapAsmResourceBlob::allocate(numElements, /*align=*/64,
                                             /*dataIsMutable=*/true);
        copyIntAttrIntoBlob<uint8_t, /*numBits=*/1>(blob, attr);
        return DenseResourceElementsAttr::get(st, "dense_elements_i1",
                                              std::move(blob));
      case 8:
        blob = HeapAsmResourceBlob::allocate(numElements, /*align=*/64,
                                             /*dataIsMutable=*/true);
        copyIntAttrIntoBlob<uint8_t>(blob, attr);
        return DenseResourceElementsAttr::get(st, "dense_elements_i8",
                                              std::move(blob));
      case 16:
        blob = HeapAsmResourceBlob::allocate(2 * numElements, /*align=*/64,
                                             /*dataIsMutable=*/true);
        copyIntAttrIntoBlob<uint16_t>(blob, attr);
        return DenseResourceElementsAttr::get(st, "dense_elements_i16",
                                              std::move(blob));
      case 32:
        blob = HeapAsmResourceBlob::allocate(4 * numElements, /*align=*/64,
                                             /*dataIsMutable=*/true);
        copyIntAttrIntoBlob<uint32_t>(blob, attr);
        return DenseResourceElementsAttr::get(st, "dense_elements_i32",
                                              std::move(blob));
      case 64:
        blob = HeapAsmResourceBlob::allocate(8 * numElements, /*align=*/64,
                                             /*dataIsMutable=*/true);
        copyIntAttrIntoBlob<uint64_t>(blob, attr);
        return DenseResourceElementsAttr::get(st, "dense_elements_i64",
                                              std::move(blob));
      default:
        return {};
      }
    } else if (auto attr = llvm::dyn_cast<DenseFPElementsAttr>(elementsAttr)) {
      AsmResourceBlob blob;
      switch (bitWidth) {
      case 8:
        blob = HeapAsmResourceBlob::allocate(numElements, /*align=*/64,
                                             /*dataIsMutable=*/true);
        copyFPAttrIntoBlob<uint8_t>(blob, attr);
        return DenseResourceElementsAttr::get(st, "dense_elements_f8",
                                              std::move(blob));
      case 16:
        blob = HeapAsmResourceBlob::allocate(2 * numElements, /*align=*/64,
                                             /*dataIsMutable=*/true);
        copyFPAttrIntoBlob<uint16_t>(blob, attr);
        return DenseResourceElementsAttr::get(st, "dense_elements_f16",
                                              std::move(blob));
      case 32:
        blob = HeapAsmResourceBlob::allocate(4 * numElements, /*align=*/64,
                                             /*dataIsMutable=*/true);
        copyFPAttrIntoBlob<uint32_t>(blob, attr);
        return DenseResourceElementsAttr::get(st, "dense_elements_f32",
                                              std::move(blob));
      case 64:
        blob = HeapAsmResourceBlob::allocate(8 * numElements, /*align=*/64,
                                             /*dataIsMutable=*/true);
        copyFPAttrIntoBlob<uint64_t>(blob, attr);
        return DenseResourceElementsAttr::get(st, "dense_elements_f64",
                                              std::move(blob));
      default:
        return {};
      }
    }
    return {};
  }
};

} // namespace

std::unique_ptr<OperationPass<void>> createImportResourcesPass() {
  return std::make_unique<ImportResourcesPass>();
}

} // namespace mlir::iree_compiler::IREE::Util
