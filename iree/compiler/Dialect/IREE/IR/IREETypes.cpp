// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"

#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "llvm/ADT/BitVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/Interfaces/CastInterfaces.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {

//===----------------------------------------------------------------------===//
// ListType
//===----------------------------------------------------------------------===//

namespace detail {

struct ListTypeStorage : public TypeStorage {
  ListTypeStorage(Type elementType) : elementType(elementType) {}

  /// The hash key used for uniquing.
  using KeyTy = Type;
  bool operator==(const KeyTy &key) const { return key == elementType; }

  static ListTypeStorage *construct(TypeStorageAllocator &allocator,
                                    const KeyTy &key) {
    // Initialize the memory using placement new.
    return new (allocator.allocate<ListTypeStorage>()) ListTypeStorage(key);
  }

  Type elementType;
};

}  // namespace detail

// static
bool ListType::isCompatible(Type type) { return true; }

// static
bool ListType::canImplicitlyCast(Type from, Type to) {
  if (from.isa<IREE::VariantType>() || to.isa<IREE::VariantType>()) return true;
  if (from.isa<TensorType>() && to.isa<TensorType>()) return true;
  return from == to;
}

ListType ListType::get(Type elementType) {
  return Base::get(elementType.getContext(), elementType);
}

ListType ListType::getChecked(Type elementType, Location location) {
  return Base::getChecked(location, elementType);
}

ListType ListType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                              Type elementType) {
  return Base::getChecked(emitError, elementType.getContext(), elementType);
}

Type ListType::getElementType() { return getImpl()->elementType; }

//===----------------------------------------------------------------------===//
// PtrType
//===----------------------------------------------------------------------===//

namespace detail {

struct PtrTypeStorage : public TypeStorage {
  PtrTypeStorage(Type targetType) : targetType(targetType) {}

  /// The hash key used for uniquing.
  using KeyTy = Type;
  bool operator==(const KeyTy &key) const { return key == targetType; }

  static PtrTypeStorage *construct(TypeStorageAllocator &allocator,
                                   const KeyTy &key) {
    // Initialize the memory using placement new.
    return new (allocator.allocate<PtrTypeStorage>()) PtrTypeStorage(key);
  }

  Type targetType;
};

}  // namespace detail

PtrType PtrType::get(Type targetType) {
  return Base::get(targetType.getContext(), targetType);
}

PtrType PtrType::getChecked(Type targetType, Location location) {
  return Base::getChecked(location, targetType);
}

PtrType PtrType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                            Type targetType) {
  return Base::getChecked(emitError, targetType.getContext(), targetType);
}

Type PtrType::getTargetType() { return getImpl()->targetType; }

//===----------------------------------------------------------------------===//
// TiedOpInterface
//===----------------------------------------------------------------------===//

llvm::Optional<unsigned> detail::getTiedResultOperandIndex(
    Operation *op, unsigned resultIndex) {
  auto storageAttr =
      op->getAttrOfType<ArrayAttr>(TiedOpInterface::getStorageAttrName());
  if (!storageAttr) return llvm::None;
  auto valueAttrs = storageAttr.getValue();
  if (valueAttrs.empty()) return llvm::None;
  int64_t value = valueAttrs[resultIndex].cast<IntegerAttr>().getInt();
  if (value == TiedOpInterface::kUntiedIndex) return llvm::None;
  auto tiedOp = cast<TiedOpInterface>(op);
  unsigned tiedOperandsOffset = tiedOp.getTiedOperandsIndexAndLength().first;
  return tiedOperandsOffset + static_cast<unsigned>(value);
}

void detail::setTiedResultOperandIndex(Operation *op, unsigned resultIndex,
                                       llvm::Optional<unsigned> operandIndex) {
  auto indices = getTiedResultOperandIndices(op);
  if (indices.empty()) {
    indices.resize(op->getNumResults(), TiedOpInterface::kUntiedIndex);
  } else {
    // Well, getTiedResultOperandIndices() returns indices into the full range
    // of the op, but in the attribute, we expect to store ranges into the range
    // returned by `getTiedOperandsIndexAndLength`.
    auto tiedOp = cast<TiedOpInterface>(op);
    unsigned tiedOperandsOffset = tiedOp.getTiedOperandsIndexAndLength().first;
    for (auto &index : indices) {
      if (index != TiedOpInterface::kUntiedIndex) index -= tiedOperandsOffset;
    }
  }

  indices[resultIndex] = operandIndex.hasValue()
                             ? operandIndex.getValue()
                             : TiedOpInterface::kUntiedIndex;
  op->setAttr(TiedOpInterface::getStorageAttrName(),
              Builder(op).getIndexArrayAttr(indices));
}

SmallVector<int64_t, 4> detail::getTiedResultOperandIndices(Operation *op) {
  SmallVector<int64_t, 4> indices;
  auto storageAttr =
      op->getAttrOfType<ArrayAttr>(TiedOpInterface::getStorageAttrName());
  if (!storageAttr) return indices;
  auto valueAttrs = storageAttr.getValue();
  if (valueAttrs.empty()) return indices;
  auto tiedOp = cast<TiedOpInterface>(op);
  unsigned tiedOperandsOffset = tiedOp.getTiedOperandsIndexAndLength().first;
  indices.resize(op->getNumResults());
  for (unsigned i = 0; i < valueAttrs.size(); ++i) {
    int64_t index = valueAttrs[i].cast<IntegerAttr>().getInt();
    indices[i] = index != TiedOpInterface::kUntiedIndex
                     ? tiedOperandsOffset + index
                     : TiedOpInterface::kUntiedIndex;
  }
  return indices;
}

Value TiedOpInterface::findTiedBaseValue(Value derivedValue) {
  Value baseValue = derivedValue;
  while (auto definingOp =
             dyn_cast_or_null<TiedOpInterface>(baseValue.getDefiningOp())) {
    auto tiedValue = definingOp.getTiedResultOperand(baseValue);
    if (!tiedValue) break;
    baseValue = tiedValue;
  }
  return baseValue;
}

LogicalResult detail::verifyTiedOp(TiedOpInterface tiedOp) {
  auto storageAttr =
      tiedOp->getAttrOfType<ArrayAttr>(TiedOpInterface::getStorageAttrName());
  if (!storageAttr || storageAttr.getValue().empty()) {
    return success();
  }
  auto tiedOperandIndices = storageAttr.getValue();
  if (tiedOperandIndices.size() != tiedOp->getNumResults()) {
    return tiedOp.emitError("op results/tied operand indices mismatch");
  }
  return success();
}

void excludeTiedOperandAndResultIndices(
    ArrayRef<unsigned> excludedOperandIndices,
    ArrayRef<unsigned> excludedResultIndices,
    SmallVector<int64_t, 4> &tiedOperandIndices) {
  SmallVector<int64_t, 4> oldTiedOperandIndices = tiedOperandIndices;
  tiedOperandIndices.clear();

  // To adjust operand indices we need to know the how many operands to offset
  // the indices by - if 2 operands before operand N were removed then we know
  // it needs to be -2. This is nasty but that's why we have this helper
  // function.
  unsigned numBits = 1;
  if (!excludedOperandIndices.empty()) {
    numBits += *std::max_element(excludedOperandIndices.begin(),
                                 excludedOperandIndices.end());
  }
  llvm::BitVector excludedOperands(numBits, false);
  for (unsigned i = 0; i < excludedOperandIndices.size(); ++i) {
    excludedOperands[excludedOperandIndices[i]] = true;
  }

  for (auto it : llvm::enumerate(oldTiedOperandIndices)) {
    unsigned resultIndex = it.index();
    if (llvm::is_contained(excludedResultIndices, resultIndex)) {
      continue;  // result removed
    }

    int64_t tiedOperandIndex = it.value();
    if (tiedOperandIndex != TiedOpInterface::kUntiedIndex) {
      // Check whether this operand is removed. If so, untie. We need to do this
      // before calculating the new operand index given `excludedOperandIndices`
      // contains the old indices.
      if (llvm::is_contained(excludedOperandIndices, tiedOperandIndex)) {
        tiedOperandIndex = TiedOpInterface::kUntiedIndex;
      }

      // Count up the number of removed operands prior to this one.
      unsigned offset = 0;
      for (unsigned i = 0; i < tiedOperandIndex; ++i) {
        if (i < excludedOperands.size() && excludedOperands[i]) ++offset;
      }

      tiedOperandIndex -= offset;
    }
    tiedOperandIndices.push_back(tiedOperandIndex);
  }
}

// At the end so it can use functions above:
#include "iree/compiler/Dialect/IREE/IR/IREEOpInterfaces.cpp.inc"

}  // namespace IREE

//===----------------------------------------------------------------------===//
// IREEDialect
//===----------------------------------------------------------------------===//

void IREEDialect::registerTypes() {
  addTypes<IREE::ByteBufferType, IREE::ListType, IREE::MutableByteBufferType,
           IREE::PtrType, IREE::VariantType>();
}

}  // namespace iree_compiler
}  // namespace mlir
