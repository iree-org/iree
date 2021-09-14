// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "llvm/ADT/BitVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Parser.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {

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
  if (from.isa<VariantType>() || to.isa<VariantType>()) {
    return true;
  } else if (from.isa<TensorType>() && to.isa<TensorType>()) {
    return true;
  }
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

Type PtrType::getTargetType() const { return getImpl()->targetType; }

//===----------------------------------------------------------------------===//
// IREE::Util::TiedOpInterface
//===----------------------------------------------------------------------===//

llvm::Optional<unsigned> detail::getTiedResultOperandIndex(
    Operation *op, unsigned resultIndex) {
  auto storageAttr =
      op->getAttrOfType<ArrayAttr>(TiedOpInterface::getStorageAttrName());
  if (!storageAttr) return llvm::None;
  auto valueAttrs = storageAttr.getValue();
  if (valueAttrs.empty()) return llvm::None;
  auto tiedOp = cast<TiedOpInterface>(op);
  resultIndex -= tiedOp.getTiedResultsIndexAndLength().first;
  int64_t value = valueAttrs[resultIndex].cast<IntegerAttr>().getInt();
  if (value == TiedOpInterface::kUntiedIndex) return llvm::None;
  unsigned tiedOperandsOffset = tiedOp.getTiedOperandsIndexAndLength().first;
  return tiedOperandsOffset + static_cast<unsigned>(value);
}

void detail::setTiedResultOperandIndex(Operation *op, unsigned resultIndex,
                                       llvm::Optional<unsigned> operandIndex) {
  auto tiedOp = cast<TiedOpInterface>(op);
  auto resultRange = tiedOp.getTiedResultsIndexAndLength();
  resultIndex -= resultRange.first;

  auto indices = getTiedResultOperandIndices(op);
  if (indices.empty()) {
    indices.resize(resultRange.second, TiedOpInterface::kUntiedIndex);
  } else {
    // Well, getTiedResultOperandIndices() returns indices into the full range
    // of the op, but in the attribute, we expect to store ranges into the range
    // returned by `getTiedOperandsIndexAndLength`.
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
  auto resultRange = tiedOp.getTiedResultsIndexAndLength();
  unsigned tiedOperandsOffset = tiedOp.getTiedOperandsIndexAndLength().first;
  indices.resize(resultRange.second);
  for (unsigned i = 0; i < valueAttrs.size(); ++i) {
    int64_t index = valueAttrs[i].cast<IntegerAttr>().getInt();
    indices[i] = index != TiedOpInterface::kUntiedIndex
                     ? tiedOperandsOffset + index
                     : TiedOpInterface::kUntiedIndex;
  }
  return indices;
}

// static
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

// static
bool TiedOpInterface::hasAnyTiedUses(Value value) {
  for (auto &use : value.getUses()) {
    auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(use.getOwner());
    if (!tiedOp) continue;
    if (tiedOp.isOperandTied(use.getOperandNumber())) return true;
  }
  return false;
}

bool detail::isOperandTied(Operation *op, unsigned operandIndex) {
  auto tiedOp = dyn_cast<TiedOpInterface>(op);
  if (!tiedOp) return false;
  SmallVector<Value> results;
  auto tiedIndices = tiedOp.getTiedResultOperandIndices();
  for (unsigned i = 0; i < tiedIndices.size(); ++i) {
    if (tiedIndices[i] == operandIndex) {
      return true;
    }
  }
  return false;
}

SmallVector<Value> detail::getOperandTiedResults(Operation *op,
                                                 unsigned operandIndex) {
  auto tiedOp = dyn_cast<TiedOpInterface>(op);
  if (!tiedOp) return {};
  auto resultRange = tiedOp.getTiedResultsIndexAndLength();
  SmallVector<Value> results;
  auto tiedIndices = tiedOp.getTiedResultOperandIndices();
  for (unsigned i = 0; i < tiedIndices.size(); ++i) {
    if (tiedIndices[i] == operandIndex) {
      results.push_back(op->getResult(resultRange.first + i));
    }
  }
  return results;
}

LogicalResult detail::verifyTiedOp(TiedOpInterface tiedOp) {
  auto tiedOperandIndices = tiedOp.getTiedResultOperandIndices();
  if (tiedOperandIndices.empty()) return success();
  auto resultRange = tiedOp.getTiedResultsIndexAndLength();
  if (tiedOperandIndices.size() != resultRange.second) {
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

//===----------------------------------------------------------------------===//
// IREE::Util::SizeAwareTypeInterface
//===----------------------------------------------------------------------===//

static bool isValueUsableForOp(Value value, Operation *forOp) {
  if (forOp->getBlock() == nullptr) {
    // Op is not in a block; can't analyze (maybe?).
    return false;
  }
  auto *definingBlock = value.getParentBlock();
  if (definingBlock == forOp->getBlock()) {
    // Defined in the same block; ensure block order.
    if (value.isa<BlockArgument>()) return true;
    if (value.getDefiningOp()->isBeforeInBlock(forOp)) return true;
  } else if (definingBlock->isEntryBlock()) {
    // Entry block always dominates - fast path for constants.
    return true;
  } else {
    // See if block the value is defined in dominates the forOp block.
    // TODO(benvanik): optimize this, it's terribly expensive to recompute.
    DominanceInfo dominanceInfo(forOp->getParentOp());
    return dominanceInfo.dominates(definingBlock, forOp->getBlock());
  }
  return false;
}

// static
Value SizeAwareTypeInterface::findSizeValue(Value resourceValue,
                                            Operation *forOp) {
  // See if the value is produced by a size-aware op; we can just ask for the
  // size it has tied. Walking upward is always good as we know any size we find
  // dominates |forOp|.
  SmallVector<Value> worklist;
  worklist.push_back(resourceValue);
  while (!worklist.empty()) {
    auto value = worklist.pop_back_val();
    auto *definingOp = value.getDefiningOp();
    if (!definingOp) continue;
    if (auto sizeAwareOp =
            llvm::dyn_cast<IREE::Util::SizeAwareOpInterface>(definingOp)) {
      return sizeAwareOp.getResultSizeFromValue(value);
    }
    if (auto tiedOp = llvm::dyn_cast<IREE::Util::TiedOpInterface>(definingOp)) {
      auto tiedOperand = tiedOp.getTiedResultOperand(value);
      if (tiedOperand) worklist.push_back(tiedOperand);
    }
  }

  // Walk the users to see if any can report the size.
  worklist.push_back(resourceValue);
  while (!worklist.empty()) {
    auto value = worklist.pop_back_val();
    for (auto &use : value.getUses()) {
      if (auto sizeAwareOp = llvm::dyn_cast<IREE::Util::SizeAwareOpInterface>(
              use.getOwner())) {
        auto sizeValue = sizeAwareOp.getOperandSize(use.getOperandNumber());
        if (sizeValue) {
          if (isValueUsableForOp(sizeValue, forOp)) return sizeValue;
        }
      }
      if (auto tiedOp =
              llvm::dyn_cast<IREE::Util::TiedOpInterface>(use.getOwner())) {
        worklist.append(tiedOp.getOperandTiedResults(use.getOperandNumber()));
      }
    }
  }

  return {};
}

// static
Value SizeAwareTypeInterface::queryValueSize(Location loc, Value resourceValue,
                                             OpBuilder &builder) {
  auto sizeAwareType =
      resourceValue.getType().dyn_cast<IREE::Util::SizeAwareTypeInterface>();
  if (!sizeAwareType) {
    return {};  // Not a sized type.
  }
  if (!builder.getInsertionPoint().getNodePtr()->isKnownSentinel()) {
    Operation &insertionPt = *builder.getInsertionPoint();
    auto sizeValue = sizeAwareType.findSizeValue(resourceValue, &insertionPt);
    if (sizeValue) {
      return sizeValue;  // Found in IR.
    }
  }
  // TODO(benvanik): make this cleaner.
  auto *definingOp = resourceValue.getDefiningOp();
  if (auto sizeAwareOp =
          llvm::dyn_cast_or_null<IREE::Util::SizeAwareOpInterface>(
              definingOp)) {
    return sizeAwareOp.getResultSizeFromValue(resourceValue);
  } else if (auto inferSizeType =
                 resourceValue.getType()
                     .dyn_cast<IREE::Util::InferTypeSizeInterface>()) {
    return inferSizeType.inferSizeFromValue(loc, resourceValue, builder);
  }
  return {};
}

//===----------------------------------------------------------------------===//
// IREE::Util::UtilDialect
//===----------------------------------------------------------------------===//

// At the end so it can use functions above:
#include "iree/compiler/Dialect/Util/IR/UtilOpInterfaces.cpp.inc"
#include "iree/compiler/Dialect/Util/IR/UtilTypeInterfaces.cpp.inc"

void UtilDialect::registerTypes() {
  addTypes<IREE::Util::ByteBufferType, IREE::Util::ListType,
           IREE::Util::MutableByteBufferType, IREE::Util::PtrType,
           IREE::Util::VariantType>();
}

//===----------------------------------------------------------------------===//
// Type printing and parsing
//===----------------------------------------------------------------------===//

Type UtilDialect::parseType(DialectAsmParser &parser) const {
  Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());
  llvm::StringRef spec = parser.getFullSymbolSpec();
  if (spec == "variant") {
    return IREE::Util::VariantType::get(getContext());
  } else if (spec.consume_front("ptr")) {
    if (!spec.consume_front("<") || !spec.consume_back(">")) {
      parser.emitError(parser.getCurrentLocation())
          << "malformed ptr type '" << parser.getFullSymbolSpec() << "'";
      return Type();
    }
    auto variableType = mlir::parseType(spec, getContext());
    if (!variableType) {
      parser.emitError(parser.getCurrentLocation())
          << "invalid ptr object type specification: '"
          << parser.getFullSymbolSpec() << "'";
      return Type();
    }
    return IREE::Util::PtrType::getChecked(variableType, loc);
  } else if (spec == "byte_buffer") {
    return IREE::Util::ByteBufferType::get(getContext());
  } else if (spec == "mutable_byte_buffer") {
    return IREE::Util::MutableByteBufferType::get(getContext());
  } else if (spec.consume_front("list")) {
    if (!spec.consume_front("<") || !spec.consume_back(">")) {
      parser.emitError(parser.getCurrentLocation())
          << "malformed list type '" << parser.getFullSymbolSpec() << "'";
      return Type();
    }
    Type elementType;
    if (spec == "?") {
      elementType = IREE::Util::VariantType::get(getContext());
    } else {
      elementType = mlir::parseType(spec, getContext());
    }
    if (!elementType) {
      parser.emitError(parser.getCurrentLocation())
          << "invalid list element type specification: '"
          << parser.getFullSymbolSpec() << "'";
      return Type();
    }
    return IREE::Util::ListType::getChecked(elementType, loc);
  }
  emitError(loc, "unknown IREE type: ") << spec;
  return Type();
}

void UtilDialect::printType(Type type, DialectAsmPrinter &os) const {
  if (type.isa<IREE::Util::VariantType>()) {
    os << "variant";
  } else if (auto ptrType = type.dyn_cast<IREE::Util::PtrType>()) {
    os << "ptr<" << ptrType.getTargetType() << ">";
  } else if (type.isa<IREE::Util::ByteBufferType>()) {
    os << "byte_buffer";
  } else if (type.isa<IREE::Util::MutableByteBufferType>()) {
    os << "mutable_byte_buffer";
  } else if (auto listType = type.dyn_cast<IREE::Util::ListType>()) {
    os << "list<";
    if (listType.getElementType().isa<IREE::Util::VariantType>()) {
      os << "?";
    } else {
      os << listType.getElementType();
    }
    os << ">";
  } else {
    llvm_unreachable("unhandled IREE type");
  }
}

}  // namespace Util
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
