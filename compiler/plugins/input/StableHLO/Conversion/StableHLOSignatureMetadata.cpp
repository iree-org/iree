// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/StableHLO/Conversion/StableHLOSignatureMetadata.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler::stablehlo {
namespace {

struct SlotExpansion {
  SmallVector<int64_t> offsets;
  SmallVector<int64_t> counts;
};

static FailureOr<SlotExpansion>
buildSlotExpansion(TypeRange types, const TypeConverter &converter) {
  SlotExpansion expansion;
  int64_t offset = 0;
  for (Type type : types) {
    SmallVector<Type> converted;
    if (failed(converter.convertType(type, converted))) {
      return failure();
    }
    expansion.offsets.push_back(offset);
    expansion.counts.push_back(converted.size());
    offset += converted.size();
  }
  return expansion;
}

static bool isExpandedSlot(Type type, int64_t count) {
  return isa<TupleType>(type) || count != 1;
}

static LogicalResult remapSlotReference(Operation *op, StringRef name,
                                        const SlotExpansion &targets,
                                        TypeRange targetTypes,
                                        NamedAttrList &attrs) {
  Attribute attribute = attrs.get(name);
  if (!attribute) {
    return success();
  }
  auto index = dyn_cast<IntegerAttr>(attribute);
  if (!index || !index.getValue().isSignedIntN(64)) {
    return op->emitError() << "'" << name << "' must hold a valid slot index";
  }
  int64_t slot = index.getInt();
  if (slot < 0 || slot >= static_cast<int64_t>(targetTypes.size())) {
    return op->emitError() << "'" << name << "' slot index is out of range";
  }
  if (isExpandedSlot(targetTypes[slot], targets.counts[slot])) {
    return op->emitError() << "'" << name << "' cannot refer to "
                           << (isa<TupleType>(targetTypes[slot])
                                   ? "a tuple slot"
                                   : "an expanded signature slot");
  }
  if (!APInt(64, targets.offsets[slot])
           .isSignedIntN(index.getValue().getBitWidth())) {
    return op->emitError() << "remapped '" << name
                           << "' does not fit its attribute type";
  }
  attrs.set(name, IntegerAttr::get(index.getType(), targets.offsets[slot]));
  return success();
}

// Expanded slots may carry only an empty dictionary. A scalar slot retains its
// attributes, with any cross-signature references updated to the new offsets.
static FailureOr<DictionaryAttr>
remapSlotAttrs(Operation *op, Attribute attribute, Type type, int64_t count,
               TypeRange targetTypes, const SlotExpansion &targets,
               ArrayRef<StringRef> references) {
  auto dictionary = dyn_cast<DictionaryAttr>(attribute);
  if (!dictionary) {
    return op->emitError("slot attributes must be dictionaries");
  }
  if (isExpandedSlot(type, count)) {
    if (!dictionary.empty()) {
      return op->emitError()
             << "attributes on "
             << (isa<TupleType>(type) ? "a tuple slot"
                                      : "an expanded signature slot")
             << " are unsupported";
    }
    return dictionary;
  }
  NamedAttrList remapped(dictionary);
  for (StringRef name : references) {
    if (failed(remapSlotReference(op, name, targets, targetTypes, remapped))) {
      return failure();
    }
  }
  return remapped.getDictionary(op->getContext());
}

static FailureOr<ArrayAttr>
remapValueAttrs(Operation *op, ArrayAttr attributes, TypeRange types,
                const SlotExpansion &slots, TypeRange targetTypes,
                const SlotExpansion &targets, ArrayRef<StringRef> references) {
  if (!attributes) {
    return ArrayAttr{};
  }
  if (attributes.size() != types.size()) {
    return op->emitError("slot attribute count must match the signature");
  }
  SmallVector<Attribute> result;
  for (auto [type, attribute, count] :
       llvm::zip_equal(types, attributes, slots.counts)) {
    auto remapped = remapSlotAttrs(op, attribute, type, count, targetTypes,
                                   targets, references);
    if (failed(remapped)) {
      return failure();
    }
    result.append(count, *remapped);
  }
  return ArrayAttr::get(op->getContext(), result);
}

static FailureOr<IntegerAttr>
remapTiedOperand(Operation *op, Attribute attribute, Type resultType,
                 int64_t resultCount, TypeRange inputTypes,
                 const SlotExpansion &inputs) {
  auto tie = dyn_cast<IntegerAttr>(attribute);
  if (!tie || !tie.getValue().isSignedIntN(64)) {
    return op->emitError(
        "'iree.abi.tied_operands' must hold valid operand indices");
  }
  int64_t operand = tie.getInt();
  if (operand == -1) {
    return tie;
  }
  if (operand < 0 || operand >= static_cast<int64_t>(inputTypes.size())) {
    return op->emitError(
        "'iree.abi.tied_operands' operand index is out of range");
  }
  if (isExpandedSlot(resultType, resultCount) ||
      isExpandedSlot(inputTypes[operand], inputs.counts[operand])) {
    return op->emitError("'iree.abi.tied_operands' cannot refer to an "
                         "expanded signature slot");
  }
  if (!APInt(64, inputs.offsets[operand])
           .isSignedIntN(tie.getValue().getBitWidth())) {
    return op->emitError("remapped 'iree.abi.tied_operands' index does not "
                         "fit its attribute type");
  }
  return IntegerAttr::get(tie.getType(), inputs.offsets[operand]);
}

static FailureOr<ArrayAttr> remapTiedOperands(Operation *op, ArrayAttr ties,
                                              FunctionType type,
                                              const SlotExpansion &inputs,
                                              const SlotExpansion &results) {
  if (!ties) {
    return ArrayAttr{};
  }
  if (ties.size() != type.getNumResults()) {
    return op->emitError(
        "'iree.abi.tied_operands' must have one entry per result");
  }
  SmallVector<Attribute> remapped;
  for (auto [resultType, attribute, count] :
       llvm::zip_equal(type.getResults(), ties, results.counts)) {
    auto tie = remapTiedOperand(op, attribute, resultType, count,
                                type.getInputs(), inputs);
    if (failed(tie)) {
      return failure();
    }
    remapped.append(count, *tie);
  }
  return ArrayAttr::get(op->getContext(), remapped);
}

} // namespace

FailureOr<SignatureMetadata>
remapSignatureMetadata(Operation *op, FunctionType type,
                       SignatureMetadata metadata,
                       const TypeConverter &converter) {
  auto inputs = buildSlotExpansion(type.getInputs(), converter);
  auto results = buildSlotExpansion(type.getResults(), converter);
  if (failed(inputs) || failed(results)) {
    return op->emitError("signature type conversion failed");
  }
  auto argAttrs = remapValueAttrs(op, metadata.argAttrs, type.getInputs(),
                                  *inputs, type.getResults(), *results,
                                  {"tf.aliasing_output", "iree.abi.output"});
  if (failed(argAttrs)) {
    return failure();
  }
  auto resultAttrs =
      remapValueAttrs(op, metadata.resultAttrs, type.getResults(), *results,
                      type.getInputs(), *inputs, {"iree.abi.tied"});
  if (failed(resultAttrs)) {
    return failure();
  }
  auto ties =
      remapTiedOperands(op, metadata.tiedOperands, type, *inputs, *results);
  if (failed(ties)) {
    return failure();
  }
  return SignatureMetadata{*argAttrs, *resultAttrs, *ties};
}

} // namespace mlir::iree_compiler::stablehlo
