// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_UTIL_IR_UTILTYPES_H_
#define IREE_COMPILER_DIALECT_UTIL_IR_UTILTYPES_H_

#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/CallInterfaces.h"

#include <numeric>

// clang-format off: must be included after all LLVM/MLIR headers.
#include "iree/compiler/Dialect/Util/IR/UtilEnums.h.inc" // IWYU pragma: keep
// clang-format on

namespace mlir::iree_compiler::IREE::Util {

class GlobalOpInterface;
class GlobalAccessorOpInterface;
class GlobalAddressOpInterface;
class GlobalLoadOpInterface;
class GlobalStoreOpInterface;
class ShapeAwareOpInterface;
class TiedOpInterface;

//===----------------------------------------------------------------------===//
// Common types
//===----------------------------------------------------------------------===//

// Status code table mapping to iree::StatusCode in the runtime.
enum class StatusCode : int32_t {
  Ok = 0,
  Cancelled = 1,
  Unknown = 2,
  InvalidArgument = 3,
  DeadlineExceeded = 4,
  NotFound = 5,
  AlreadyExists = 6,
  PermissionDenied = 7,
  ResourceExhausted = 8,
  FailedPrecondition = 9,
  Aborted = 10,
  OutOfRange = 11,
  Unimplemented = 12,
  Internal = 13,
  Unavailable = 14,
  DataLoss = 15,
  Unauthenticated = 16,
  Deferred = 17,
  Incompatible = 18,
  DoNotUseReservedForFutureExpansionUseDefaultInSwitchInstead_ = 20
};

struct ValueAccess {
  bool isRead : 1;
  bool isWrite : 1;
  bool isDiscard : 1;
  bool isNone() const { return !isRead && !isWrite && !isDiscard; }
  bool isReadOnly() const { return isRead && !isWrite && !isDiscard; }
  ValueAccess() : isRead(false), isWrite(false), isDiscard(false) {}
  ValueAccess(bool isRead, bool isWrite, bool isDiscard)
      : isRead(isRead), isWrite(isWrite), isDiscard(isDiscard) {}
  static ValueAccess None() { return ValueAccess(false, false, false); }
  static ValueAccess ReadOnly() { return ValueAccess(true, false, false); }
  static ValueAccess ReadWrite() { return ValueAccess(true, true, false); }
  static ValueAccess WriteOnly() { return ValueAccess(false, true, false); }
  static ValueAccess DiscardWrite() { return ValueAccess(false, true, true); }
};

// An (offset, length) range within a size-aware resource.
struct SubrangeOperand {
  // Base resource the subrange references into.
  Value resource;
  // Size of the full base resource.
  Value resourceSize;
  // Offset into the base resource the range begins.
  Value offset;
  // Total length of the range within the base resource.
  Value length;
};

//===----------------------------------------------------------------------===//
// Op utilities common in util patterns and folders
//===----------------------------------------------------------------------===//

// Returns true if |value| can be used by the operation at the insertion point.
bool isValueUsableForOp(Value value, Block *block,
                        Block::iterator insertionPoint);
// Returns true if |value| can be used by |op|.
bool isValueUsableForOp(Value value, Operation *op);

// Tries to reorder the producer of |value| above |consumerOp|.
// Returns true if the move was successful.
bool tryMoveProducerBefore(Value value, Operation *consumerOp);

// Returns true if the given callable op is public or external (no body).
// Such callables cannot have their signature changed without (potentially)
// breaking linking.
bool isPublicOrExternal(CallableOpInterface callableOp);

//===----------------------------------------------------------------------===//
// Global and structural interface utilities
//===----------------------------------------------------------------------===//

namespace detail {

LogicalResult verifyGlobalOp(GlobalOpInterface globalOp);
LogicalResult verifyGlobalAddressOp(GlobalAddressOpInterface addressOp,
                                    SymbolTableCollection &symbolTable);
LogicalResult verifyGlobalLoadOp(GlobalLoadOpInterface loadOp,
                                 SymbolTableCollection &symbolTable);
LogicalResult verifyGlobalStoreOp(GlobalStoreOpInterface storeOp,
                                  SymbolTableCollection &symbolTable);

} // namespace detail

IREE::Util::GlobalOpInterface
lookupGlobalOp(Operation *accessorOp, SymbolRefAttr globalRefAttr,
               SymbolTableCollection &symbolTable);

//===----------------------------------------------------------------------===//
// Tied operand interface utilities
//===----------------------------------------------------------------------===//

namespace detail {

void getAllTiedOperands(Operation *op, SmallVectorImpl<int64_t> &indices);
std::optional<unsigned> getTiedResultOperandIndex(Operation *op,
                                                  unsigned resultIndex);
void setTiedResultOperandIndex(Operation *op, unsigned resultIndex,
                               std::optional<unsigned> operandIndex);
SmallVector<int64_t> getTiedResultOperandIndices(Operation *op);
bool isOperandTied(Operation *tiedOp, unsigned operandIndex);
SmallVector<Value> getOperandTiedResults(Operation *op, unsigned operandIndex);
LogicalResult verifyTiedOp(TiedOpInterface tiedOp);

} // namespace detail

// Resets or removes the indices in |tiedOperandIndices| based on the given
// exclusion lists.
void excludeTiedOperandAndResultIndices(
    ArrayRef<unsigned> excludedOperandIndices,
    ArrayRef<unsigned> excludedResultIndices,
    SmallVector<int64_t> &tiedOperandIndices);

//===----------------------------------------------------------------------===//
// Forward defines for InferIntDivisibilityOpInterface
// See implementations in IntegerDivisibility.h.
//===----------------------------------------------------------------------===//

class ConstantIntDivisibility {
public:
  ConstantIntDivisibility() = default;
  ConstantIntDivisibility(uint64_t udiv, uint64_t sdiv)
      : udivVal(udiv), sdivVal(sdiv) {}

  bool operator==(const ConstantIntDivisibility &other) const {
    return udivVal == other.udivVal && sdivVal == other.sdivVal;
  }

  uint64_t udiv() const { return this->udivVal; }
  uint64_t sdiv() const { return this->sdivVal; }

  // Returns the union (computed separately for signed and unsigned bounds)
  // for this range and `other`.
  ConstantIntDivisibility getUnion(const ConstantIntDivisibility &other) const {
    return ConstantIntDivisibility(
        /*udiv=*/std::gcd(udiv(), other.udiv()),
        /*sdiv=*/std::gcd(sdiv(), other.sdiv()));
  }

private:
  uint64_t udivVal;
  uint64_t sdivVal;

  friend raw_ostream &operator<<(raw_ostream &os,
                                 const ConstantIntDivisibility &div);
};

inline raw_ostream &operator<<(raw_ostream &os,
                               const ConstantIntDivisibility &div) {
  os << "ConstantIntDivisibility(udiv = " << div.udivVal
     << ", sdiv = " << div.sdivVal << ")";
  return os;
}

class IntegerDivisibility {
public:
  IntegerDivisibility(ConstantIntDivisibility value)
      : value(std::move(value)) {}
  IntegerDivisibility(
      std::optional<ConstantIntDivisibility> value = std::nullopt)
      : value(std::move(value)) {}
  // Gets the minimum divisibility of 1 that is used to indicate that the value
  // cannot be analyzed further.
  static IntegerDivisibility getMinDivisibility() {
    return IntegerDivisibility(ConstantIntDivisibility(1, 1));
  }

  bool isUninitialized() const { return !value.has_value(); }
  const ConstantIntDivisibility &getValue() const {
    assert(!isUninitialized());
    return *value;
  }

  bool operator==(const IntegerDivisibility &rhs) const {
    return value == rhs.value;
  }

  static IntegerDivisibility join(const IntegerDivisibility &lhs,
                                  const IntegerDivisibility &rhs) {
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;
    return IntegerDivisibility(lhs.getValue().getUnion(rhs.getValue()));
  }

  void print(raw_ostream &os) const { os << value; }

private:
  std::optional<ConstantIntDivisibility> value;
};

inline raw_ostream &operator<<(raw_ostream &os,
                               const IntegerDivisibility &div) {
  div.print(os);
  return os;
}

using SetIntDivisibilityFn =
    llvm::function_ref<void(Value, const ConstantIntDivisibility &)>;

//===----------------------------------------------------------------------===//
// Shape-aware interface utilities
//===----------------------------------------------------------------------===//

// Walks the SSA use-def chain upwards to find the dynamic dimensions of the
// value. Returns None if the shape cannot be found.
std::optional<ValueRange> findDynamicDims(Value shapedValue);

// Walks the SSA use-def chain upwards to find the requested dimension of the
// value if the dimension is dynamic. Returns the static size if the dim is
// static, and null if the walk fails.
// NOTE: If querying more than one dimension prefer findDynamicDims instead
// of calling this multiple times for the same |shapedValue|.
OpFoldResult findDim(Value shapedValue, int64_t dim);

// Walks the SSA use-def chain to find the dynamic dimensions of the value.
// Returns None if the shape cannot be found or if it is defined after
// {|block|, |insertionPoint|}.
std::optional<ValueRange> findDynamicDims(Value shapedValue, Block *block,
                                          Block::iterator insertionPoint);

// Returns the dynamic dimensions for the value at |idx|.
// |dynamicDims| is zero or more dynamic dimensions corresponding to the
// |values| list of arbitrary types.
// Shaped types will return zero or more dynamic dimension values.
// Sized types will return exactly one value.
ValueRange findDynamicDimsInList(unsigned idx, ValueRange values,
                                 ValueRange dynamicDims);

// Returns the size of the size-aware typed value at |idx| in |values|.
// |dynamicDims| is zero or more dynamic dimensions corresponding to the
// |values| list of arbitrary types.
Value findValueSizeInList(unsigned idx, ValueRange values,
                          ValueRange dynamicDims);

// Returns dimension values for each dynamic dimension of the given |value|.
// |value| must be a ShapedType. The returned value range will be empty if the
// shape is fully static.
SmallVector<Value> buildDynamicDimsForValue(Location loc, Value value,
                                            OpBuilder &builder);

// Returns dimension values for each dynamic dimension of the given |values|.
// |values| must all have ShapedTypes. The returned value range will be empty if
// all shapes are fully static.
SmallVector<Value> buildDynamicDimsForValues(Location loc, ValueRange values,
                                             OpBuilder &builder);

// Builds a ranked shape with all dimension values for the given operand.
SmallVector<Value> buildOperandShape(ShapeAwareOpInterface op,
                                     unsigned operandIdx, OpBuilder &builder);

// Builds a ranked shape with all dimension values for the given result.
SmallVector<Value> buildResultShape(ShapeAwareOpInterface op,
                                    unsigned resultIdx, OpBuilder &builder);

//===----------------------------------------------------------------------===//
// Alignment and byte offset/length manipulation
//===----------------------------------------------------------------------===//

// Aligns |value| to |alignment|, rounding up if needed.
static inline uint64_t align(uint64_t value, uint64_t alignment) {
  return (value + (alignment - 1)) & ~(alignment - 1);
}
static inline uint64_t align(uint64_t value, const APInt &alignment) {
  return align(value, alignment.getZExtValue());
}

// Returns the bit-width of the scalar type. If the type is complex, it returns
// the type of individual elements * 2 (1 for real and 1 for complex).
static inline unsigned getTypeBitWidth(Type type) {
  if (auto complexType = dyn_cast<ComplexType>(type)) {
    return 2 * complexType.getElementType().getIntOrFloatBitWidth();
  }
  if (auto vectorType = dyn_cast<VectorType>(type)) {
    return vectorType.getNumElements() *
           getTypeBitWidth(vectorType.getElementType());
  }
  return type.getIntOrFloatBitWidth();
}

// HACK: we currently have no way to specify packing on types and as such have
// to guess (poorly) what physical storage for each type looks like. The
// heuristic for non-power-of-two bit width types is to take the next
// power-of-two byte width that can fit at least 2 elements, up to 64-bit.
static inline unsigned getTypePhysicalStorageBitWidth(Type type) {
  unsigned logicalBitWidth = getTypeBitWidth(type);
  unsigned desiredBitWidth = logicalBitWidth * 2;   // at least 2
  desiredBitWidth = std::min(desiredBitWidth, 64u); // no larger than 64-bits
  return llvm::PowerOf2Ceil(llvm::divideCeil(desiredBitWidth, 8)) * 8;
}

// Returns the number of bytes an element of the given type occupies in memory.
// This is in the default dense conversion to machine words where sizes must be
// powers of two aligned to bytes.
//
// Examples:
//   getRoundedElementByteWidth(i1) = 1
//   getRoundedElementByteWidth(i23) = 4
//   getRoundedElementByteWidth(i32) = 4
//   getRoundedElementByteWidth(bf16) = 2
//   getRoundedElementByteWidth(i33) = 8
//   getRoundedElementByteWidth(complex<f32>) = 8
static inline int32_t getRoundedElementByteWidth(Type type) {
  if (auto complexType = dyn_cast<ComplexType>(type)) {
    return 2 * getRoundedElementByteWidth(complexType.getElementType());
  }
  // TODO(ravishankarm): evaluate if this vector packing works with sub-byte
  // element types.
  if (auto vectorType = dyn_cast<VectorType>(type)) {
    return vectorType.getNumElements() *
           getRoundedElementByteWidth(vectorType.getElementType());
  }
  unsigned bitsUnaligned = type.getIntOrFloatBitWidth();
  assert(bitsUnaligned > 0 && "0-width types unsupported");
  // Round up to 8-bit aligned bytes.
  unsigned byteAligned = (bitsUnaligned + 8 - 1) / 8;
  // Round up to the next power of two (unless already a power of two).
  return llvm::PowerOf2Ceil(byteAligned);
}

// Returns the number of physical bytes required to store |elementCount|
// elements of |elementType| in a packed representation. This will include any
// additional padding required to round out the type to the next power-of-two
// machine word size.
//
// Examples:
//   getRoundedPhysicalStorageSize(1, i2) = 1 (1-byte aligned packed)
//   getRoundedPhysicalStorageSize(3, i3) = 2 (1-byte aligned packed)
//   getRoundedPhysicalStorageSize(4, i32) = 16 (4-byte aligned native)
//   getRoundedPhysicalStorageSize(4, i33) = 32 (8-byte aligned packed)
static inline int64_t getRoundedPhysicalStorageSize(int64_t elementCount,
                                                    Type elementType) {
  const unsigned logicalBitWidth = getTypeBitWidth(elementType);
  switch (logicalBitWidth) {
  case 1:
    return elementCount * sizeof(uint8_t);
  case 8:
  case 16:
  case 32:
  case 64:
    return elementCount * logicalBitWidth / 8;
  default:
    break; // sub-byte handling below
  }
  // Round up to the next power of two (unless already a power of two) of the
  // 8-bit aligned logical bit width.
  const unsigned physicalBitWidth = getTypePhysicalStorageBitWidth(elementType);
  const unsigned elementsPerPhysicalWord = physicalBitWidth / logicalBitWidth;
  const int64_t unalignedBitCount =
      llvm::divideCeil(elementCount, elementsPerPhysicalWord) *
      physicalBitWidth;
  return llvm::divideCeil(align(unalignedBitCount, physicalBitWidth), 8);
}
static inline int64_t getRoundedPhysicalStorageSize(ShapedType type) {
  return getRoundedPhysicalStorageSize(type.getNumElements(),
                                       type.getElementType());
}

} // namespace mlir::iree_compiler::IREE::Util

#include "iree/compiler/Dialect/Util/IR/UtilAttrInterfaces.h.inc" // IWYU pragma: export
#include "iree/compiler/Dialect/Util/IR/UtilOpInterfaces.h.inc" // IWYU pragma: export
#include "iree/compiler/Dialect/Util/IR/UtilTypeInterfaces.h.inc" // IWYU pragma: export

// clang-format off: must be included after all LLVM/MLIR headers.
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/Util/IR/UtilAttrs.h.inc" // IWYU pragma: keep
// clang-format on

// clang-format off: must be included after all LLVM/MLIR headers.
#define GET_TYPEDEF_CLASSES
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h.inc" // IWYU pragma: keep
// clang-format on

#endif // IREE_COMPILER_DIALECT_UTIL_IR_UTILTYPES_H_
