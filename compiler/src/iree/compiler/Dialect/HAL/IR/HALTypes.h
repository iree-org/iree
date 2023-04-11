// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_IR_HALTYPES_H_
#define IREE_COMPILER_DIALECT_HAL_IR_HALTYPES_H_

#include <cstdint>

#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

// clang-format off: must be included after all LLVM/MLIR headers.
#include "iree/compiler/Dialect/HAL/IR/HALEnums.h.inc"  // IWYU pragma: keep
// clang-format on

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

#include "iree/compiler/Dialect/HAL/IR/HALAttrInterfaces.h.inc"  // IWYU pragma: export
#include "iree/compiler/Dialect/HAL/IR/HALOpInterfaces.h.inc"  // IWYU pragma: export
#include "iree/compiler/Dialect/HAL/IR/HALTypeInterfaces.h.inc"  // IWYU pragma: export

//===----------------------------------------------------------------------===//
// Enum utilities
//===----------------------------------------------------------------------===//

// Returns a stable identifier for the MLIR element type or nullopt if the
// type is unsupported in the ABI.
std::optional<int32_t> getElementTypeValue(Type type);

// Returns a stable identifier for the MLIR encoding type or 0 (opaque) if the
// type is unsupported in the ABI.
std::optional<int32_t> getEncodingTypeValue(Attribute attr);

template <typename T>
inline bool allEnumBitsSet(T value, T required) {
  return (static_cast<uint32_t>(value) & static_cast<uint32_t>(required)) ==
         static_cast<uint32_t>(required);
}

//===----------------------------------------------------------------------===//
// Alignment
//===----------------------------------------------------------------------===//

// Returns the common (minimum) alignment between |lhs| and |rhs| or nullopt if
// either is unaligned.
llvm::MaybeAlign commonAlignment(llvm::MaybeAlign lhs, llvm::MaybeAlign rhs);

// Returns either the constant offset |value| or the alignment of the offset
// inferred from the IR. Returns nullopt if no alignment is available.
std::optional<uint64_t> lookupOffsetOrAlignment(Value value);

//===----------------------------------------------------------------------===//
// Object types
//===----------------------------------------------------------------------===//

struct AllocatorType : public Type::TypeBase<AllocatorType, Type, TypeStorage> {
  using Base::Base;
};

struct BufferType
    : public Type::TypeBase<BufferType, Type, TypeStorage,
                            IREE::Util::InferTypeSizeInterface::Trait,
                            IREE::Util::ReferenceTypeInterface::Trait> {
  using Base::Base;

  Value inferSizeFromValue(Location loc, Value value, OpBuilder &builder) const;
};

struct BufferViewType
    : public Type::TypeBase<BufferViewType, Type, TypeStorage,
                            IREE::Util::InferTypeSizeInterface::Trait,
                            IREE::Util::ReferenceTypeInterface::Trait> {
  using Base::Base;

  Value inferSizeFromValue(Location loc, Value value, OpBuilder &builder) const;
};

struct ChannelType : public Type::TypeBase<ChannelType, Type, TypeStorage> {
  using Base::Base;
};

struct CommandBufferType
    : public Type::TypeBase<CommandBufferType, Type, TypeStorage> {
  using Base::Base;
};

struct DescriptorSetLayoutType
    : public Type::TypeBase<DescriptorSetLayoutType, Type, TypeStorage> {
  using Base::Base;
};

struct DeviceType : public Type::TypeBase<DeviceType, Type, TypeStorage> {
  using Base::Base;
};

struct EventType : public Type::TypeBase<EventType, Type, TypeStorage> {
  using Base::Base;
};

struct ExecutableType
    : public Type::TypeBase<ExecutableType, Type, TypeStorage> {
  using Base::Base;
};

struct PipelineLayoutType
    : public Type::TypeBase<PipelineLayoutType, Type, TypeStorage> {
  using Base::Base;
};

struct FenceType : public Type::TypeBase<FenceType, Type, TypeStorage> {
  using Base::Base;
};

struct RingBufferType
    : public Type::TypeBase<RingBufferType, Type, TypeStorage> {
  using Base::Base;
};

struct SemaphoreType : public Type::TypeBase<SemaphoreType, Type, TypeStorage> {
  using Base::Base;
};

//===----------------------------------------------------------------------===//
// Struct types
//===----------------------------------------------------------------------===//

// A tuple containing runtime values for a descriptor set binding.
// The buffer specified may be either a !hal.buffer or an index of a binding
// table slot to source the buffer from.
struct DescriptorSetBindingValue {
  Value ordinal;
  Value buffer;
  Value byteOffset;
  Value byteLength;
};

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

// It's unfortunate this is required.
namespace mlir {

template <>
struct FieldParser<
    std::optional<mlir::iree_compiler::IREE::HAL::CollectiveReductionOp>> {
  static FailureOr<mlir::iree_compiler::IREE::HAL::CollectiveReductionOp> parse(
      AsmParser &parser) {
    std::string value;
    if (parser.parseKeywordOrString(&value)) return failure();
    auto result = mlir::iree_compiler::IREE::HAL::symbolizeEnum<
        mlir::iree_compiler::IREE::HAL::CollectiveReductionOp>(value);
    if (!result.has_value()) return failure();
    return result.value();
  }
};
static inline AsmPrinter &operator<<(
    AsmPrinter &printer,
    std::optional<mlir::iree_compiler::IREE::HAL::CollectiveReductionOp>
        param) {
  printer << (param.has_value()
                  ? mlir::iree_compiler::IREE::HAL::stringifyEnum(param.value())
                  : StringRef{""});
  return printer;
}

template <>
struct FieldParser<
    std::optional<mlir::iree_compiler::IREE::HAL::DescriptorFlags>> {
  static FailureOr<mlir::iree_compiler::IREE::HAL::DescriptorFlags> parse(
      AsmParser &parser) {
    std::string value;
    if (parser.parseKeywordOrString(&value)) return failure();
    auto result = mlir::iree_compiler::IREE::HAL::symbolizeEnum<
        mlir::iree_compiler::IREE::HAL::DescriptorFlags>(value);
    if (!result.has_value()) return failure();
    return result.value();
  }
};
static inline AsmPrinter &operator<<(
    AsmPrinter &printer,
    std::optional<mlir::iree_compiler::IREE::HAL::DescriptorFlags> param) {
  printer << (param.has_value()
                  ? mlir::iree_compiler::IREE::HAL::stringifyEnum(param.value())
                  : StringRef{""});
  return printer;
}

static inline AsmPrinter &operator<<(
    AsmPrinter &printer, mlir::iree_compiler::IREE::HAL::DescriptorType param) {
  printer << mlir::iree_compiler::IREE::HAL::stringifyEnum(param);
  return printer;
}

}  // namespace mlir

// clang-format off: must be included after all LLVM/MLIR headers.
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/HAL/IR/HALAttrs.h.inc"  // IWYU pragma: keep
// clang-format on

#endif  // IREE_COMPILER_DIALECT_HAL_IR_HALTYPES_H_
