// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_IR_HALTYPES_H_
#define IREE_COMPILER_DIALECT_HAL_IR_HALTYPES_H_

#include <cstdint>
#include <optional>

#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilTraits.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
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
#include "iree/compiler/Dialect/HAL/IR/HALEnums.h.inc" // IWYU pragma: keep
#include "iree/compiler/Dialect/HAL/IR/HALOpInterfaces.h.inc" // IWYU pragma: export
// clang-format on

namespace mlir::iree_compiler::IREE::HAL {

class DeviceTargetAttr;
class TargetRegistry;

using BuildDeviceTargetMatchFn = std::function<Value(
    Location loc, Value device, IREE::HAL::DeviceTargetAttr targetAttr,
    OpBuilder &builder)>;

#include "iree/compiler/Dialect/HAL/IR/HALAttrInterfaces.h.inc" // IWYU pragma: export
#include "iree/compiler/Dialect/HAL/IR/HALTypeInterfaces.h.inc" // IWYU pragma: export

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

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

  static constexpr StringLiteral name = "hal.allocator";
};

struct BufferType
    : public Type::TypeBase<BufferType, Type, TypeStorage,
                            IREE::Util::InferTypeSizeInterface::Trait,
                            IREE::Util::ReferenceTypeInterface::Trait> {
  using Base::Base;

  static constexpr StringLiteral name = "hal.buffer";

  Value inferSizeFromValue(Location loc, Value value, OpBuilder &builder) const;
};

struct BufferViewType
    : public Type::TypeBase<BufferViewType, Type, TypeStorage,
                            IREE::Util::InferTypeSizeInterface::Trait,
                            IREE::Util::ReferenceTypeInterface::Trait> {
  using Base::Base;

  static constexpr StringLiteral name = "hal.buffer_view";

  Value inferSizeFromValue(Location loc, Value value, OpBuilder &builder) const;
};

struct ChannelType : public Type::TypeBase<ChannelType, Type, TypeStorage> {
  using Base::Base;

  static constexpr StringLiteral name = "hal.channel";
};

struct CommandBufferType
    : public Type::TypeBase<CommandBufferType, Type, TypeStorage> {
  using Base::Base;

  static constexpr StringLiteral name = "hal.command_buffer";
};

struct DeviceType
    : public Type::TypeBase<DeviceType, Type, TypeStorage,
                            mlir::OpTrait::IREE::Util::ImplicitlyCaptured,
                            IREE::Util::ReferenceTypeInterface::Trait> {
  using Base::Base;

  static constexpr StringLiteral name = "hal.device";

  // Resolves to any device at runtime.
  // This is unlikely to be what any particular caller wants and should be
  // avoided outside of testing/debugging passes that don't care about
  // multi-targeting.
  static Value resolveAny(Location loc, OpBuilder &builder);
};

struct EventType : public Type::TypeBase<EventType, Type, TypeStorage> {
  using Base::Base;

  static constexpr StringLiteral name = "hal.event";
};

struct ExecutableType
    : public Type::TypeBase<ExecutableType, Type, TypeStorage> {
  using Base::Base;

  static constexpr StringLiteral name = "hal.executable";
};

struct FenceType : public Type::TypeBase<FenceType, Type, TypeStorage> {
  using Base::Base;

  static constexpr StringLiteral name = "hal.fence";
};

struct FileType : public Type::TypeBase<FileType, Type, TypeStorage> {
  using Base::Base;

  static constexpr StringLiteral name = "hal.file";
};

struct SemaphoreType : public Type::TypeBase<SemaphoreType, Type, TypeStorage> {
  using Base::Base;

  static constexpr StringLiteral name = "hal.semaphore";
};

//===----------------------------------------------------------------------===//
// Struct list types
//===----------------------------------------------------------------------===//

// A tuple containing runtime values for a descriptor set binding.
// The buffer specified may be either a !hal.buffer or an index of a binding
// table slot to source the buffer from.
struct PipelineBindingValue {
  Value ordinal;
  Value buffer;
  Value byteOffset;
  Value byteLength;
};

// A tuple containing runtime values for a binding.
// The buffer specified may be either a !hal.buffer or an index of a binding
// table slot to source the buffer from.
struct BindingValue {
  Value buffer;
  Value byteOffset;
  Value byteLength;
};

template <typename T>
struct StaticRange {
  T min;
  T max;
  StaticRange(T value) : min(value), max(value) {}
  StaticRange(T min, T max) : min(min), max(max) {}
};

} // namespace mlir::iree_compiler::IREE::HAL

// clang-format off: must be included after all LLVM/MLIR headers.
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/HAL/IR/HALAttrs.h.inc" // IWYU pragma: keep
// clang-format on

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

namespace mlir::iree_compiler::IREE::HAL {

} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_DIALECT_HAL_IR_HALTYPES_H_
