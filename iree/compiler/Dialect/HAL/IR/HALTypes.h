// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_IR_HALTYPES_H_
#define IREE_COMPILER_DIALECT_HAL_IR_HALTYPES_H_

#include <cstdint>

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
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

// clang-format off: must be included after all LLVM/MLIR headers.
#include "iree/compiler/Dialect/HAL/IR/HALEnums.h.inc"    // IWYU pragma: keep
#include "iree/compiler/Dialect/HAL/IR/HALStructs.h.inc"  // IWYU pragma: keep
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
llvm::Optional<int32_t> getElementTypeValue(Type type);

// Returns a stable identifier for the MLIR encoding type or 0 (opaque) if the
// type is unsupported in the ABI.
llvm::Optional<int32_t> getEncodingTypeValue(Attribute attr);

template <typename T>
inline bool allEnumBitsSet(T value, T required) {
  return (static_cast<uint32_t>(value) & static_cast<uint32_t>(required)) ==
         static_cast<uint32_t>(required);
}

//===----------------------------------------------------------------------===//
// Object types
//===----------------------------------------------------------------------===//

class AllocatorType : public Type::TypeBase<AllocatorType, Type, TypeStorage> {
 public:
  using Base::Base;
};

class BufferType
    : public Type::TypeBase<BufferType, Type, TypeStorage,
                            IREE::Util::InferTypeSizeInterface::Trait,
                            IREE::Util::ReferenceTypeInterface::Trait> {
 public:
  using Base::Base;

  Value inferSizeFromValue(Location loc, Value value, OpBuilder &builder) const;
};

class BufferViewType
    : public Type::TypeBase<BufferViewType, Type, TypeStorage,
                            IREE::Util::InferTypeSizeInterface::Trait,
                            IREE::Util::ReferenceTypeInterface::Trait> {
 public:
  using Base::Base;

  Value inferSizeFromValue(Location loc, Value value, OpBuilder &builder) const;
};

class CommandBufferType
    : public Type::TypeBase<CommandBufferType, Type, TypeStorage> {
 public:
  using Base::Base;
};

class DescriptorSetType
    : public Type::TypeBase<DescriptorSetType, Type, TypeStorage> {
 public:
  using Base::Base;
};

class DescriptorSetLayoutType
    : public Type::TypeBase<DescriptorSetLayoutType, Type, TypeStorage> {
 public:
  using Base::Base;
};

class DeviceType : public Type::TypeBase<DeviceType, Type, TypeStorage> {
 public:
  using Base::Base;
};

class EventType : public Type::TypeBase<EventType, Type, TypeStorage> {
 public:
  using Base::Base;
};

class ExecutableType
    : public Type::TypeBase<ExecutableType, Type, TypeStorage> {
 public:
  using Base::Base;
};

class ExecutableLayoutType
    : public Type::TypeBase<ExecutableLayoutType, Type, TypeStorage> {
 public:
  using Base::Base;
};

class RingBufferType
    : public Type::TypeBase<RingBufferType, Type, TypeStorage> {
 public:
  using Base::Base;
};

class SemaphoreType : public Type::TypeBase<SemaphoreType, Type, TypeStorage> {
 public:
  using Base::Base;
};

//===----------------------------------------------------------------------===//
// Struct types
//===----------------------------------------------------------------------===//

// Returns the intersection (most conservative) constraints |lhs| ∩ |rhs|.
BufferConstraintsAttr intersectBufferConstraints(BufferConstraintsAttr lhs,
                                                 BufferConstraintsAttr rhs);

// TODO(benvanik): runtime buffer constraint queries from the allocator.
// We can add folders for those when the allocator is strongly-typed with
// #hal.buffer_constraints and otherwise leave them for runtime queries.
class BufferConstraintsAdaptor {
 public:
  BufferConstraintsAdaptor(Location loc, Value allocator);

  Value getMaxAllocationSize(OpBuilder &builder);
  Value getMinBufferOffsetAlignment(OpBuilder &builder);
  Value getMaxBufferRange(OpBuilder &builder);
  Value getMinBufferRangeAlignment(OpBuilder &builder);

 private:
  Location loc_;
  Value allocator_;
  BufferConstraintsAttr bufferConstraints_;
};

// A tuple containing runtime values for a descriptor set binding.
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

// clang-format off: must be included after all LLVM/MLIR headers.
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/HAL/IR/HALAttrs.h.inc"  // IWYU pragma: keep
// clang-format on

#endif  // IREE_COMPILER_DIALECT_HAL_IR_HALTYPES_H_
